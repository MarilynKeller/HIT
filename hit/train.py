import socket
print(f"Running on machine {socket.gethostname()}")

import glob
import os
import pickle
import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import tqdm
from training.logging import log_canonical_meshes, log_slices
import yaml
import wandb

from model.mysmpl import MySmpl
from omegaconf import OmegaConf
from skimage.io import imsave

from hit.model.hit_model import HITModel
import hit.hit_config as cg
import hit.training.metrics as metrics
from hit.training.losses import *
from hit.utils.tensors import cond_create
from hit.training.dataloader_mri import MRI_SHAPE, MRIDataset

class HITTraining(pl.LightningModule):
    def __init__(self, cfg, checkpoint_fct=None):
        super().__init__()
        self.cfg = cfg
        self.smpl_cfg = cfg.smpl_cfg
        self.data_cfg = cfg.data_cfg
        self.train_cfg = cfg.train_cfg
    
        if self.train_cfg['to_train'] == 'pretrain':
            pass
            
        elif self.train_cfg['to_train'] == 'occ':
            self.data_cfg.synthetic = False
            
        elif self.train_cfg['to_train'] == 'compression':
            self.data_cfg.synthetic = False
            self.train_cfg['optimize_disp'] = False
            self.train_cfg['optimize_occ'] = False
            self.train_cfg['compressor'] = True
            self.train_cfg['lambda_compression'] = 1
  
        else:
            raise NotImplementedError(f"Unknown training mode {self.train_cfg['to_train']}")
    
        self.smpl = MySmpl(model_path=cg.smplx_models_path, gender=self.cfg.smpl_cfg.gender)
        self.hit = HITModel(train_cfg=self.cfg.train_cfg, smpl=self.smpl)
        self.hit.initialize(pretrained=False, train_cfg = self.hit.train_cfg)

        
        self.max_queries = self.train_cfg['max_queries'] # to prevent OOM at inference time
        
        print(self.hit.tissues_occ) 
        
        if cfg.seed_everything > -1:
            pl.seed_everything(cfg.seed_everything)
        

    def configure_optimizers(self):
        
        
            
        if self.train_cfg['to_train'] == 'occ':
            params_to_optimize = []
            if self.train_cfg['forward_beta_mlp']:
                params_to_optimize.append({'params' : self.hit.fwd_beta.parameters(), 'weight_decay':self.cfg['weight_decay_fwd_beta']})
            if self.train_cfg['optimize_occ']:
                params_to_optimize.append({'params' : self.hit.tissues_occ.parameters()})
            if self.train_cfg['optimize_disp']:
                params_to_optimize.append({'params' : self.hit.deformer.disp_network.parameters(), 'weight_decay':self.cfg['weight_decay_disp']})
            if self.train_cfg['pose_bs']:
                params_to_optimize.append({'params' : self.hit.deformer.pose_bs.parameters()})
            if self.train_cfg['optimize_compressor'] and self.train_cfg['compressor']:
                params_to_optimize.append({'params' : self.hit.deformer.compressor.parameters(), 'weight_decay':self.cfg['weight_decay_compressor']})
            if self.train_cfg['optimize_generator']:
                params_to_optimize.append({'params' : self.hit.generator.parameters()})
            if self.train_cfg['optimize_lbs']:
                params_to_optimize.append({'params' : self.hit.deformer.lbs_network.parameters(), 'weight_decay':self.cfg['weight_decay_lbs']})
            if self.train_cfg['mri_values']:
                params_to_optimize.append({'params': self.hit.mri_val_net.parameters()})

        elif self.train_cfg['to_train'] == 'pretrain':
            params_to_optimize = []
            if self.train_cfg['forward_beta_mlp']:
                params_to_optimize.append({'params' : self.hit.fwd_beta.parameters()})
            params_to_optimize.append({'params' : self.hit.deformer.disp_network.parameters(), 'weight_decay':self.cfg['weight_decay_disp']})
            params_to_optimize.append({'params' : self.hit.deformer.lbs_network.parameters(), 'weight_decay':self.cfg['weight_decay_lbs']})
            if self.train_cfg['pose_bs']:
                params_to_optimize.append({'params' : self.hit.deformer.pose_bs.parameters()})
            
        elif self.train_cfg['to_train'] == 'compression':
            params_to_optimize = [{'params' : self.hit.deformer.compressor.parameters()}]
        else:
            raise NotImplementedError(f"Unknown training mode {self.train_cfg['to_train']}")

        optimizer = torch.optim.Adam(params_to_optimize, lr=self.cfg.lr, betas=(self.cfg.adam_beta1, self.cfg.adam_beta2))

        if self.train_cfg['to_train'] == 'compression':
            monitor_value = 'train/compression_loss'
            return optimizer
        else:
            monitor_value = 'train/loss_occ'
            
        if self.cfg['use_scheduler']:
            lr_scheduler = {
                    'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True),
                    'monitor': monitor_value
                }
            # 'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1, last_epoch=-1, verbose=True),
            return [optimizer], [lr_scheduler]
        else:
            return optimizer

    def state_dict(self, *args, **kwargs):
        return self.hit.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        self.hit.load_state_dict(*args, **kwargs)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(MRIDataset.from_config(self.smpl_cfg, self.data_cfg, self.train_cfg, split='train'), drop_last=True,
            num_workers=self.cfg.num_workers, batch_size=self.cfg['batch_size'], pin_memory=False, shuffle=True)

    def val_dataloader(self):
        mri_dataset = MRIDataset.from_config(self.smpl_cfg, self.data_cfg, self.train_cfg, split='val')
        print(f"The val dataset used {self.cfg.num_workers} workers")
        return torch.utils.data.DataLoader(mri_dataset, drop_last=True,
            num_workers=self.cfg.num_workers, batch_size=self.cfg['batch_size'], pin_memory=False, shuffle=False)
        
    def test_dataloader(self):
        return torch.utils.data.DataLoader(MRIDataset.from_config(self.smpl_cfg, self.data_cfg, self.train_cfg, split='test'), drop_last=True,
            num_workers=0, batch_size=1, pin_memory=False, shuffle=False)
        
    # def on_before_optimizer_step(self, optimizer, optimizer_idx):
    #     # Compute the 2-norm for each layer
    #     # If using mixed precision, the gradients are already unscaled here
    #     norms = pl.utilities.grad_norm(optimizer, norm_type=2)
    #     self.log_dict(norms)
    

    def forward(self, batch, points, eval_mode=False, cond=None, unposed=False, template=False, use_part_id=False):
            
        """ Forward pass of the HIT model
        
        Args:
            batch (dict): Dictionary containing the input smpl parameters
            points (torch.tensor): Query points to evaluate of shape [B,T,3]
            eval_mode (bool): If True, the model is in evaluation mode
            cond: Dictionary to condition the submodules MLPs
            unposed: If True, the model will infer the tissues for the body provided in the batch in the canonical X pose
            template: If True, the model will infer the tissues for the template body (average shape, canonical pose)
            use_part_id (bool): If True, the model will load the skinning weight of the query points from the batch. Set to False if they are not available.  
        """     
           
        smpl_output = self.smpl(**batch, return_verts=True, return_full_pose=True)
        
        if use_part_id:
            part_id = batch['part_id']
            skinning_weights = batch['skinning_weights']
        else:
            part_id=None
            skinning_weights=None
        
        # Print the current step
        # print(f"step={self.global_step:05d}")
        # if self.global_step == 10470:
        #     print("Last step before the bug")
        #     import ipdb; ipdb.set_trace()
        output = self.hit.query(points,
                                smpl_output, 
                                part_id=part_id, 
                                skinning_weights=skinning_weights, 
                                eval_mode=eval_mode, 
                                template=template,
                                unposed=unposed)
        return output
 
 
    def pretraining_step(self, batch, batch_idx):
        
        cond = cond_create(batch['betas'], batch['body_pose'], smpl=self.smpl)
        smpl_output = self.smpl(**batch, return_verts=True, return_full_pose=True) 
        
        loss = 0
        # Beta disp loss     
        if self.train_cfg.lambda_betas > 0:
            beta_loss = compute_beta_disp_loss(self, batch, batch_idx, cond, 'train', self.train_cfg)
            loss = loss + beta_loss
            
        # if self.train_cfg.lambda_beta0 > 0:
        #     beta0_loss = compute_beta0_loss(self, batch, batch_idx, cond)
        #     loss = loss + beta0_loss  
            
        if self.train_cfg['lambda_lbs'] > 0 and self.train_cfg['optimize_lbs']:
            lbs_loss = compute_lbs_regularization_loss(self, batch, batch_idx, cond)
            loss = loss + lbs_loss
            
        if self.train_cfg['lambda_pose_bs'] > 0 and self.train_cfg['pose_bs']: 
            loss_pose_bs = pose_bs_loss(self, batch, batch_idx, cond)
            loss = loss + loss_pose_bs    
         
        if self.train_cfg['forward_beta_mlp']:
            x_c = get_template_verts(batch, self.smpl)
            linear_beta_loss = compute_linear_beta_loss(self, batch, cond, smpl_output, x_c, on_surface=True)
            loss = loss + linear_beta_loss
        

        # Log gradient norms
        nan_grad = False
        for name, param in self.named_parameters():
            if param.grad is not None:
                self.log(f'gradient/grad_norm_{name}', param.grad.norm(), on_step=True, on_epoch=False)
                # check if isNan
                param_norm = param.grad.norm()
                if torch.isnan(param_norm):
                    import ipdb; ipdb.set_trace()
                    nan_grad = True
                    # import ipdb; ipdb.set_trace()
        
        if nan_grad:
            self.checkpoint_fct('grad_nan')
               
            
        return loss
           
    def training_step(self, batch, batch_idx): 

        # If the dataset visualization flag is enabled 
        if self.cfg['render_dataset']:
            self.export_visuals(batch, batch['seq_names'], split="train")
            return
        
        if self.train_cfg['to_train'] == 'pretrain':
            loss = self.pretraining_step(batch, batch_idx)
            return loss

        B = batch['betas'].shape[0]
        cond = cond_create(batch['betas'], batch['body_pose'])
        smpl_output = self.smpl(**batch, return_verts=True, return_full_pose=True)
                
        loss = 0
        
        template = False
        unposed = False
        if self.train_cfg['to_train'] == 'pretrain':
            unposed = True
            
        if self.data_cfg.subjects != 'all':
            if self.cfg.overfit_style == 'template':
                template=True
            elif self.cfg.overfit_style == 'unposed':
                unposed = True
            elif self.cfg.overfit_style == 'posed':
                pass
            else:
                raise NotImplementedError(f"Unknown overfit style {self.cfg.overfit_style}")

        

        # ------------------------------- SMPL surface losses
        
        if self.train_cfg['lambda_compression'] > 0 and self.train_cfg['compressor'] and self.train_cfg['optimize_compressor']:
            loss_compression = compute_compression_loss(self, batch, batch_idx, smpl_output, cond)
            loss = loss + loss_compression

        # Beta disp loss     
        if self.train_cfg.lambda_betas > 0:
            beta_loss = compute_beta_disp_loss(self, batch, batch_idx, cond, 'train', self.train_cfg)
            loss = loss + beta_loss
           
        if self.train_cfg.lambda_surf > 0:
            surf_loss = compute_surface_loss(self, batch, batch_idx, cond) 
            loss = loss + surf_loss    
            
        if self.train_cfg.lambda_canonicalization > 0:
            # compute_canonicalization_loss
            canonicalization_loss = compute_canonicalization_loss(self, batch, batch_idx, smpl_output, cond)
            loss = loss + canonicalization_loss
            
        # ------------------------------- Inside points losses
        
        x_f = batch['mri_points']  
        x_f.requires_grad_()
        n_points = batch['mri_occ'].shape[0] * batch['mri_occ'].shape[1] # = B*T
        
        # Pass inside points
        output = self.forward(batch, x_f, unposed=unposed, template=template, use_part_id=True)
        pred_occ = output['pred_occ']
        x_c = output['pts_c']
        # wandb.log({"train/can_inside_pts": wandb.Object3D(pts_c_inside[0].detach().cpu().numpy())})
        
        # Occupancy loss
        occ_loss = compute_occupancy_loss(self, batch, pred_occ)
        loss = loss + occ_loss 
        
        
        # force disp to be zero for beta=zero  
        if self.train_cfg.lambda_beta0 > 0:
            beta0_loss = compute_beta0_loss(self, batch, batch_idx, cond)
            loss = loss + beta0_loss       
         
        if self.train_cfg['forward_beta_mlp']:
            linear_beta_loss = compute_linear_beta_loss(self, batch, cond, smpl_output, x_c)
            loss = loss + linear_beta_loss
                 
        
        if self.train_cfg['lambda_pose_bs'] > 0 and self.train_cfg['pose_bs']: 
            loss_pose_bs = pose_bs_loss(self, smpl_output, batch, batch_idx, cond)
            loss = loss + loss_pose_bs           
            
        if self.train_cfg['lambda_outcan'] > 0 :          
            loss_outside_can = compute_outside_can_loss(self, smpl_output, batch, batch_idx)
            loss = loss + loss_outside_can           
                     
        if self.train_cfg['lambda_hands'] > 0 and 'hands_can_points' in batch:
            loss_hands =  compute_hands_loss(self, batch)
            loss = loss + loss_hands         
            
        if self.train_cfg['lambda_lbs'] > 0 and self.train_cfg['optimize_lbs']:
            lbs_loss = compute_lbs_regularization_loss(self, batch, batch_idx, cond)
            loss = loss + lbs_loss
            
        if self.train_cfg['lambda_comp0'] > 0 and self.train_cfg['compressor'] and self.train_cfg['optimize_compressor']:
            comp0_loss = compute_comp0_loss(self, batch, batch_idx, smpl_output, cond, x_c, pred_occ)
            loss = loss + comp0_loss

        if self.train_cfg['mri_values'] > 0:
            mri_loss = compute_mri_loss(self, batch, batch_idx, smpl_output, cond, x_c)
            loss = loss + mri_loss
                 
        # Eikonal loss - untested
        if self.train_cfg.lambda_eikonal > 0:
            smooth_loss = compute_smooth_loss(batch, x_c)
            loss = loss + smooth_loss
           
        # Accuracy 
        y = batch['mri_occ'].long()
        if len(self.train_cfg.mri_labels) > 1:
            pred = torch.argmax(pred_occ, -1)
            accuracy = torch.sum(pred == y) / n_points
        else:
            pred = pred_occ > 0.5
            accuracy = torch.sum(pred == 1) / n_points

        self.log("train/accuracy", accuracy)
        
                
        # if loss>1 and self.current_epoch > 0:
        #     print(f"WARNING: loss={loss:.4f} is too high. The subjects in the batch are {batch['seq_names']}")
            # Uncomment to print the sampled nb of pts
            # for bi in range(B):
            #     
            #     print(f"Points sampled for subject {bi}")
            #     for ci in range(len(self.train_cfg['mri_labels'])):
            #         ni = torch.sum(batch['mri_occ'][bi] == float(ci))
            #         print(f"\tClass {ci}: {ni} points")
            
        self.log("train/loss", loss)
        
        for name, param in self.named_parameters():
            if param.grad is not None:
                self.log(f'gradient/grad_norm_{name}', param.grad.norm(), on_step=True, on_epoch=False)
                
        if loss.isnan():
            print("NAN in loss")
            import ipdb; ipdb.set_trace()

        # if self.global_step == 10430:
        #     print("Last 2 step before the bug")
        #     import ipdb; ipdb.set_trace()
        return loss
    
    # def on_after_backward(self):
    #     print('After backward')
    #     max_grad = max([param.grad.abs().max() for  name, param in self.named_parameters() if param.grad is not None])
    #     print(f"Max grad: {max_grad}")

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        if self.cfg.print_grad_clip:
            # print('Before optimizer step')
            max_grad = max([param.grad.abs().max() for  name, param in self.named_parameters() if param.grad is not None])
            print(f"Step {self.global_step:05d} - Max grad: {max_grad}")
        
        # torch.nn.utils.clip_grad_norm_(self.hit.parameters(), 10000) #cfg.clip_grad_norm
        # max_grad = max([param.grad.abs().max() for  name, param in self.named_parameters() if param.grad is not None])
        # print(f'\tAfter norm clipping:  {max_grad}')
        if self.cfg.gradient_clip_val is not None:
            torch.nn.utils.clip_grad_value_(self.hit.parameters(), 100)
            if self.cfg.print_grad_clip:
                max_grad = max([param.grad.abs().max() for  name, param in self.named_parameters() if param.grad is not None])
                print(f'\tAfter value clipping:  {max_grad}')


    def validation_step(self, batch, batch_idx):

        if self.train_cfg['to_train'] == 'pretrain':
                    
            cond = cond_create(batch['betas'], batch['body_pose'])
            
            # Log slices
            slice_values = ["sw", "beta"]
            if self.cfg.train_cfg['forward_beta_mlp']:
                slice_values.append("fwd_beta")
            log_slices(self, batch, values=slice_values) #betas   

            loss = 0
            # Beta disp loss     
            if self.train_cfg.lambda_betas > 0:
                beta_loss = compute_beta_disp_loss(self, batch, batch_idx, cond, step='val', train_cfg=self.train_cfg)
                loss = loss + beta_loss
                
            # if self.train_cfg.lambda_beta0 > 0:
            #     beta0_loss = compute_beta0_loss(self, batch, batch_idx, cond)
            #     loss = loss + beta0_loss  
                
            if self.train_cfg['lambda_lbs'] > 0 and self.train_cfg['optimize_lbs']:
                lbs_loss = compute_lbs_regularization_loss(self, batch, batch_idx, cond, step='val')
                loss = loss + lbs_loss
                
            return loss
        
        if self.train_cfg['to_train'] == 'compression': 
            smpl_output = self.smpl(**batch, return_verts=True, return_full_pose=True)
            loss = compute_compression_loss(self, batch, batch_idx, smpl_output, cond, step='val')
            return loss
            
        elif self.train_cfg['to_train'] == 'occ':
            
            # Compute the occupancy loss for the validation set
            val_metric = self.compute_val_loss(batch)
            
            # Log the metrics
            for key, val in val_metric.items():
                if 'part' in key: # Log the per part dice score in another panel
                    self.log(f"val_part/{key}", val, sync_dist=True)
                else:
                    self.log(f"val/{key}", val, sync_dist=True)
        
            # Log evaluation on slices
            slice_values = ["occ", "sw", "beta", "comp"]
            if self.cfg.train_cfg['forward_beta_mlp']:
                slice_values.append("fwd_beta")
            log_slices(self, batch, values=slice_values)

            # Log the tissue meshes of the canonical body mesh
            if (self.current_epoch % self.cfg.export_canmeshes_every_n_epochs) == 0:
                log_canonical_meshes(self, batch)
                    
            return val_metric
        
        else:
            raise NotImplementedError(f"Unknown training mode {self.train_cfg['to_train']}")

    def test_step(self, batch, batch_idx):        
        test_metric = {}
        if not self.cfg.visuals_only:
            test_metric = self.compute_test_loss(batch)
            # test_metric.update({'subject_name': batch['seq_names'][0]})
            # print(f"Subject {batch['seq_names'][0]}")
            # for key, val in test_metric.items():
            #     self.log(f"test/{key}", val, sync_dist=True)

        if self.cfg.eval_export_visuals or self.cfg['render_dataset']:
            print('Exporting visuals')
            self.export_visuals(batch, batch['seq_names'])
            
        return test_metric

    def compute_val_loss(self, batch):
        points = batch['mri_points']
        gt_occ = batch['mri_occ']
        body_mask = batch['body_mask']
        batch_size, n_pts, _ = points.shape
        
        val_loss_dict = {}

        # prevent OOM
        pred_occ = []
        part_id = []
        # for pts in torch.split(points, self.max_queries//batch_size, dim=1):
        #
        output = self.forward(batch, points, eval_mode=True, use_part_id=True)
        occ = output['pred_occ']
        pred_occ.append(occ)      
        
        pred_occ = torch.cat(pred_occ, dim=1)
        
        if len(self.train_cfg.mri_labels) > 1:
            pred = torch.argmax(pred_occ, -1)
        else:
            pred = pred_occ > 0.5
        part_id = batch['part_id']

        val_loss_dict = metrics.validation_eval(self.cfg, pred, gt_occ, part_id, body_mask)

        return val_loss_dict
        
    def compute_test_loss(self, batch):
        
        points = batch['mri_points']
        gt_label = batch['mri_occ']
        part_id = batch['part_id']
        skinning_weights = batch['skinning_weights']
        body_mask = batch['body_mask']
        batch_size, n_pts, _ = points.shape
        
        smpl_output = self.smpl(**batch, return_verts=True, return_full_pose=True)

        # Predict the points
        pred_occ = []
        print('Evaluating all the points in the MRI...')
        indices = torch.arange(0, n_pts)
        for idx in tqdm.tqdm(torch.split(indices, self.max_queries, dim=0)):
            output = self.hit.query(points[:,idx], 
                                     smpl_output, 
                                     eval_mode=True, 
                                     part_id=part_id[:,idx].to(points.device), 
                                     skinning_weights=skinning_weights[:,idx].to(points.device))
                # output = self.forward(batch, pts, eval_mode=True, use_part_id=True)
            pred_occ.append(output['pred_occ'])
        pred_occ = torch.cat(pred_occ, dim=1)
        # pred_occ = output['pred_occ']
        if len(self.train_cfg.mri_labels) > 1:
            pred_label = torch.argmax(pred_occ, -1)
        else:
            pred_label = pred_occ > 0.5
        print("Eval done.")
        
        # compute the metrics
        test_loss_dict = {}
        
        accuracy = metrics.compute_accuracy(pred_label, gt_label)
        test_loss_dict.update({'accuracy': accuracy})
        
        # To fix
        # part_dice_dict = metrics.compute_part_dice(self.data_cfg,pred_label, gt_label, part_id, body_mask)
        # test_loss_dict.update(part_dice_dict)
        
        mri_shape = [max(batch['mri_size'][:,i]).item() for i in range(3)]
        
        images_dict = {}
        for li, mri_label in enumerate(['NO', 'LT', 'AT', 'BONE']):
            pred_occ = (pred_label == li)
            # 
            gt_occ = (gt_label == li)
            
            if mri_label == 'NO':
                pred_occ = torch.logical_and(pred_occ, body_mask)
                gt_occ = torch.logical_and(gt_occ, body_mask)      
                  
            test_loss_dict[f"dice_{mri_label}"] = metrics.compute_dice(pred_occ, gt_occ)
            test_loss_dict[f"comp_{mri_label}"] = metrics.compute_composition(pred_occ, gt_occ, batch['body_mask'], mri_shape=mri_shape)
            # test_loss_dict[f"hd_{label}"] = metrics.compute_hd(pred_occ, gt_occ, mri_shape=MRI_SHAPE, mri_coords=batch['points_uv'] ,spacing=batch['mri_resolution'])

            if not self.cfg.slices:
               continue
            
            dst_slices_dir = os.path.join(self.logger.save_dir, 'test', 'slices', f'epoch={self.current_epoch:05d}_step={self.global_step:05d}', mri_label)  
            
            mri_uvs = batch['mri_coords']
            batch_images = metrics.per_slice_prediction(pred_occ, gt_occ, mri_uvs, mri_shape, dst_slices_dir, batch['seq_names'], level=0.5, return_image=True)     
            images_dict[li] = batch_images
            
        if not self.cfg.slices:
           return dict(test_loss_dict)

        # Generate images for comparison
        palette = {
            0: (0.0, 0.0, 0.0),
            1: (0.803921568627451, 0.3607843137254902, 0.3607843137254902),
            2: (0.8549019607843137, 0.6470588235294118, 0.12549019607843137),
            3: (0.8784313725490196, 0.8901960784313725, 0.8431372549019608)
        }
        idx = 2
        # TODO: 
        h,w,d = MRI_SHAPE
    
        for b in range(batch_size):
            for idx in range(d):
                composite_gt = np.zeros((h,w,3))
                composite_pred = np.zeros((h,w,3))
                for li, mri_label in enumerate(['NO', 'LT', 'AT', 'BONE']):
                    if mri_label=='NO':
                        continue
                    # import ipdb; ipdb.set_trace()
                    composite_gt += np.repeat(images_dict[li][b][idx]['gt'].astype(int).reshape(h,w,1),3, axis=-1) * palette[li]
                    composite_pred += np.repeat(images_dict[li][b][idx]['pred'].astype(int).reshape(h,w,1),3, axis=-1) * palette[li]
                    # Compute differences
                    dst_slices_dir = os.path.join(self.logger.save_dir, 'test', 'slices', f'epoch={self.current_epoch:05d}_step={self.global_step:05d}', mri_label)      
                    out_folder_ = os.path.join(dst_slices_dir, batch['seq_names'][b], 'new_images')
                    os.makedirs(out_folder_, exist_ok=True)

                    gt = images_dict[li][b][idx]['gt'].astype(int)
                    pred = images_dict[li][b][idx]['pred'].astype(int)
                
                    x = np.where(gt==False, None, gt==pred)
                    x[x==None] = 2
                    x[x==True] = 3
                    x[x==False] = 4
                    x = x-2
                    colors = np.array([np.array([0.,0.,0.]), np.array([0.,1.,0.]), np.array([1.,0.,0.])])
                    difference = colors[x.astype(int)]
                    colors = np.array([np.array([0.,0.,0.]), np.array(palette[1])])
                    gt = colors[gt.astype(int)]
                    colors = np.array([np.array([0.,0.,0.]), np.array(palette[2])])
                    pred = colors[pred.astype(int)]
                    difference = np.concatenate([gt, pred, difference], axis=1)
                    # Save the images
                    imsave(f'{out_folder_}/difference_{idx:05d}.png', (difference*255).astype(np.uint8))
                
                dst_slices_dir = os.path.join(self.logger.save_dir, 'test', 'slices', f'epoch={self.current_epoch:05d}_step={self.global_step:05d}', 'composites')      
                out_folder = os.path.join(dst_slices_dir, batch['seq_names'][b], 'comparison')
                os.makedirs(out_folder, exist_ok=True)
                im = np.concatenate([composite_gt, composite_pred], axis=1)
                imsave(f'{out_folder}/comparison_{idx:05d}.png', (im*255).astype(np.uint8))  
                
                # Good final comaprison 
                bg_mask = (np.where(composite_gt== np.array([0.,0.,0.]), 0., 1.).sum(-1)/3.).astype(np.uint8)  
                img = (np.where(composite_gt== np.array([0.,0.,0.]), 2, composite_gt==composite_pred).all(-1)).astype(np.uint8) 
                colors = np.array([np.array([1.,0.,0.]), np.array([0.,1.,0.]), np.array([0.,0.,0.])])
                img = colors[img] * np.repeat(bg_mask.reshape(h,w,1), 3, axis=-1)
                dst_slices_dir = os.path.join(self.logger.save_dir, 'test', 'slices', f'epoch={self.current_epoch:05d}_step={self.global_step:05d}', 'all_comparison')      
                out_folder = os.path.join(dst_slices_dir, batch['seq_names'][b], 'comparison')
                os.makedirs(out_folder, exist_ok=True)
                im = np.concatenate([composite_gt, composite_pred, img], axis=1)
                imsave(f'{out_folder}/comparison_{idx:05d}.png', (im*255).astype(np.uint8))  

        return dict(test_loss_dict)
   

    def test_epoch_end(self, outputs):
        """ Aggregate test predictions.

        Args:
            outputs (list): list of dictionaries containing scores
        """
        
        # import ipdb; ipdb.set_trace()
        test_metric_keys = outputs[0].keys()
        test_metric_stacked = {key: torch.stack([x[key] for x in outputs]) for key in test_metric_keys}
        val_subjects = [data['seq_names'][0] for data in self.test_dataloader()]
        
        agg_metric = {}
        for key in test_metric_keys:
            agg_metric[key] = torch.mean(torch.abs(self.all_gather(test_metric_stacked[key])))
            
        sorted_keys = ['accuracy', 
                           'dice_LT', 
                           'dice_AT', 
                           'dice_BONE',
                           'dice_NO',
                           'comp_LT',
                           'comp_AT', 
                           'comp_BONE',
                           'comp_NO', 
                           ]

        part_metric = {}
        key_list = list(agg_metric.keys())
        for key in key_list:
            if key not in sorted_keys:
                part_metric[key] = agg_metric[key]
                del agg_metric[key]

        with open(self.logger.save_dir +'/part_metric.pkl', 'wb') as fp:
            pickle.dump(part_metric, fp)
        # for key in key_list:
        #     if key not in ['accuracy','dice_NO','comp_NO', 'dice_LT', 'comp_LT', 'dice_AT', 'comp_AT' ]:
        #         del agg_metric[key]
                

        # 
        # Print results as a table
        print(f"Test results:") 
        from prettytable import PrettyTable
        x = PrettyTable()
        val_metric_print = {key: [x[key].item() for x in outputs] for key in agg_metric}
        x.add_column("Subject", val_subjects)
        for key in sorted_keys:
            if 'iou' in key:
                continue
            x.add_column(key, val_metric_print[key])
        x.float_format = '.3'
        print(x.get_string(sortby="accuracy", reversesort=True))

        # Log the pretty table to file
        test_res_table_path = os.path.join(self.logger.save_dir, f'test_persubject_table.txt')
        with open(test_res_table_path, 'w') as w:
            w.write(x.get_string(sortby="accuracy", reversesort=True))
        print(f"Per subject results table saved to {test_res_table_path}")
        
        
        # log table to wandb
        wandb.log({"test/per_subject_metrics": wandb.Html(x.get_html_string())})
        wandb.log({"test/per_subject_metrics_table": wandb.Table(data=x.rows, columns=x.field_names)})
        
        # Print mean results as a table
        val_metric_print = {key: [float(val.item() if torch.is_tensor(val) else val)] for key, val in agg_metric.items()}
        x = PrettyTable()
        x.add_column(" ", ['Average'])
        for key in sorted_keys:
            if 'iou' in key:
                continue
            x.add_column(key, val_metric_print[key])    
        x.float_format = '.3'
        print(x)
        
        # Log to a file
        test_res_table_path = os.path.join(self.logger.save_dir, f'test_mean_table.txt')
        with open(test_res_table_path, 'w') as w:
            w.write(x.get_string())
        
        wandb.log({"test/mean_metric": wandb.Html(x.get_html_string())})
        wandb.log({"test/mean_metric_table": wandb.Table(data=x.rows, columns=x.field_names)})        
            
        # Save result
        if self.trainer.is_global_zero:  # to avoid deadlock
            for key, val in agg_metric.items():
                if 'part' in key: # Log the per part dice score in another panel
                    self.log(f"test_part/{key}", val, sync_dist=True)
                else:
                    self.log(f"test/{key}", val, sync_dist=True)
            test_metric = {key: float(val.item() if torch.is_tensor(val) else val) for key, val in agg_metric.items()}

            dst_path = os.path.join(self.logger.save_dir, f'test_epoch={self.current_epoch:05d}_step={self.global_step:05d}.yml')
            with open(dst_path, 'w') as f:
                yaml.dump(test_metric, f) 

            print('\n\nTest results are saved in:', dst_path, end='\n\n')

        #   
        
        
 



@hydra.main(config_path="configs", config_name="config") 
def main(cfg):
    
    torch.set_float32_matmul_precision('high')
   
    # Create output directory
    out_dir = os.path.join(cg.trained_models_folder, cfg.exp_name )
    os.makedirs(out_dir, exist_ok=True)
    print(f"Logs and checkpoints will be saved to {out_dir}")

    # Load config
    exp_cfg_file = os.path.join(cg.trained_models_folder, cfg.exp_name, 'config.yaml')
    OmegaConf.save(cfg, exp_cfg_file)
    print(f'Created config file {exp_cfg_file}')
    
    if cfg.run_eval:
        print('WARNING: Batch size set to 1 for evaluation.')
        cfg['batch_size'] = 1
        
    # In rendering dataset mode
    if cfg.render_dataset:
        cfg.trainer['max_epochs'] = 1

    # create trainer
    if cfg.train_cfg['to_train'] == 'compression':
        ckp_monitor = 'compression_val/loss'  
        filename='model-epoch={epoch:04d}-val_accuracy={compression_val/loss:.6f}'
        mode='min'
    elif cfg.train_cfg['to_train'] == 'pretrain':
        ckp_monitor = 'val/loss_lbs'
        filename='model-epoch={epoch:04d}-loss_lbs={val/loss_lbs:.6f}'
        mode='min'
    else:
        ckp_monitor = 'val/accuracy'
        filename='model-epoch={epoch:04d}-val_accuracy={val/accuracy:.6f}'
        mode='max'
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=f'{out_dir}/ckpts/',
        filename=filename, 
        every_n_epochs=cfg.checkpoint_every_n_epochs,
        save_top_k=1,  # Save the best model
        monitor = ckp_monitor,
        mode=mode, # The best model is the one with the highest val/accuracy 
        save_last=True, verbose=True, auto_insert_metric_name=False,
    )
    
    def checkpoint_fct(ckp_name):
        trainer.save_checkpoint("f'{out_dir}/ckpts/{ckp_name}.ckpt")

    
    if cg.logger == 'wandb':
        resuming_run = False
        if cfg.wdboff:
            wandbmode = 'disabled'
        else:
            wandbmode = 'online'
        # if previous run exists, resume it
        if cfg.resume and len(glob.glob(os.path.join(out_dir, 'wandb', 'run*')))>0:
            dirpath = os.path.join(out_dir, 'wandb', 'run*')
            run_folders = glob.glob(dirpath)
            run_folders_sorted = sorted(run_folders, key=os.path.getmtime, reverse=True)
            id = run_folders_sorted[0].split('/')[-1].split('-')[-1]
            print(f"Resuming run {id}")
            resuming_run = True
            logger = pl.loggers.WandbLogger(name=cfg.exp_name, entity=cg.wandb_entity, project=cg.wandb_project_name, id=id, save_dir=out_dir, log_model=False, resume='must', mode=wandbmode)
        else:
            logger = pl.loggers.WandbLogger(name=cfg.exp_name, entity=cg.wandb_entity, project=cg.wandb_project_name, save_dir=out_dir, log_model=False, mode=wandbmode)
    else:
        logger = pl.loggers.TensorBoardLogger(out_dir, os.path.basename(out_dir))
    
    # log all the config parameters to wandb
    all_cfg = {**cfg.train_cfg, **cfg.data_cfg, **cfg.smpl_cfg}
    if not resuming_run:
        logger.experiment.config.update(all_cfg)

    trainer = pl.Trainer(**cfg.trainer,
                         callbacks=[checkpoint_callback, pl.callbacks.TQDMProgressBar(refresh_rate=10)],
                         logger=logger
                        )

    # path to resume training
    if cfg.ckpt_path is not None and os.path.exists(cfg.ckpt_path):
        ckpt_path = cfg.ckpt_path
    elif cfg.run_eval:
        # ckpt_path = os.path.join(checkpoint_callback.dirpath, f"{checkpoint_callback.CHECKPOINT_NAME_LAST}.ckpt")
        from utils.exppath import Exppath
        exppath = Exppath(cfg.exp_name)
        ckpt_path = exppath.get_best_checkpoint()
        # ckpt_path = checkpoint_callback.best_model_path # this is not working for some reason
    elif os.path.exists(os.path.join(checkpoint_callback.dirpath, f"{checkpoint_callback.CHECKPOINT_NAME_LAST}.ckpt")):
        ckpt_path = os.path.join(checkpoint_callback.dirpath, f"{checkpoint_callback.CHECKPOINT_NAME_LAST}.ckpt")
    else:
        ckpt_path = None
    # create model
    model = HITTraining(cfg, checkpoint_fct=checkpoint_fct)
    if cfg.run_eval:
        trainer.test(model, ckpt_path=ckpt_path, verbose=True)
    else:
        trainer.fit(model, ckpt_path=ckpt_path)
    

if __name__ == "__main__":
    main()