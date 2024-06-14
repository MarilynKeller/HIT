
import glob
import os
import pathlib
import pickle
import hydra
import numpy as np
import pytorch_lightning as pl
import smplx
import torch
import tqdm
import trimesh
import yaml
from training.losses import *
import wandb

from matplotlib import pyplot as plt
from model.mysmpl import MySmpl
from omegaconf import OmegaConf
from skimage.io import imsave

from hit.model.hit_model import HITModel
import hit.hit_config as cg
import hit.training.metrics as metrics
from hit.training.losses import *
from hit.utils.experiments import col_at, col_bn, col_lt, col_no
from hit.utils.smpl_utils import get_template_verts, x_pose_like
from hit.utils.tensors import cond_create
from hit.utils.slice_extractor import SliceLevelSet
from hit.training.dataloader_mri import MRI_SHAPE, MRIDataset
from hit.utils.renderer import Renderer


class HITTraining(pl.LightningModule):
    def __init__(self, cfg):
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
     
        self.renderer = Renderer()
        
        self.max_queries = self.train_cfg['max_queries'] # to prevent OOM at inference time
        
        print(self.hit.tissues_occ) 
        

    def configure_optimizers(self):
        
        params_to_optimize = []
        
        if self.train_cfg['forward_beta_mlp']:
            params_to_optimize.append({'params' : self.hit.fwd_beta.parameters()})
            
        if self.train_cfg['to_train'] == 'occ':
            if self.train_cfg['optimize_occ']:
                params_to_optimize.append({'params' : self.hit.tissues_occ.parameters()})
            if self.train_cfg['optimize_disp']:
                params_to_optimize.append({'params' : self.hit.deformer.disp_network.parameters()})
            if self.train_cfg['pose_bs']:
                params_to_optimize.append({'params' : self.hit.deformer.pose_bs.parameters()})
            if self.train_cfg['optimize_compressor'] and self.train_cfg['compressor']:
                params_to_optimize.append({'params' : self.hit.deformer.compressor.parameters(), 'weight_decay':self.cfg['weight_decay_compressor']})
            if self.train_cfg['optimize_generator']:
                params_to_optimize.append({'params' : self.hit.generator.parameters()})
            if self.train_cfg['optimize_lbs']:
                params_to_optimize.append({'params' : self.hit.deformer.lbs_network.parameters()})
            # @varora
            # add mri network parameters for optimizer
            if self.train_cfg['mri_values']:
                params_to_optimize.append({'params': self.hit.mri_val_net.parameters()})

        elif self.train_cfg['to_train'] == 'pretrain':
            params_to_optimize = []
            params_to_optimize.append({'params' : self.hit.deformer.disp_network.parameters()})
            params_to_optimize.append({'params' : self.hit.deformer.lbs_network.parameters()})
            if self.train_cfg['pose_bs']:
                params_to_optimize.append({'params' : self.hit.deformer.pose_bs.parameters()})
            
        elif self.train_cfg['to_train'] == 'compression':
            params_to_optimize = [{'params' : self.hit.deformer.compressor.parameters()}]
        else:
            raise NotImplementedError(f"Unknown training mode {self.train_cfg['to_train']}")

        optimizer = torch.optim.Adam(params_to_optimize, lr=self.cfg.lr)
        # return optimizer
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
        # return self.val_dataloader()
        # print(f"*********WARNING: using val dataset for test")
        return torch.utils.data.DataLoader(MRIDataset.from_config(self.smpl_cfg, self.data_cfg, self.train_cfg, split='test'), drop_last=True,
            num_workers=0, batch_size=1, pin_memory=False, shuffle=False)
        
    # def on_before_optimizer_step(self, optimizer, optimizer_idx):
    #     # Compute the 2-norm for each layer
    #     # If using mixed precision, the gradients are already unscaled here
    #     norms = pl.utilities.grad_norm(optimizer.
    #     self.log_dict(norms)


    def forward(self, batch, points, eval_mode=False, cond=None, unposed=False, template = False, 
                only_near_smpl=False, use_part_id=False):
            
            smpl_output = self.smpl(**batch, return_verts=True, return_full_pose=True)
            # smpl_output.v_shaped = batch['v_shaped']
            # smpl_output.v_free = batch['body_verts']
            # smpl_output.n_free = batch['body_normals']
            
            if use_part_id:
                part_id = batch['part_id']
                skinning_weights = batch['skinning_weights']
            else:
                part_id=None
                skinning_weights=None
            #
        
                
            # if self.train_cfg.get('use_precomputed_dist', 'False') is True:
            #     return self.hit.query(points, smpl_output, 
            #                                     is_inside=batch['is_inside'], 
            #                                     signed_dist=batch['signed_dist'],canonical=True)
            # else:
            return self.hit.query(points,
                                   smpl_output, 
                                   part_id=part_id, 
                                   skinning_weights=skinning_weights, 
                                   eval_mode=eval_mode, 
                                   template=template,
                                   unposed=unposed)
 
 
    def pretraining_step(self, batch, batch_idx):
        
        cond = cond_create(batch['betas'], batch['body_pose'], smpl=self.smpl)
        # smpl_output = self.smpl(**batch, return_verts=True, return_full_pose=True) 
        
        loss = 0
        # Beta disp loss     
        if self.train_cfg.lambda_betas > 0:
            beta_loss = compute_beta_disp_loss(self, batch, batch_idx, cond)
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
            loss_compression = compute_compression_loss(self, batch, batch_idx, smpl_output)
            loss = loss + loss_compression
            
        if self.train_cfg['to_train'] == 'compression':
            return loss_compression
            
        # Beta disp loss     
        if self.train_cfg.lambda_betas > 0:
            beta_loss = compute_beta_disp_loss(self, batch, batch_idx, cond)
            loss = loss + beta_loss
           
        if self.train_cfg.lambda_surf > 0:
            surf_loss = compute_surface_loss(self, batch, batch_idx, cond) 
            loss = loss + surf_loss    
            


        # ------------------------------- Inside points losses
        
        pts = batch['mri_points']  
        pts.requires_grad_()
        n_points = batch['mri_occ'].shape[0] * batch['mri_occ'].shape[1] # = B*T
        
        # Pass inside points
        output = self.forward(batch, pts, unposed=unposed, template=template, use_part_id=True)
        pred_occ = output['pred_occ']
        pts_c_inside = output['pts_c']
        # wandb.log({"train/can_inside_pts": wandb.Object3D(pts_c_inside[0].detach().cpu().numpy())})
        
        # Occupancy loss
        occ_loss = compute_occupancy_loss(self, batch, pred_occ)
        loss = loss + occ_loss 
        
        
        # force disp to be zero for beta=zero  
        if self.train_cfg.lambda_beta0 > 0:
            beta0_loss = compute_beta0_loss(self, batch, batch_idx, cond)
            loss = loss + beta0_loss       
         
        if self.train_cfg['forward_beta_mlp'] and self.train_cfg['forward_beta_mlp']:
            linear_beta_loss = compute_linear_beta_loss(self, batch, cond, pts_c_inside)
            loss = loss + self.train_cfg['lambda_linear_beta'] * linear_beta_loss
                 
        
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
            comp0_loss = compute_comp0_loss(self, batch, batch_idx, smpl_output, cond, pts_c_inside, pred_occ)
            loss = loss + comp0_loss

        # @varora
        if self.train_cfg['mri_values'] > 0:
            mri_loss = compute_mri_loss(self, batch, batch_idx, smpl_output, cond, pts_c_inside)
            loss = loss + mri_loss
         
            
        # Eikonal loss - untested
        if self.train_cfg.lambda_eikonal > 0:
            smooth_loss = compute_smooth_loss(batch, pts_c_inside)
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
        return loss


    def pretraining_validation_step(self, batch, batch_idx):
        
        cond = cond_create(batch['betas'], batch['body_pose'])
        # smpl_output = self.smpl(**batch, return_verts=True, return_full_pose=True) 
        
        loss = 0
        # Beta disp loss     
        if self.train_cfg.lambda_betas > 0:
            beta_loss = compute_beta_disp_loss(self, batch, batch_idx, cond, step='val')
            loss = loss + beta_loss
            
        # if self.train_cfg.lambda_beta0 > 0:
        #     beta0_loss = compute_beta0_loss(self, batch, batch_idx, cond)
        #     loss = loss + beta0_loss  
            
        if self.train_cfg['lambda_lbs'] > 0 and self.train_cfg['optimize_lbs']:
            lbs_loss = compute_lbs_regularization_loss(self, batch, batch_idx, cond, step='val')
            loss = loss + lbs_loss
            
        return loss

    def validation_step(self, batch, batch_idx):
        # If the dataset visualization flag is enabled 
        if self.cfg['render_dataset']:
            self.export_visuals(batch, batch['seq_names'], split="val")
            return
        
        if self.train_cfg['to_train'] == 'pretrain':
            return self.pretraining_validation_step(batch, batch_idx)
        
 
        if self.train_cfg['to_train'] == 'compression': 
            smpl_output = self.smpl(**batch, return_verts=True, return_full_pose=True)
            # set to zero for testing
            cond = cond_create(batch['betas']) 
  
            smpl_template_verts = get_template_verts(batch, self.smpl)
            
            xv_p = batch['body_verts']
            xf_p = batch['body_verts_free']
            
            # Global learned in xf_c
            # xv_c = smpl_template_verts
            # smpl_tfs = get_gdna_bone_transfo(smpl, smpl_output)
            # xf_c = self.hit.canonicalize_from_similar(xf_p, xv_c, smpl_tfs, cond)
            
            # d = xv_p - xf_p
            # d_pred = self.hit.deformer.compressor(xf_c, cond) 
            
            # Global learned in xf_s
            xv_c = smpl_template_verts
            smpl_tfs = smpl_output.tfs #get_gdna_bone_transfo(smpl, smpl_output)
            xf_s = self.hit.canonicalize_from_similar(xf_p, xv_c, smpl_tfs, cond, undo_shape=False)
            
            d = xv_p - xf_p
            d_pred = self.hit.deformer.compressor(xf_s, cond)   

            # Convert to mm
            # TODO make a nice weighted loss using clamp
            d = d * 1000
            d_pred = d_pred * 1000
            vi = 3504 #vertex index on the belly
            loss_fct_comp = torch.nn.MSELoss()
            loss_compression = loss_fct_comp(d_pred, d)
            self.log("compression_val/loss", loss_compression)
            self.log("compression_val/d_err_belly", ((torch.abs(d_pred-d)[:,vi])).mean())
            self.log("compression_val/d_err_max", torch.linalg.norm(d_pred-d, axis=-1).max())
            
        else:    
            val_metric = self.compute_val_loss(batch)
            for key, val in val_metric.items():
                if 'part' in key: # Log the per part dice score in another panel
                    self.log(f"val_part/{key}", val, sync_dist=True)
                else:
                    self.log(f"val/{key}", val, sync_dist=True)
                    
            if batch_idx > 0:# extract and render meshes for the first batch only
                return 
        
        
            smpl_output = self.smpl(**batch, return_verts=True, return_full_pose=True)
        
            #self.conf.train_cfg['build_val_meshes_every_n_epochs']-1:
            # Evaluate disp fielf
            sl = SliceLevelSet()
            # sl_comp = SliceLevelSet(res=0.01) # Use a lower resolution for 
            for z0 in [-0.2, -0.4, 0, 0.3] :#np.linspace(-0.4,0.3,3):
      
                xc = sl.gen_slice_points(z0=z0)
                xc_batch = torch.FloatTensor(xc).to(self.device).unsqueeze(0).expand(batch['betas'].shape[0], -1, -1)
                
                output = self.hit.query(xc_batch, smpl_output, unposed=True, eval_mode=True)
                cond = cond_create(batch['betas'], batch['body_pose'], self.hit.generator)
                disp = self.hit.deformer.disp_network(xc_batch, cond)
                
                pts_c = output['pts_c']
                weights = output['weights']
                occ = output['pred_occ']
                
                
                if len(self.train_cfg.mri_labels) > 1 and not self.train_cfg['to_train'] in ['pretrain', 'compression']:
                    occ = torch.softmax(occ, -1)
                    occ = torch.argmax(occ, -1)
                else:
                    occ = torch.sigmoid(occ)
                    
                values_np = occ.detach().cpu().numpy()[0]
                disp_np = disp.detach().cpu().numpy()[0]
                
                # value = np.linalg.norm(value, axis=-1) 
 
                if not self.train_cfg['to_train'] == 'compression':      
                    sl.plot_slice_levelset(disp_np, values=values_np, to_plot=False)
                    im_path = os.path.join('/tmp', f'disp_{z0:.2f}.png')
                    plt.savefig(im_path)
                    plt.close()
                    print(f'Saved disp image to {im_path}')  
                    wandb.log({f"slices_disp/disp_{z0}": wandb.Image(im_path)})
                
                if self.train_cfg['compressor']:
                    d = self.hit.deformer.compressor(xc_batch, cond_create(torch.zeros_like(batch['betas'])))
                    d_np = d.detach().cpu().numpy()[0]
                    sl.plot_slice_levelset(d_np, values=d_np, to_plot=False, iscompression=True)
                    im_path = os.path.join('/tmp', f'compression_{z0:.2f}.png')
                    plt.savefig(im_path)
                    plt.close()
                    print(f'Saved disp image to {im_path}')  
                    wandb.log({f"slices_comp/compression_{z0}": wandb.Image(im_path)})
                
                # log image in wandb
                
            # Reconstruct occupancy of the canonical space
            try:
            # if True:
                for li in self.train_cfg.mri_labels:
                    if li == 'NO':
                        continue
                    else:
                        channel_index = self.train_cfg['mri_labels'].index(li)
                    can_mesh = self.generate_canonical_mesh(batch, channel_index)
                    can_mesh_path = os.path.join(self.logger.save_dir, 'val', 'can_mesh.glb')
                    os.makedirs(os.path.dirname(can_mesh_path), exist_ok=True)
                    can_mesh.export(can_mesh_path)
                    wandb.log({f"val_image/can_mesh_{li}": wandb.Object3D(can_mesh_path)})
            except Exception as e:
                print(f"Could not generate canonical mesh: {e}")

            # Infer test subjects
            # print(self.current_epoch)
            if (self.current_epoch+1) % self.cfg.build_val_meshes_every_n_epochs == 0:
                if self.train_cfg['to_train'] == 'compression':
                    return
                try:
                    image = self.generate_renders(batch, smpl_output)
                except Exception as e:
                    print(f"Could not generate renderings: {e}")
                    image = None
                    
                if image is not None:
                    self.logger.log_image(key="val_image/renderings", images=[image.float()])
                else:
                    print("Could not generate renderings")
                    
            return val_metric

    def test_step(self, batch, batch_idx):        
        test_metric = {}
        if not self.cfg.visuals_only:
            test_metric = self.compute_test_loss(batch)
            # val_metric.update({'subject_name': batch['seq_names'][0]})
            # print(f"Subject {batch['seq_names'][0]}")
            # for key, val in val_metric.items():
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
        #
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
        
        
    def generate_renders(self, batch, smpl_output):
        
        im_list = [] # each image will be a line
        smpl_free_image_tensor = None
        
        im_list = []
                   
        for li in  self.train_cfg.mri_labels:
            
            if li == 'NO':
                continue
            
            channel_label = li
            channel_index = self.train_cfg['mri_labels'].index(li)
            
            meshes, smpl_meshes, smpl_free_meshes, images, smpl_images, smpl_free_images = self.visualize(batch, channel=channel_index, smpl_output=smpl_output)
            if len(meshes) == 0:
                print(f"The output meshes for channel {channel_label} could not be reconstructed (this is normal during the first iterations)")
                continue
            
            # Rendering of the predicted tissue
            images_tensor = torch.from_numpy(np.concatenate(images[:6], axis=1)).permute(2, 0, 1) 
            im_list.append(images_tensor)
            
            # Rendering of the SMPL body
            smpl_image_tensor = torch.from_numpy(np.concatenate(smpl_images[:6], axis=1)).permute(2, 0, 1)
            smpl_free_image_tensor = torch.from_numpy(np.concatenate(smpl_free_images[:6], axis=1)).permute(2, 0, 1)

        if smpl_free_image_tensor is not None:
            im_list.append(smpl_free_image_tensor)
            if self.train_cfg['free_verts'] :
                # Print smpl image in this case because it differs from the free verts one
                im_list.append(smpl_image_tensor)

        if im_list:
            image = torch.cat(im_list, dim=1) 
        else:
            image=None
            
        return image   
    
    def generate_canonical_mesh(self, batch, channel_index):
        
            smpl_data  = {}
            smpl_data['betas'] = torch.zeros_like(batch['betas'])
            smpl_data['body_pose'] = x_pose_like(batch['body_pose'])
            smpl_data['global_orient'] = torch.zeros_like(batch['global_orient'])   
            smpl_data['transl'] = torch.zeros_like(batch['transl'])
            smpl_data['global_orient_init'] = torch.zeros_like(batch['global_orient_init'])
            smpl_output = self.smpl(**smpl_data, return_verts=True, return_full_pose=True)
            # 
            
            meshes = self.hit.extract_mesh(smpl_output, channel=channel_index, use_mise=True, template=True)
        
            return meshes[0]   

    def export_visuals(self, batch, seq_names:list, split: str='test'):
        # run qualifies if the export is done in the train or test data
        # This enables visualization
        
        smpl_output = self.smpl(**batch, return_verts=True, return_full_pose=True)
        # smpl_output.v_shaped = batch['v_shaped']
        # smpl_output.v_free = batch['body_verts']
        # smpl_output.n_free = batch['body_normals']
        # batch_size = smpl_output.vertices.shape[0]
        
        # smpl_output.betas[0] = torch.zeros_like(smpl_output.betas[0]) # Set to zero pose for visualization of the can space
        # smpl_output.body_pose[0] = torch.zeros_like(smpl_output.body_pose[0])

        # if len(self.train_cfg.mri_labels) == 1:
        #     raise NotImplementedError('Only one tissue is not supported yet')
        
        if self.data_cfg.synthetic:
           channel_list = [channel_list[1]]
           
        for li in  self.train_cfg.mri_labels:
            
            if li == 'NO':
                continue

            channel_label = li
            channel_index = self.train_cfg['mri_labels'].index(li)
            
            # If the dataset visualization selected only run marching cubes through the gt
            if not self.cfg['render_dataset']:
                meshes, smpl_meshes, smpl_free_meshes, images, smpl_images, smpl_free_images = self.visualize(batch, channel=channel_index, smpl_output=smpl_output)
                
            meshes_gt, smpl_meshes, smpl_free_meshes, images_gt, smpl_images, smpl_free_images = self.visualize(batch, channel=channel_index, smpl_output=smpl_output, gt=True)
            to_log = []
            if self.cfg['render_dataset']:
                for b_ind, (smpl_mesh, smpl_mesh_free, mesh_gt, image_gt, smpl_image, smpl_free_image) in enumerate(zip(smpl_meshes, smpl_free_meshes, meshes_gt, images_gt, smpl_images, smpl_free_images)):
                    seq_name = seq_names[b_ind]

                    dst_mesh_dir = os.path.join(self.logger.save_dir, split, 'meshes', f'epoch={self.current_epoch:05d}_step={self.global_step:05d}', seq_name)
                    dst_image_dir = os.path.join(self.logger.save_dir, split, f'images_{channel_label}', f'epoch={self.current_epoch:05d}_step={self.global_step:05d}', seq_name)
                    dst_smpl_image_dir = os.path.join(self.logger.save_dir, split, 'smpl_images', f'epoch={self.current_epoch:05d}_step={self.global_step:05d}', seq_name)
                    for path in [dst_mesh_dir, dst_image_dir, dst_smpl_image_dir]:
                        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

                    mesh_gt.export( os.path.join(dst_mesh_dir, f'gt_{channel_label}.obj'))
                    smpl_mesh.export( os.path.join(dst_mesh_dir, f'smpl.obj'))
                    smpl_mesh_free.export( os.path.join(dst_mesh_dir, f'smpl_free.obj'))
                    
                    dst_image_file = os.path.join(dst_image_dir, f'0.png')
                    dst_image_gt_file = os.path.join(dst_image_dir, f'gt_0.png')
                    dst_smpl_file = os.path.join(dst_smpl_image_dir, f'0_base.png')
                    dst_smpl_free_file = os.path.join(dst_smpl_image_dir, f'0.png')
                    
                    imsave(dst_image_gt_file, image_gt)
                    imsave(dst_smpl_file, smpl_image)
                    imsave(dst_smpl_free_file, smpl_free_image)
                    print(f'Exported visuals for {seq_name} in \n\t{dst_image_file} \n\t{dst_image_gt_file} \n\t {dst_smpl_file} \n\t {dst_smpl_free_file}')

            else:
                for b_ind, (mesh, smpl_mesh, smpl_mesh_free, mesh_gt, image, image_gt, smpl_image, smpl_free_image) in enumerate(zip(meshes, smpl_meshes, smpl_free_meshes, meshes_gt, images, images_gt, smpl_images, smpl_free_images)):
                    seq_name = seq_names[b_ind]

                    dst_mesh_dir = os.path.join(self.logger.save_dir, split, 'meshes', f'epoch={self.current_epoch:05d}_step={self.global_step:05d}', seq_name)
                    dst_image_dir = os.path.join(self.logger.save_dir, split, f'images_{channel_label}', f'epoch={self.current_epoch:05d}_step={self.global_step:05d}', seq_name)
                    dst_smpl_image_dir = os.path.join(self.logger.save_dir, split, 'smpl_images', f'epoch={self.current_epoch:05d}_step={self.global_step:05d}', seq_name)
                    for path in [dst_mesh_dir, dst_image_dir, dst_smpl_image_dir]:
                        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

                    mesh.export( os.path.join(dst_mesh_dir, f'{channel_label}.obj'))
                    mesh_gt.export( os.path.join(dst_mesh_dir, f'gt_{channel_label}.obj'))
                    smpl_mesh.export( os.path.join(dst_mesh_dir, f'smpl.obj'))
                    smpl_mesh_free.export( os.path.join(dst_mesh_dir, f'smpl_free.obj'))
                    
                    dst_image_file = os.path.join(dst_image_dir, f'0.png')
                    dst_image_gt_file = os.path.join(dst_image_dir, f'gt_0.png')
                    dst_smpl_file = os.path.join(dst_smpl_image_dir, f'0_base.png')
                    dst_smpl_free_file = os.path.join(dst_smpl_image_dir, f'0.png')
                    
                    imsave(dst_image_file, image)
                    imsave(dst_image_gt_file, image_gt)
                    imsave(dst_smpl_file, smpl_image)
                    imsave(dst_smpl_free_file, smpl_free_image)
                    print(f'Exported visuals for {seq_name} in \n\t{dst_image_file} \n\t{dst_image_gt_file} \n\t {dst_smpl_free_file} \n\t {dst_smpl_free_file}')
                
                    self.logger.log_image(key="test_image/renderings_{channel_label}", images=[image.astype(float), image_gt.astype(float), 
                                                                                                smpl_image.astype(float)])
            # self.logger.log_image(key="test_image/meshes", images=wandb.Object3D([mesh_c1_path, mesh_c2_path]))

    @torch.no_grad()
    def visualize(self, batch, channel, smpl_output, gt=False):
        
        channel_name = self.train_cfg['mri_labels'][channel]
        if channel_name =='LT':
            col = col_lt
        elif channel_name =='AT':
            col = col_at
        elif channel_name =='NO':
            col = col_no
        elif channel_name == 'BONE':
            col = col_bn
        else:
            raise ValueError(f"Unknown channel {channel_name}")

        if self.cfg.eval_export_visuals:
            mise_resolution0=64
        else:
            mise_resolution0=16
        if gt == True:
            extract_batch = batch
        else:
            extract_batch = None
        

        if gt == True:
        # if False:
            meshes = self.hit.extract_mesh(smpl_output, channel, use_mise=True, 
                                                  mise_resolution0=mise_resolution0, batch=extract_batch)
        else: 
            meshes = self.hit.extract_shaped_mesh( smpl_output, channel, use_mise=False, mise_resolution0=mise_resolution0)
            mesh_p_list = []
            for m_can in meshes:
                mesh_p = self.hit.pose_unposed_tissue_mesh(m_can, smpl_output, do_compress=False)
                mesh_p_list.append(mesh_p)
            # import ipdb; ipdb.set_trace()
            meshes = mesh_p_list
        
        global_orient_init = None
        smpl_vertices = smpl_output.vertices.clone()
        if 'global_orient_init' in batch:  # normalize visualization wrt the first frame
            global_orient_init = smplx.lbs.batch_rodrigues(batch['global_orient_init'].reshape(-1, 3)) # B,3,3
            smpl_vertices = (global_orient_init.transpose(1, 2) @ smpl_vertices.transpose(1, 2)).transpose(1, 2)
            smpl_free_vertices = (global_orient_init.transpose(1, 2) @ batch['body_verts_free'].transpose(1, 2)).transpose(1, 2)
            for i, mesh in enumerate(meshes):
                v = torch.Tensor(mesh.vertices[None,:,:]).cuda()
                mesh.vertices = (global_orient_init.transpose(1, 2) @ v.transpose(1, 2)).transpose(1, 2).cpu().numpy()[0]
                mesh.visual.vertex_colors = col
                # mesh.vertices = (global_orient_init[i].T @ mesh.vertices.T).T # For some reason, this numpy version from original COAP code does not behave as the pytorch version so I used pytorch

        rnd_images, rnd_smpl_images, rnd_smpl_free_images = [], [], []
        for i, mesh in enumerate(meshes):
            # save image
            rnd_images.append((self.renderer.render_mesh(
                torch.tensor(mesh.vertices).float().unsqueeze(0).to(self.device),
                torch.tensor(mesh.faces).unsqueeze(0).to(self.device),
                torch.tensor(mesh.visual.vertex_colors).float().cuda()[None,...,:3]/255,
                mode='t'
            )[0].squeeze(0) * 255.).cpu().numpy().astype(np.uint8))
            
            # 
            rnd_smpl_images.append((self.renderer.render_mesh(
                smpl_vertices[i].float().unsqueeze(0).to(self.device),
                self.smpl.faces_tensor.unsqueeze(0).to(self.device),
                mode='p'
            )[0].squeeze(0) * 255.).cpu().numpy().astype(np.uint8))
            rnd_smpl_free_images.append((self.renderer.render_mesh(
                smpl_free_vertices[i].float().unsqueeze(0).to(self.device),
                self.smpl.faces_tensor.unsqueeze(0).to(self.device),
                mode='p'
            )[0].squeeze(0) * 255.).cpu().numpy().astype(np.uint8))            
            # if True:
            #     import pyrender
            #     _scene = pyrender.Scene(ambient_light=[.1, 0.1, 0.1], bg_color=[1.0, 1.0, 1.0])
            #     VIEWER = pyrender.Viewer(_scene, use_raymond_lighting=True, run_in_thread=True)
            #     VIEWER.scene.add(pyrender.Mesh.from_trimesh(mesh))
            #     VIEWER.scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(vertices=smpl_vertices[i].cpu().numpy(), faces=self.smpl.faces)))

        smpl_meshes = []
        for i in range(smpl_vertices.shape[0]):
            smpl_mesh = trimesh.Trimesh(vertices=smpl_vertices[i].cpu().numpy(), faces=self.smpl.faces_tensor.cpu().numpy())
            smpl_meshes.append(smpl_mesh)
        smpl_free_meshes = []
        for i in range(smpl_free_vertices.shape[0]):
            smpl_free_mesh = trimesh.Trimesh(vertices=smpl_free_vertices[i].cpu().numpy(), faces=self.smpl.faces_tensor.cpu().numpy())
            smpl_free_meshes.append(smpl_free_mesh)
        return meshes, smpl_meshes, smpl_free_meshes, rnd_images, rnd_smpl_images, rnd_smpl_free_images
      

    
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
        ckp_monitor = 'val/beta_loss'
        filename='model-epoch={epoch:04d}-val_accuracy={val/beta_loss:.6f}'
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
        ckpt_path = os.path.join(checkpoint_callback.dirpath, f"{checkpoint_callback.CHECKPOINT_NAME_LAST}.ckpt")
        # from utils.exppath import Exppath
        # exppath = Exppath(cfg.exp_name)
        # ckpt_path = exppath.get_best_checkpoint()
        # ckpt_path = checkpoint_callback.best_model_path # this is not working for some reason
    elif os.path.exists(os.path.join(checkpoint_callback.dirpath, f"{checkpoint_callback.CHECKPOINT_NAME_LAST}.ckpt")):
        ckpt_path = os.path.join(checkpoint_callback.dirpath, f"{checkpoint_callback.CHECKPOINT_NAME_LAST}.ckpt")
    else:
        ckpt_path = None
    # create model
    model = HITTraining(cfg)
    if cfg.run_eval:
        trainer.test(model, ckpt_path=ckpt_path, verbose=True)
    else:
        trainer.fit(model, ckpt_path=ckpt_path)
    

if __name__ == "__main__":
    main()