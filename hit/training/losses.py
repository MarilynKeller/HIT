 
# This file contains the loss functions used in the training of the model.

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
import wandb
from utils.smpl_utils import get_template_verts, mpimesh2glb
from utils.tensors import cond_create, eik_loss
from hit.utils.slice_extractor import SliceLevelSet


def compute_occupancy_loss(hit_pl, batch, pred_occ):
    w_list = []
    n_points = batch['mri_occ'].shape[0] * batch['mri_occ'].shape[1] # = B*T
    for ci in range(len(hit_pl.train_cfg['mri_labels'])):
        wi = 1-(torch.sum(batch['mri_occ'] == float(ci))/n_points)
        w_list.append(wi)
        
    weight = torch.hstack(w_list)

    if len(hit_pl.train_cfg.mri_labels) == 1:
        if hit_pl.data_cfg.synthetic:
            target_occ_val = 1 # The synthetic data loader returns 0 or 1 depending if the point is inside or outside the SMPL mesh
        else:
            if hit_pl.train_cfg.mri_labels == ['LT']:
                target_occ_val = 1
            elif hit_pl.train_cfg.mri_labels == ['AT']:
                target_occ_val = 2
            else:
                raise ValueError(f"Unknown tissue type {hit_pl.train_cfg.mri_labels}")
        occ_loss = F.binary_cross_entropy_with_logits(pred_occ, (batch['mri_occ']==target_occ_val).float())
    else:
        occ_loss = F.cross_entropy(input = pred_occ.reshape(-1, pred_occ.shape[-1]),  # (BxT, 4)
                            target = batch['mri_occ'].long().reshape(-1),  # (BxT)
                            weight = weight)
        
    occ_loss = hit_pl.train_cfg.lambda_occ * occ_loss
    hit_pl.log("train/loss_occ", occ_loss)
    
    # Add per part occupancy loss (volumetric)
    # part_loss = metrics.compute_loss_per_part(batch=batch, smpl_body=hit_pl.smpl_body, pred_occ=pred_occ, weight=weight)
    # for part in part_loss:
    #         wandb.log({f"Per Part occupancy loss/{part}": part_loss[part]})
                
    return occ_loss


def compute_beta_disp_loss(hit_pl, batch, batch_idx, cond, step, train_cfg):
    """Force the beta dislacement field to yield the same displacement as the SMPL model"""
    
    if train_cfg.random_beta_disp:
        betas = (torch.rand(batch['betas'].shape).to(hit_pl.device)-0.5) * 4
        cond = cond_create(betas)
    else:
        betas = batch['betas']
    
    v_s = hit_pl.smpl.forward_canonical(betas).vertices # Vertex on the shaped mesh   
    v_c = hit_pl.smpl.forward_canonical(torch.zeros_like(betas)).vertices # Vertex on the canonical mesh
    # v_c_pred = hit_pl.hit.deformer.query_cano(v_s, {'betas': cond['betas']}, apply_pose_bs=False)
    disp = hit_pl.hit.deformer.query_betadisp(v_s, cond)
    # import ipdb; ipdb.set_trace()
    v_c_pred = v_s + disp
    
    beta_loss = F.mse_loss(v_c_pred*100, v_c*100) # We pass the metric values in cm for better numerical stability
    
    # Per part loss for cannonical loss is not volumetric
    # part_loss = metrics.compute_loss_per_part_p2p(smpl_canonical_pred*100, smpl_canonical_gt*100, loss_type='canonical', part_ids=hit_pl.smpl.part_ids) 
    # for part in part_loss:
    #     wandb.log({f"Per Part beta disp loss/{part}": part_loss[part]})    
    if hit_pl.current_epoch %hit_pl.cfg.export_canmeshes_every_n_epochs == 0 and step == 'train':
        mesh = hit_pl.smpl.psm(v_c_pred,  display=False)
        mesh_path = mpimesh2glb(mesh)
        wandb.log({f"train_can_space/unshape_pred": wandb.Object3D(mesh_path)})
        
        mesh = hit_pl.smpl.psm(v_c, v_c-v_c_pred, maxval=0.05, display=False)
        mesh_path = mpimesh2glb(mesh)
        wandb.log({f"train_can_space/beta_disp_with_pred_error": wandb.Object3D(mesh_path)})
         
    # import ipdb; ipdb.set_trace()
    beta_loss = hit_pl.train_cfg.lambda_betas * beta_loss
    hit_pl.log(f"{step}/beta_loss", beta_loss)
    return beta_loss

    # wandb.log({"train/can_surf_pts": wandb.Object3D(pts_c_surface[0].detach().cpu().numpy())})
    # wandb.log({"train/target_can_surf_pts": wandb.Object3D(smpl_template_verts[0].detach().cpu().numpy())})
    
def Bdense_loss(x_s, x_c, B_xc, beta):
    
    fwd_disp = torch.matmul(B_xc.transpose(-2,-1), beta.unsqueeze(1).repeat(1,x_c.shape[1], 1).unsqueeze(-1)).squeeze(-1) # B, N, 3
    
    # beta_rep = beta.repeat(x_c.shape[1], 1).reshape(-1,1,10) # Repeat the betas for each point (B*N,1, 10)
    # B_xc_concat = B_xc.reshape(-1, 10, 3) # B*N, 10, 3
    # fwd_disp_flat = torch.bmm(beta_rep, B_xc_concat) # B*N, 1, 3
    # fwd_disp = fwd_disp_flat.view(-1, x_c.shape[1], 3) # B, N, 3
    # import ipdb; ipdb.set_trace()
    linear_beta_loss = F.mse_loss(x_s - x_c, fwd_disp)
    return linear_beta_loss
    
 
def compute_linear_beta_loss(hit_pl, batch, cond, smpl_output, pts_c_inside, on_surface=False):
    # Enforce disp linearity wrt beta  
    
    # shaped_verts = hit_pl.smpl.forward_canonical(cond['betas']).vertices
    
    # x_s = torch.rand(batch['betas'].shape[0], 6000, 3).to(hit_pl.device)*2-1
    # disp = hit_pl.hit.deformer.query_betadisp(x_s, cond)
    # x_c = x_s + disp
    
    with torch.no_grad():
        if on_surface:
            x_s = hit_pl.smpl.forward_canonical(batch['betas']).vertices
            x_c =  hit_pl.smpl.forward_canonical(torch.zeros_like(batch['betas'])).vertices

        else:
            x_m = batch['mri_points'] 
            x_c = pts_c_inside
            x_s = hit_pl.hit.canonicalize_from_similar(x_m, x_c, smpl_output.tfs, cond, undo_shape=False)
            # x_c = hit_pl.hit.canonicalize_from_similar(pts_p, pts_c, smpl_output.tfs, cond, undo_shape=True)

    # We want to enforce the displacement to be a linear function of beta: xb-xc = beta * B(xc)
    # import ipdb; ipdb.set_trace()
    B_flat = hit_pl.hit.fwd_beta(x_c, cond) # note that fwd_beta is not conditioned on anything
    B_xc = B_flat.view(B_flat.shape[0],x_c.shape[1],10,3) # B, N, 10, 3 The  predicted local B vector in x_c
    
    linear_beta_loss = Bdense_loss(x_s, x_c, B_xc, batch['betas']) # Forces the displacement to be a linear function of beta
    
    linear_beta_loss = hit_pl.train_cfg.lambda_linear_beta * linear_beta_loss
    hit_pl.log("train/linear_beta_loss", linear_beta_loss)
    return linear_beta_loss
    

def pose_bs_loss(hit_pl, batch, batch_idx, cond):
        
        zero_beta = torch.zeros_like(batch['betas'])
        smpl_template_verts = hit_pl.smpl.forward_canonical(zero_beta).vertices
        
        smpl_posed = hit_pl.smpl.forward(zero_beta, body_pose=batch['body_pose'], global_orient=batch['global_orient'], transl=batch['transl'])

        xv_p = smpl_posed.vertices
        xv_c = smpl_template_verts
        # xf_p = batch['body_verts_free']
        

        # smpl_tfs = smpl_output.tfs
        # xv_c_pbs = hit_pl.hit.canonicalize_from_similar(xv_p, xv_c, smpl_tfs, cond, undo_shape=False)
        xv_c_pbs = hit_pl.smpl.forward_shaped(zero_beta, batch['body_pose'], global_orient=None)
        # xv_c_pbs = hit_pl.smpl.forward_shaped(batch['betas'], batch['body_pose'], global_orient=None)
        
        d = xv_c - xv_c_pbs
        d_pred = hit_pl.hit.deformer.pose_bs(xv_c, cond)
             
        if hit_pl.current_epoch % hit_pl.cfg.export_canmeshes_every_n_epochs == 0:
            mesh = hit_pl.smpl.psm(xv_c_pbs-d_pred, d_pred, display=False)
            mesh_path = mpimesh2glb(mesh)
            wandb.log({f"train_can_space/pose_dependant_blendshapes": wandb.Object3D(mesh_path)})
            
            mesh = hit_pl.smpl.psm(xv_c_pbs-d, d, display=False)
            mesh_path = mpimesh2glb(mesh)
            wandb.log({f"gt_can_space/pose_dependant_blendshapes": wandb.Object3D(mesh_path)})
            
        bs_loss = F.mse_loss(d_pred*100, d*100) # Convert to cm with *100 for better numerical stability

        bs_loss = hit_pl.train_cfg['lambda_pose_bs'] * bs_loss
        hit_pl.log("train/loss_pose_bs", bs_loss)
        return bs_loss
    

def compute_outside_can_loss(hit_pl, smpl_output, batch, batch_idx):
    # Empty canonical space outside the body loss
    if not "can_points" in batch:
        raise DeprecationWarning("The batch should contain can_points, the following is slow and deprecated")
        raise Exception("The batch should contain can_points")
        # Sample uniform points in the canonical space
        xf_p = batch['body_verts_free']
        B = pts.shape[0]
        nb_points_canspace = hit_pl.data_cfg['nb_points_canspace']
        pts_c_uniform = sample_uniform_from_max((B, nb_points_canspace, 3), xf_p.max(), padding=0.30)
        if hit_pl.train_cfg['compressor']:
            
            # Get unposed tight body shape

            smpl_tfs = smpl_output.tfs #get_gdna_bone_transfo(self.hit.smpl, smpl_output)
            xf_c = hit_pl.hit.canonicalize_from_similar(xf_p, smpl_template_verts, smpl_tfs, cond)
            # The unposed points should fit inside the compressed body
            x_c = xf_c
            
        else:
            # The unposed points should fit inside the smpl template body
            x_c = smpl_template_verts
        
        # TODO The sampling could be done once for all in the dataloader - > not really bcs personalised shape
        # But can be done when compression is integrated
        def compute_outside_mask(x_c, faces, pts):
            B = pts.shape[0]
            T = pts.shape[1]
            outside_mask = torch.zeros((B, T), dtype=torch.bool, device=pts.device)
            for bi in range(B):
                mesh_can_f = trimesh.Trimesh(vertices=x_c[bi].detach().cpu().numpy(), faces=faces, process=False)
                # query = trimesh.proximity.ProximityQuery(mesh_can_f)
                isinside = mesh_can_f.contains(pts[bi].detach().cpu().numpy())
                outside_bi = np.logical_not(isinside)
                outside_mask[bi] = torch.tensor(outside_bi, dtype=torch.bool, device=pts.device)
            return outside_mask
        
        outside_can_mask = compute_outside_mask(x_c, hit_pl.smpl.faces, pts_c_uniform)
    else:
        pts_c_uniform = batch['can_points']
        outside_can_mask = torch.logical_not(batch['can_occ'])
          
    uniform_can_pts_pred = hit_pl.forward(batch, pts_c_uniform, template=True)['pred_occ']
    uniform_can_pts_pred = F.softmax(uniform_can_pts_pred, dim=-1)[...,0]
    # WARN: Is this what you want to do
    nb_pts_outside = outside_can_mask.sum()
    
    assert nb_pts_outside > 0, "There should be points outside the canonical space"
    
    gt_outside_cc = torch.ones(nb_pts_outside).to(uniform_can_pts_pred.device) # The class of all those points should be zero
    # import ipdb; ipdb.set_trace()
    loss_outside_can = F.mse_loss(uniform_can_pts_pred[outside_can_mask], gt_outside_cc)
    
    
    # if hit_pl.current_epoch %hit_pl.cfg.export_canmeshes_every_n_epochs == 0:
    #     # list points predicted outside the canonical space
    #     B = pts_c_uniform.shape[0]

    #     pred = torch.softmax(uniform_can_pts_pred, dim=-1) #B, T, C
    #     pred_tissue = pred.argmax(dim=-1)
    #     pred_mask = torch.where(pred_tissue == 0)
    #     outside = pts_c_uniform[pred_mask][0::B]
        # wandb.log({"train_can_space/can_surf_pts": wandb.Object3D(outside.detach().cpu().numpy())})
        
    loss_outside_can = hit_pl.train_cfg.lambda_outcan * loss_outside_can
    hit_pl.log("train/loss_outside_can", loss_outside_can)
    
    return loss_outside_can


def compute_hands_loss(hit_pl, batch):
    # Forces the hands inside to be LT
    hands_pts = batch['hands_can_points']
    hands_occ = batch['hands_can_occ']
    
    gt_hands_occ = torch.zeros_like(hands_occ)
    gt_hands_occ[hands_occ.bool()] = 1 # LT class
    
    pred_hands = hit_pl.forward(batch, hands_pts, template=True)['pred_occ']
    loss_hands = F.cross_entropy(pred_hands.reshape(-1, len(hit_pl.train_cfg.mri_labels)), gt_hands_occ.long().reshape(-1))
    
    loss_hands = hit_pl.train_cfg.lambda_hands * loss_hands
    hit_pl.log("train/loss_hands", loss_hands)
            
    return loss_hands
            
            
def compute_compression_loss(hit_pl, batch, batch_idx, smpl_output, cond, step='train'):

        smpl_template_verts = get_template_verts(batch, hit_pl.smpl)
        smpl_tfs = smpl_output.tfs
        
        xv_p = batch['body_verts']
        xf_p = batch['body_verts_free']
             
        xf_s = hit_pl.hit.canonicalize_from_similar(xf_p, smpl_template_verts, smpl_tfs, cond, undo_shape=False)
        # xf_c = hit_pl.hit.canonicalize_from_similar(xf_p, smpl_template_verts, smpl_tfs, cond, undo_shape=False)
        xv_s = hit_pl.smpl.forward_canonical(batch['betas']).vertices # Vertex on the shaped mesh   
        # compute shaped on
        
        d = xv_p - xf_p # Ground truth decompression offset
        d_pred = hit_pl.hit.deformer.compressor(xf_s, cond) # Predicted decompression offset
        
        vi = 3504 #vertex index on the belly
    
        loss_fct_comp = torch.nn.MSELoss()
        
        # Convert to mm for more meaningful loss
        d = d * 1000
        d_pred = d_pred * 1000
        
        loss_compression = loss_fct_comp(d_pred, d)
        
        if step == 'train':
            hit_pl.log("train/d_err_belly", ((torch.abs(d_pred-d)[:,vi])).mean())
                                       
        elif step == 'val':
            
            hit_pl.log("val_compression/loss", loss_compression)
            hit_pl.log("val_compression/d_err_belly", ((torch.abs(d_pred-d)[:,vi])).mean())
            hit_pl.log("val_compression/d_err_max", torch.linalg.norm(d_pred-d, axis=-1).max())
        
            # Log meshes
            if hit_pl.current_epoch % hit_pl.cfg.export_canmeshes_every_n_epochs == 0:
                mesh = hit_pl.smpl.psm(xv_s, d_pred, maxval=0.08, norm=False, display=False)
                mesh_path = mpimesh2glb(mesh)
                wandb.log({f"train_can_space/shape_space_compression": wandb.Object3D(mesh_path)})
                
                mesh = hit_pl.smpl.psm(xf_p+d_pred, display=False)
                mesh_path = mpimesh2glb(mesh)
                wandb.log({f"train_can_space/uncompressed_body": wandb.Object3D(mesh_path)})
                
                mesh = hit_pl.smpl.psm(xv_s, d, maxval=0.08, norm=False, display=False)
                mesh_path = mpimesh2glb(mesh)
                wandb.log({f"gt_can_space/shape_space_compression": wandb.Object3D(mesh_path)})
                
                mesh = hit_pl.smpl.psm(xv_p, d-d_pred, maxval=0.08, norm=False, display=False)
                mesh_path = mpimesh2glb(mesh)
                wandb.log({f"gt_can_space/uncompressed_body_with_error": wandb.Object3D(mesh_path)})
        
            

        # def compression_loss_metric(d, d_pred, loss_fct_comp):
            
        #     for i in range(3):
        #         print(f"d range for dim {i}: min={d[...,i].min():.3f}, max={d[...,i].max():.3f}")
        #         di_max = d[...,i].max()
        #         di_min = d[...,i].min()

         
        loss_compression = hit_pl.train_cfg.lambda_compression * loss_compression
        hit_pl.log(f"{step}/compression_loss", loss_compression)
       
        return loss_compression
    

   
def compute_lbs_regularization_loss(hit_pl, batch, batch_idx, cond, step='train'):
    """ Forces the lbs to be equal to those of SMPL on the skin surface""" 
    
    smpl_canonical = hit_pl.smpl.forward_canonical(torch.zeros_like(batch['betas']))
    smpl_template_verts = smpl_canonical.vertices
    

    gt_weights = hit_pl.smpl.lbs_weights.unsqueeze(0).expand(batch['betas'].shape[0], -1, -1).to(smpl_template_verts.device)
    # import ipdb; ipdb.set_trace()
    # print(gt_weights.max())
    
    pred_weights = hit_pl.hit.deformer.query_weights(smpl_template_verts, {'latent': cond['lbs']})

    if (hit_pl.current_epoch % hit_pl.cfg.export_canmeshes_every_n_epochs) == 0 and step == 'train':
        mesh = hit_pl.smpl.psm(smpl_template_verts, skin_weights=pred_weights, display=False)
        mesh_path = mpimesh2glb(mesh)
        wandb.log({f"train_can_space/can_lbs": wandb.Object3D(mesh_path)})
        
        mesh = hit_pl.smpl.psm(smpl_template_verts, skin_weights=gt_weights, display=False)
        mesh_path = mpimesh2glb(mesh)
        wandb.log({f"gt_can_space/can_lbs": wandb.Object3D(mesh_path)})
        
        # difference
        mesh = hit_pl.smpl.psm(smpl_template_verts, pred_weights-gt_weights, display=False, norm=True)
        mesh_path = mpimesh2glb(mesh)
        wandb.log({f"train_can_space/can_lbs_error": wandb.Object3D(mesh_path)})
    
    # mse loss
    if hit_pl.train_cfg['lbs_loss_type'] == 'mse':
        lbs_loss = F.mse_loss(pred_weights, gt_weights)
    elif hit_pl.train_cfg['lbs_loss_type'] == 'cosine_similarity':
        lbs_loss = 1 - F.cosine_similarity(pred_weights, gt_weights, dim=-1).mean()
    elif hit_pl.train_cfg['lbs_loss_type'] == 'lbs_part_loss':
        # Write explicit loss
        # pred_weights is of shape BxNxK, where K is the number of parts. Compute the error for each part and take the mean
        lbs_loss = (pred_weights - gt_weights).pow(2).sum(dim=-1).mean()
    elif hit_pl.train_cfg['lbs_loss_type'] == 'non_zero_loss':
        # Compute the loss only for the non zero weights
        non_zero_mask = gt_weights > 0
        lbs_loss = F.mse_loss(pred_weights[non_zero_mask], gt_weights[non_zero_mask])
        # add it to the standard mse loss
        lbs_loss = lbs_loss + F.mse_loss(pred_weights[~non_zero_mask], gt_weights[~non_zero_mask])
    else:
        raise ValueError(f"Unknown lbs loss type {hit_pl.train_cfg['lbs_loss_type']}, should be 'mse' or 'lbs_part_loss'")
    
    lbs_loss = hit_pl.train_cfg.lambda_lbs * lbs_loss
    hit_pl.log(f"{step}/loss_lbs", lbs_loss)
    return lbs_loss
    

    
def compute_smooth_loss(hit_pl, batch, pts_c_inside):
    raise DeprecationWarning("This is deprecated, check first if you use")
    pred_occ_can = hit_pl.forward(batch, pts_c_inside, template=True, eval_mode=True)['pred_occ']
    smooth_loss = eik_loss(pred_occ_can, pts_c_inside)
    
    smooth_loss = hit_pl.train_cfg.lambda_eikonal * smooth_loss
    hit_pl.log("train/smooth_loss", smooth_loss)
    return smooth_loss

def compute_beta0_loss(hit_pl, batch, batch_idx, cond):

    pts_c_uniform = batch['can_points'] # pts sampled uniformely in the canonical space
    pred_disp = hit_pl.hit.deformer.disp_network(pts_c_uniform, cond)  
    loss_beta0 = torch.linalg.norm(pred_disp)
    
    # sl = SliceLevelSet()
    # z0 =0
    # xc = sl.gen_slice_points(z0=z0)
    # xc_torch = torch.FloatTensor(xc).to(hit_pl.device).unsqueeze(0).expand(batch['betas'].shape[0], -1, -1)
    # pred_disp = hit_pl.hit.deformer.disp_network(xc_torch, cond)  
    
    # sl.plot_slice_value(pred_disp[0].detach().cpu().numpy(), to_plot=True)
    loss_beta0 = hit_pl.train_cfg.lambda_beta0 * loss_beta0
    hit_pl.log("train/loss_beta0", loss_beta0)   
    
    # Per part loss for zero deformation loss is not volumetric
    # part_loss = metrics.compute_loss_per_part_p2p(pred_disp, None, loss_type='zero_disp', part_ids=hit_pl.smpl.part_ids) 
    # for part in part_loss:
    #     wandb.log({f"Per Part zero_disp loss/{part}": part_loss[part]})
    
    return loss_beta0    

def compute_surface_loss(hit_pl, batch, batch_idx, cond):
    """ Forces the occupancy of NO to be of 0.5% on the surface of the SMPL mesh"""
    
    smpl_can_verts = hit_pl.smpl.forward_canonical(torch.zeros_like(batch['betas'])).vertices
    
    pred_occ = hit_pl.forward(batch, smpl_can_verts, template=True)['pred_occ']
    pred_occ_NO = torch.softmax(pred_occ, dim=-1)[...,0] # softmax over the tissues, then take the first channel value (NO)
    
    pred_gt = 0.5*torch.ones_like(pred_occ_NO)
    
    surf_loss = F.mse_loss(pred_occ_NO, pred_gt)
    surf_loss = hit_pl.train_cfg.lambda_surf * surf_loss
    hit_pl.log("train/surf_loss", surf_loss)
    return surf_loss
    
    
    
def compute_comp0_loss(hit_pl, batch, batch_idx, smpl_output, cond, pts_c, occ_pred):
    """ Forces the compression vector to be week for LT and BONE points
    pts_c: points sampled inside the MRI BxTx3
    occ_pred: occupancy prediction for those points BxTxC
    """

    pts_p = batch['mri_points'] 

    # pts_c = batch['body_verts']
    # xf_p = batch['body_verts_free']
    
    smpl_tfs = smpl_output.tfs #get_gdna_bone_transfo(smpl, smpl_output)
    xf_s = hit_pl.hit.canonicalize_from_similar(pts_p, pts_c, smpl_tfs, cond, undo_shape=False)
    
    # compression_prediction
    d_pred = hit_pl.hit.deformer.compressor(xf_s, cond)
    
    d_pred = d_pred * 100 # Convert to cm for better numerical stability
    
    # Use gt prediction
    bone_mask = batch['mri_occ'] == 3
    lt_mask = batch['mri_occ'] == 1
    
    
    # Use pred prediction
    # bone_mask = torch.softmax(occ_pred).argmax(dim=-1) == 3
    # lt_mask = torch.softmax(occ_pred).argmax(dim=-1) == 1
    
    bone_comp_loss = F.mse_loss(d_pred[bone_mask], torch.zeros_like(d_pred[bone_mask])) # Convert to cm with *100 for better numerical stability
    lt_comp_loss = F.mse_loss(d_pred[lt_mask], torch.zeros_like(d_pred[lt_mask])) # Convert to cm with *100 for better numerical stability
    
    if hit_pl.train_cfg['comp0_loss_type'] == 'bt_lt':
        comp0_loss = bone_comp_loss + 0.2*lt_comp_loss
    elif hit_pl.train_cfg['comp0_loss_type'] == 'bt':
        comp0_loss = bone_comp_loss 
    elif hit_pl.train_cfg['comp0_loss_type'] == 'norm': # Just force the compression to be small
        comp0_loss = F.mse_loss(d_pred, torch.zeros_like(d_pred))
    else:
        raise ValueError(f"Unknown comp0 loss type {hit_pl.train_cfg['comp0_loss_type']}, should be 'bt_lt' or 'bt'")
    
    if hit_pl.train_cfg['comp0_out']:
        # To fix. This id not that easy
        raise NotImplementedError("This is not implemented yet")
        mri_out_pts = batch['mri_out_pts']
        xf_s = hit_pl.hit.canonicalize_from_similar(mri_out_pts, pts_c, smpl_tfs, cond, undo_shape=False)
        d_pred = hit_pl.hit.deformer.compressor(xf_s, cond)
        # import ipdb; ipdb.set_trace()
        comp0_loss_out = F.mse_loss(d_pred[mri_out_pts]*100, torch.zeros_like(d_pred[mri_out_pts])) # Convert to cm with *100 for better numerical stability
        comp0_loss = comp0_loss + comp0_loss_out
   
    comp0_loss = hit_pl.train_cfg.lambda_comp0 * comp0_loss
    hit_pl.log("train/comp0_loss", comp0_loss)
    return comp0_loss


def compute_mri_loss(hit_pl, batch, batch_idx, smpl_output, cond, pts_c_inside):
    # @varora
    # pts_c_inside are the 3d coords of the sampled mri points inside the canonical space, BxTx3
    mri_values_gt = batch['mri_values'] # BxT, between 0 and 1
    mri_values_pred = hit_pl.hit.mri_val_net(pts_c_inside, cond) # BxT, between 0 and 1

    # l2 loss
    #mri_loss = F.mse_loss(mri_values_pred, mri_values_gt) # Convert to cm with *100 for better numerical stability

    # inside loss
    mri_loss = F.l1_loss(mri_values_pred, mri_values_gt)     # l1 loss
    mri_loss = hit_pl.train_cfg.lambda_mri * mri_loss  # weight the loss
    hit_pl.log("train/mri_loss", mri_loss)

    # outside loss
    pts_c_uniform = batch['can_points']
    outside_can_mask = torch.logical_not(batch['can_occ'])
    uniform_mri_values_pred = hit_pl.hit.mri_val_net(pts_c_uniform, cond)
    nb_pts_outside = outside_can_mask.sum()
    gt_outside_cc = (torch.zeros(nb_pts_outside).to(uniform_mri_values_pred.device)).type(torch.float32)
    # The mri values of all those points should be zero
    #loss_outside_mri = F.l1_loss(uniform_mri_values_pred[outside_can_mask], gt_outside_cc[...,None])
    loss_outside_mri = F.mse_loss(uniform_mri_values_pred[outside_can_mask], gt_outside_cc[...,None])
    loss_outside_mri = hit_pl.train_cfg.lambda_outmri * loss_outside_mri  # weight the loss

    # log image
    sl = SliceLevelSet(res=0.004)
    axis = 'y'  # frontal cut
    z0 = -0.02  # coordinate  of the cut

    xc = sl.gen_slice_points(z0=z0, axis=axis)
    xc_batch = torch.FloatTensor(xc).to(hit_pl.device).unsqueeze(0).expand(1, -1, -1)
    mri_value = hit_pl.hit.mri_val_net(xc_batch, cond)

    _, image = sl.plot_slice_value(mri_value, to_plot=False, mri_values=True)  # This is the SMPL frontal cut image of the mri value

    # log to wandb
    wandb.log({f"train/mri_value_slice": wandb.Image(image)})
    # log mri_loss
    wandb.log({f"train/mri_loss": mri_loss})
    # log outside loss
    wandb.log({"train/loss_outside_mri": loss_outside_mri})
    # log total loss
    total_mri_loss = mri_loss + loss_outside_mri
    wandb.log({"train/total_mri_loss": total_mri_loss})

    plt.close()
    return total_mri_loss

    
def compute_canonicalization_loss(hit_pl, batch, batch_idx, smpl_output, cond, step='train'):

        # Get templte body surface
        xc = get_template_verts(batch, hit_pl.smpl)
        smpl_tfs = smpl_output.tfs
        
        # Get canonicalized body surface
        xf_p = batch['body_verts_free']         
        xf_c = hit_pl.hit.deformer.query_cano(xf_p, {'betas': cond['betas'],'thetas': cond['thetas']}) 
        
        # Convert to mm for more meaningful loss
        xf_c = xf_c * 1000
        xc = xc * 1000
        
        loss_fct = torch.nn.MSELoss()
        loss_canonicalization = loss_fct(xf_c, xc)
  
        loss_canonicalization = hit_pl.train_cfg.lambda_canonicalization * loss_canonicalization
        hit_pl.log(f"{step}/canonicalization_loss", loss_canonicalization)
       
        return loss_canonicalization