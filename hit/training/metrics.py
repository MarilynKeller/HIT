import os

import utils.figures
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from utils.video import make_gif, make_video
from skimage.io import imsave
from hit.training.dataloader_mri import MRI_SHAPE


def compute_iou(occ1, occ2, level=0.5):
    """ Computes the Intersection over Union (IoU) value for two sets of occupancy values.

    Args:
        occ1 (tensor): first set of occupancy values  (B, N)
        occ2 (tensor): second set of occupancy values  (B, N)
        level (float): threshold

    Returns:
        iou (tensor): mean IoU (scalar)
    """
    if occ1.dtype != torch.bool:
        occ1 = (occ1 >= level)
        occ2 = (occ2 >= level)

    # Compute IOU
    area_union = (occ1 | occ2).float().sum(axis=-1)
    area_intersect = (occ1 & occ2).float().sum(axis=-1)

    iou = (area_intersect / area_union.clamp_min(1))

    return iou.mean()


def compute_dice(occ1, occ2, empty_score=1.0, level=0.5):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    occ1 : array-like, bool
        Array of size (B,N), where N is the number of points predicted
    occ2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score
        The dice is computed by batch and the averaged over the batches.
        
    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `occ1` and `occ2` are switched.
    """
    if occ1.dtype != torch.bool:
        occ1 = (occ1 >= level)
        occ2 = (occ2 >= level)

    # Compute IOU
    area_sum = occ1.float().sum(axis=-1) + occ2.float().sum(axis=-1)
    area_intersect = (occ1 & occ2).float().sum(axis=-1)
    
    batch_dice = torch.nan_to_num((2. * area_intersect / area_sum), 0.0) # Some batches can be an area sum =0, leading to NaNs

    # Compute Dice coefficient
    # import ipdb; ipdb.set_trace()
    return batch_dice.mean()


def compute_per_class_accuracy(occ_gt, occ_pred):
    
    tp = torch.sum((occ_gt == 1) & (occ_pred == 1))
    tn = torch.sum((occ_gt == 0) & (occ_pred == 0))
    
    fp = torch.sum((occ_gt == 0) & (occ_pred == 1))
    fn = torch.sum((occ_gt == 1) & (occ_pred == 0))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    # f1 = 2 * (precision * recall) / (precision + recall)
    
    return accuracy, precision, recall
    
    
    

def compute_hd(occ_pred, occ_gt, mri_coords, mri_shape, spacing, level=0.5):
    """
    Computes Hausdorff, a measure of set similarity.
    Parameters
    ----------
    occ_gt : array of GT occ  (B, N)            
    occ_pred : array of pred occ   (B, N) 
    mri_coords : integegre 3D coords of each point in the MR (B, N, 3)
    mri_shape: shape of the original MRI array (H, W, D)
    spacing: resolution of the original MRI array (3)

    Returns
    -------
    hd : float
        If prediction is empty (while ground truth isn't) = np.inf
        
    Notes
    -----
    Make sure the spacing of im_gt and im_pred are same and correct. 
    SimpleITK comsiders this spacing to compute HD. 
    """
    
    
    occ_gt = occ_gt.cpu().numpy().reshape(1, *mri_shape) # B, H, W, D
    occ_pred = occ_pred.cpu().numpy().reshape(1, *mri_shape) # B, H, W, D
    spacing = spacing.cpu().numpy()
    # mri_coords = mri_coords.cpu().numpy()
    
    if occ_gt.dtype != np.bool:
        occ_gt = (occ_gt >= level).astype(np.float32)
        occ_pred = (occ_pred >= level).astype(np.float32)
        
    
    batches_hd = []
    
    for batch_id in range(occ_gt.shape[0]):
        # im_gt_array = np.zeros(mri_shape)
        # im_pred_array = np.zeros(mri_shape)
        
        # idx = mri_coords[batch_id]
        # im_gt_array[idx[:, 0], idx[:, 1], idx[:, 2]] = occ_gt[batch_id]
        # im_pred_array[idx[:, 0], idx[:, 1], idx[:, 2]] = occ_pred[batch_id]
        
        im_gt_array = occ_gt[batch_id]
        im_pred_array = occ_pred[batch_id]
        
        # import matplotlib.pyplot as plt
        # plt.imshow(im_gt_array[int(im_gt_array.shape[0]//2),:, :] - im_pred_array[int(im_pred_array.shape[0]//2),:, :])
        # plt.imshow(im_gt_array[:,int(im_gt_array.shape[1]//2), :] - im_pred_array[:,int(im_pred_array.shape[1]//2), :])
        # plt.imshow(im_gt_array[:,:,int(im_gt_array.shape[2]//2)] - im_pred_array[:,:,int(im_pred_array.shape[2]//2)])
        # plt.show()
        
        # for i in range(im_gt_array.shape[2]):
        #     plt.imshow(im_gt_array[:,:,i] - im_pred_array[:,:,i])
        #     plt.show()
        
        if occ_pred.sum() == 0:
            hd = 100
            
        else:
            import SimpleITK as sitk
            im_gt = sitk.GetImageFromArray(im_gt_array)
            im_pred = sitk.GetImageFromArray(im_pred_array)

            im_gt.SetSpacing(spacing[batch_id].tolist())
            im_pred.SetSpacing(spacing[batch_id].tolist())

            hd_filter = sitk.HausdorffDistanceImageFilter()            
            hd_filter.Execute(im_gt, im_pred)
            hd = hd_filter.GetHausdorffDistance()

            batches_hd.append(hd)
            
        hd = np.mean(batches_hd)
    return torch.tensor(hd)



def per_slice_prediction(occ_pred, occ_gt, mri_uvs, mri_shape, out_folder, subj_name, level=0.5, return_image: bool=False):
    
    # Define colors for the visualization
    gt_col = [0.5,0.9,0.3] #light green
    pred_col = [0.9,0.5,0] #orange
    intersect_col = [1,1,1]
    colors = [pred_col, gt_col, intersect_col]

    
    # The test points were sampled on a grid, so we can use the same grid to reconstruct the image
    # occ_gt = occ_gt.cpu().numpy().reshape(1, *mri_shape) # B, H, W, D
    # occ_pred = occ_pred.cpu().numpy().reshape(1, *mri_shape) # B, H, W, D
    
    # Converting prediction to float32 binary
    if occ_gt.dtype != bool:
        occ_gt = (occ_gt >= level)[0].cpu().numpy().astype(np.float32)
        occ_pred = (occ_pred >= level)[0].cpu().numpy().astype(np.float32)
    mri_uvs = mri_uvs[0].cpu().numpy().astype(np.int32)
      
    # Creating a 3D array of zeros and filling it with the occ values  
    occ_gt_3D_array = np.zeros(MRI_SHAPE)
    occ_pred_3D_array = np.zeros(MRI_SHAPE)
    
    occ_gt_3D_array[mri_uvs[:,0], mri_uvs[:,1], mri_uvs[:,2]] = occ_gt
    occ_pred_3D_array[mri_uvs[:,0], mri_uvs[:,1], mri_uvs[:,2]] = occ_pred
    
    # Artificially add a batch dimension for backward compatibility with the following code
    occ_gt = occ_gt_3D_array[None]
    occ_pred = occ_pred_3D_array[None]
    
    
    # Empty mri slice
    im_zero = np.zeros(MRI_SHAPE[:2])
      
    batch_imgs = [[]] * occ_gt.shape[0]
    for batch_id in range(occ_gt.shape[0]):

        out_folder = os.path.join(out_folder, subj_name[batch_id], 'images')
        print(f'Slices prediction saved in {out_folder}')
        os.makedirs(out_folder, exist_ok=True)
        im_gt_array = occ_gt[batch_id]
        im_pred_array = occ_pred[batch_id]
        
        #import ipdb; ipdb.set_trace()
        # im_diff = np.zeros(*im_gt_array.shape,3)
        np.save(out_folder, im_pred_array)
        for i in tqdm.tqdm(range(im_gt_array.shape[2])):
            # Show 3 subplots, the gt, the pred and the difference
            gt = im_gt_array[:,:,i]
            pred = im_pred_array[:,:,i]
            
            if(return_image):
                batch_imgs[batch_id].append({'gt': gt, 'pred': pred})

            #sp = compute_dice(torch.tensor(pred),torch.tensor(gt))
            #slice_pred.append(sp)
            # plt.subplot(1, 3, 1)
            # plt.imshow(utils.figures.stack_colored(im_zero, gt, colors))
            # plt.subplot(1, 3, 2)
            # plt.imshow(utils.figures.stack_colored(pred, im_zero, colors))
            # plt.subplot(1, 3, 3)
            # im_diff = utils.figures.stack_colored(pred, gt, colors)
            # plt.imshow(im_diff)
            # # plt.show()
            # plt.savefig(f'{out_folder}/slice_{i:05d}.png')
            # plt.close()
            
            im1 = utils.figures.stack_colored(im_zero, gt, colors)
            im2 = utils.figures.stack_colored(pred, im_zero, colors)
            im3 = utils.figures.stack_colored(pred, gt, colors)
            
            im = np.concatenate([im1, im2, im3], axis=1)
            imsave(f'{out_folder}/slice_{i:05d}.png', (im*255).astype(np.uint8))

            #import ipdb; ipdb.set_trace()
        # import ipdb; ipdb.set_trace()
        # with open(out_folder+'/slice_dice.txt', 'w') as f:
        #         for item in slice_pred:
        #             # write each item on a new line
        #             f.write("%s\n" % str(item))
        make_video(out_folder, out_folder+'.mkv', frame_rate=2, img_string = 'slice_%05d.png')
        make_gif(out_folder, out_folder+'.gif', frame_rate=2, img_string = 'slice_*.png')
    if(return_image):
        return batch_imgs
   
            # plt.imshow(im_gt_array[:,:,i] - im_pred_array[:,:,i])
            # plt.show()


def compute_composition(pred_occ, gt_occ, body_mask, mri_shape, level=0.5):
    """
    Compute the percentage error for the tissue
    """

    pred_occ = (pred_occ >= level).long()
    gt_occ = (gt_occ >= level).long()

    pred_percent = torch.sum(pred_occ, -1) / torch.sum(body_mask, -1)
    gt_percent = torch.sum(gt_occ, -1) / torch.sum(body_mask, -1)
    # import ipdb; ipdb.set_trace()
    print(f'gt: {gt_percent.item()*100}% pred: {pred_percent.item()*100}%')

    # diff = torch.abs(pred_percent - gt_percent)
    # return  torch.mean(diff)
    return  pred_percent - gt_percent

def compute_accuracy(pred_label, gt_label):
    nb_points = gt_label.shape[1] * gt_label.shape[0]
    accuracy = (torch.sum(pred_label == gt_label) / nb_points)
    return accuracy

def compute_loss_per_part(batch, smpl_body, pred_occ, weight=None, **kwargs):
    # import ipdb; ipdb.set_trace()
    # from utils.smpl_utils import get_skinning_weights_batch
    points = batch['can_points']
    part_id = batch['part_id']
    batch_size, n_pts, _ = points.shape
    # TODO: batch skinning weights
    smpl_output = smpl_body(**batch, return_verts=True, return_full_pose=True)
    
    from utils.smpl_utils import new_part_dict_kin_tree as new_part_dict

    part_loss = dict.fromkeys(list(new_part_dict.keys()), 0.)
    # TODO: a way to batch ?
    # TODO: is the argmax thing correct
    import torch.nn.functional as F
    for b in range(batch_size):
        for part in new_part_dict:
            for idx in new_part_dict[part]:
                part_idx = (part_id[b] == idx).nonzero()
                if('outside_can_mask' in kwargs):
                    # Buggy ?
                    part_loss[part] += F.cross_entropy(
                                    input = pred_occ[b][kwargs['outside_can_mask'][b]][part_idx.squeeze(-1)].reshape(-1, pred_occ.shape[-1]),  # (T, 4)
                                    target = batch['mri_occ'][b][part_idx].long().reshape(-1),  # (T)
                                    weight = weight)
                else:
                    part_loss[part] += F.cross_entropy(
                                        input = pred_occ[b][part_idx].reshape(-1, pred_occ.shape[-1]),  # (T, 4)
                                        target = batch['mri_occ'][b][part_idx].long().reshape(-1),  # (T)
                                        weight = weight)
    return part_loss
   
# TODO: arg defaults and order is misleading
def compute_loss_per_part_p2p(q1, q2=None, loss_type:str=None, part_ids=None):
    # import ipdb; ipdb.set_trace()
    from utils.smpl_utils import new_part_dict_kin_tree as new_part_dict
    loss_fct_comp = torch.nn.MSELoss()
    import torch.nn.functional as F

    part_loss = dict.fromkeys(list(new_part_dict.keys()), 0.)
    for part in new_part_dict:
        for idx in new_part_dict[part]:
            part_idx = (part_ids == idx).nonzero()
            if(loss_type == 'compression'):
                part_loss[part] += loss_fct_comp(q1[:,part_idx,:], q2[:,part_idx,:])
            elif(loss_type == 'canonical'):    
                part_loss[part] += F.mse_loss(q1[:,part_idx,:], q2[:,part_idx,:])
            elif(loss_type == 'zero_disp'):    
                part_loss[part] +=  torch.linalg.norm(q1[:,part_idx,:])
            else:
                print('Unrecognized loss type!')
                return
    return part_loss  
    
def compute_part_dice(data_cfg, pred_label, gt_label, part_id, body_mask):
    dict_part_name = {0: 'global',
 1: 'head',
 2: 'leftCalf',
 3: 'leftFingers',
 4: 'leftFoot',
 5: 'leftForeArm',
 6: 'leftHand',
 7: 'leftShoulder',
 8: 'leftThigh',
 9: 'leftToes',
 10: 'leftUpperArm',
 11: 'neck',
 12: 'rightCalf',
 13: 'rightFingers',
 14: 'rightFoot',
 15: 'rightForeArm',
 16: 'rightHand',
 17: 'rightShoulder',
 18: 'rightThigh',
 19: 'rightToes',
 20: 'rightUpperArm',
 21: 'spine',
 22: 'spine1',
 23: 'spine2'}

    part_indices = list(range(24))
    
    # B = pred_label.shape[0]
    data_shape = pred_label.shape
    
    # import ipdb; ipdb.set_trace()
    # body_mask[part_id==1]
    # pred_label = pred_label.reshape(-1)
    # gt_label = gt_label.reshape(-1)
    # part_id = part_id.reshape(-1)
    # body_mask = body_mask.reshape(-1)
    
    part_dict = {}
    for part in part_indices:
        
        part_name = dict_part_name[part]
        
        
        # part_points = torch.nonzero((part_id==part).long(),as_tuple=True)[0] # Query points that are in the part # 1462
        part_points_mask = (part_id==part)
        # body_mask[part_points_indices[0]][part_points_indices[1]]
        # part_points = torch.cat([part_points_indices[0].unsqueeze(-1), part_points_indices[1].unsqueeze(-1)],dim=-1)
        
        nb_part_points = torch.count_nonzero(part_points_mask)
        # print(f'Number of points in part {part_name}: {nb_part_points}')
        
        # if nb_part_points == 0:
        #     print(f'WARNING: No points in part {part}')
        #     for li, mri_label in enumerate(data_cfg['mri_labels']):
        #         part_dict[f"dice_part_{part_name}_{mri_label}"] = torch.tensor([0.])
        #     continue
        
        # import ipdb; ipdb.set_trace()
        pred_part = pred_label[part_points_mask]
        gt_part = gt_label[part_points_mask]
        body_mask_part = body_mask[part_points_mask]
        
        # import ipdb; ipdb.set_trace()
        for li, mri_label in enumerate(data_cfg['mri_labels']):
            part_name = dict_part_name[part]

            tissue_pred_batch = torch.zeros(data_shape, dtype=torch.bool, device=pred_label.device)
            tissue_pred_batch[part_points_mask] = (pred_part==li)
            
            tissue_gt_batch = torch.zeros(data_shape, dtype=torch.bool, device=gt_label.device)
            tissue_gt_batch[part_points_mask] = (gt_part==li)
            
            if mri_label == 'NO':

                tissue_pred_batch = torch.logical_and(tissue_pred_batch, body_mask)
                tissue_gt_batch = torch.logical_and(tissue_gt_batch, body_mask)      
            
            dice = compute_dice(tissue_pred_batch, tissue_gt_batch)
            part_dict[f"dice_part_{part_name}_{mri_label}"] = dice
            # print(f'Part {part_name} {mri_label} DICE: {dice.item()}')
                    
    return part_dict


def validation_eval(cfg, pred_label, gt_label, part_id, body_mask):
    val_loss_dict = {}
    #import ipdb; ipdb.set_trace()
    # todo restore
    # part_dice_dict = compute_part_dice(data_cfg, pred_label, gt_label, part_id, body_mask)
    # val_loss_dict.update(part_dice_dict)
    for li, mri_label in enumerate(cfg.train_cfg['mri_labels']):
        tissue_pred = pred_label==li
        tissue_gt = gt_label==li
        
        if mri_label == 'NO':    
            tissue_pred = torch.logical_and(tissue_pred, body_mask)
            tissue_gt = torch.logical_and(tissue_gt, body_mask)        
        
        val_loss_dict[f"dice_{mri_label}"] = compute_dice(tissue_pred, tissue_gt)
        # Add per part per tissue
        from utils.smpl_utils import new_part_dict_kin_tree
        for body_part in new_part_dict_kin_tree:
            # TODO: no possibility for batching ?
            val_loss_dict[f"dice_{mri_label}_{body_part}_part"] = 0
            for b in range(tissue_pred.shape[0]):
                val_loss_dict[f"dice_{mri_label}_{body_part}_part"] += compute_dice(tissue_pred[b,tissue_gt[b].nonzero()], 
                                                                                    tissue_gt[b,tissue_gt[b].nonzero()])
    
    val_loss_dict["accuracy"] = compute_accuracy(pred_label, gt_label)
    return val_loss_dict

