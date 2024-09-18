       
import torch
import torch.nn.functional as F

def occ2sdf(scores, channel):
    """ Convert per channel occupancy scores to SDF
    Args:
        occ : B, T, C # Occupancy scores for the B batches, T points and C channels
        channel : int # Channel to extract the SDF for (must be in [0, C-1])
    Returns:
        sdf_value : B, T # a float value sucht that sdf[x, channel]=0.5 when (occ[x, channel]==occ[x, ci] & channel = argmax(occ))
    """
    occ_all_tissue = F.softmax(scores, dim=-1) 

    sorted_idx = torch.argsort(occ_all_tissue, dim=-1, descending=True)
                
    best_idx = sorted_idx[:,:,0] 
    best_non_channel_idx = best_idx
    best_non_channel_idx[best_idx==channel] = sorted_idx[:,:,1][best_idx==channel]
    
    channel_score = occ_all_tissue[..., channel]
    best_non_channel_score = occ_all_tissue.gather(-1, best_non_channel_idx.unsqueeze(-1)).squeeze(-1)
    
    assert (best_non_channel_idx==channel).sum() == 0
    sdf_tissue = channel_score / (channel_score + best_non_channel_score)
    sdf = sdf_tissue.cpu().squeeze(0)
    return sdf


def occ_to_percentage(occ_array_inside, labels, is_inside=None, mask_with_inside=True):

    if is_inside is None:
        # import ipdb; ipdb.set_trace()
        is_inside = torch.ones_like(occ_array_inside)

    body_volume = is_inside.sum()
        
     # Compute the percentage of each tissue
    tissue_percentage = []
    for ti, t_label in enumerate(labels):
        is_tissue = ( occ_array_inside == ti)
        if is_inside is not None and mask_with_inside:
            tissue_volume = (is_tissue * is_inside).sum()
        else:
            tissue_volume = is_tissue.sum()
        ratio = tissue_volume / body_volume
        print(f"{t_label} : {ratio*100:.2f} %")   
        tissue_percentage.append(ratio)
    return tissue_percentage
    
    
    
def mri_data_to_percentage(data, labels, mask_with_inside=True):
    """ Given an mri subject data, returns the percentage of each tissue.
    """   
    print("MRI GT tissue percentage:")  
    tissue_percentage = occ_to_percentage(data['mri_seg'], labels, is_inside=data['mri_seg_BODY'],
                                          mask_with_inside=mask_with_inside)
    return tissue_percentage
 
        