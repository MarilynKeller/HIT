       
import torch


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
 
        