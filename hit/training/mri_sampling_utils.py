import importlib

import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter

def metric_2_mri(mri_data, query_points):
    '''THis function works weather the data have a batch dimension or not. 
    Comments are for the cases where there is no batch dimension.
    If there is a batch dimention, consider a (B,...) added in front of each dim'''
    center = mri_data['mri_center'].cpu().numpy()  # (N_slices, 3)
    resolution = mri_data['mri_resolution'].cpu().numpy() # (N_slices, 2)
    pi = query_points.copy() # (T, 3)
    
    # the xy mri resolution depends on the mri slice, so first we compute in which slice the point is projected
    z_meters = pi[...,2] # (T) z coordinate in metric
    z_res = resolution[...,2].mean() # resolution (N_slices, 2) has the same resolution for each slice, so we take the mean
    # if len(z_meters.shape) == 2:
    #     z_res = z_res.reshape((1, -1))
    z_mri = (np.round(z_meters / z_res)).astype(int)  # (T) z coordinate in mri space   
    
    # clip values to be in the range of the mri
    z_mri = z_mri.clip(0, resolution.shape[0]-1) # (T) clip to the mri volume
    # import ipdb; ipdb.set_trace()
    if len(z_mri.shape) == 2:
        z_mri = z_mri.reshape((-1))
    res_slice = resolution[..., z_mri[:], :] # resolution for each slice (T, 3)
    center_slice = center[..., z_mri[:], :] # center for each slice ( T, 3)

    # pi = np.matmul(pi, np.diag(1/np.array(resolution)))
    pi = 1/res_slice * pi
    pi = pi + center_slice #todo make that clean

    return pi


def mri_2_metric(mri_data, query_points):
    center = mri_data['mri_center'].cpu().numpy()
    resolution = mri_data['mri_resolution'].cpu().numpy()
    pi = query_points.copy().astype(np.float32)

    slice_index = pi[...,2].astype(int)
    slice_index_clipped = slice_index.clip(0, resolution.shape[0]-1) # clip to the mri volume
    
    # On top and bottom of the MRI, we extend the center xy to the nearest slice
    # center z remains the slice index
    slice_center = center[slice_index_clipped]
    
    # For points on top and bottom of the MRI, we use the resolution of the nearest slice
    slice_res = resolution[slice_index_clipped]
    # pi = np.matmul(np.diag(np.array(resolution)), pi)
    pi = pi - slice_center #todo make that clean
    pi = pi * slice_res 

    return pi

   

def unposed_2_posed(mri_data, query_points, root_rot=False, b_ind=None):
    """Apply the root translation. If root_rot is True, also apply the root rotation"""
    
    root_trans = mri_data['root_trans'].cpu().numpy()
    pi = query_points.copy()
    

    
    #translate to center on smpl joint 0
    if root_rot:
        inv_root_rot = mri_data['inv_root_rot'].cpu().numpy()
        hips_center =  mri_data['root_joint'].cpu().numpy()
        
        pi = pi - hips_center
        pi = np.matmul(pi, inv_root_rot) #rotate
        pi = pi + hips_center
    
    if(b_ind is not None):
        pi = pi + root_trans[b_ind]
    else:
        pi = pi + root_trans

    return pi

def posed_2_unposed(mri_data, query_points, root_rot=False):
    """Undo the root translation. If root_rot is True, also undo the root rotation"""
    root_trans = mri_data['root_trans'].cpu().numpy()

    pi = query_points.copy()

    pi = pi - root_trans
    
    if root_rot:
        hips_center =  mri_data['root_joint']
        inv_root_rot = mri_data['inv_root_rot'] 
        pi = pi - hips_center
        pi = np.matmul(pi, inv_root_rot.T)
        pi = pi + hips_center

    return pi


def sample_grid(array, coords, interp_order):
    """
    Sample float coordinates in 3d array arr, by rounding values
    """
    
    # coords = coords.cpu().numpy()
    if interp_order == 0:
        mri_indices = np.round(coords).astype(int)

        x_index = mri_indices[..., 0]
        y_index = mri_indices[..., 1]
        z_index = mri_indices[..., 2]

        valid_mask_x = np.logical_and(x_index < array.shape[0], x_index > 0)
        valid_mask_y = np.logical_and(y_index < array.shape[1], y_index > 0)
        valid_mask_z = np.logical_and(z_index < array.shape[2], z_index > 0)
        valid_mask = np.logical_and(valid_mask_x, valid_mask_y)
        valid_mask = np.logical_and(valid_mask, valid_mask_z)

        # mri_seg.gather(0, mri_indices)

        # import ipdb; ipdb.set_trace()
        val_shape = [coords.shape[0]] + [s for s in  array.shape[3:]]# Direct conversion to list seems to fail
        val = np.zeros(val_shape, dtype=np.float32)
        val[valid_mask] = array[x_index[valid_mask], y_index[valid_mask], z_index[valid_mask], ...]

    elif interp_order > 0 :
        from scipy import ndimage
        array = array.astype(np.float32)
        val = ndimage.map_coordinates(array, coords.T, order=interp_order)
#    
        # if np.max(ndimage.map_coordinates(arr, coords.T, order=0)) > 0.0:
        #     occ0 = ndimage.map_coordinates(arr, coords.T, order=0)
        #     occ1 = ndimage.map_coordinates(arr, coords.T, order=1)
        #     occ2 = ndimage.map_coordinates(arr, coords.T, order=2)
        #     print(np.unique(occ0))
        #     print(np.unique(occ1))
        #     np.all(occ1==occ0)
        
        val = (val>0.5).astype(np.float32)
    return val


def query2mri(mri_data, query_points, b_ind=None):
    '''Given 3D points in the SMPL(trans=0) space, ruturn the 3D mri coordinates'''
    pi = query_points.copy()
    pi = unposed_2_posed(mri_data, pi, b_ind=b_ind) # this undo the root translation    
    pi = metric_2_mri(mri_data, pi) # metric space to mri space  
    return pi  


def load_occupancy(mri_data, query_points, interp_order=0, b_ind=None ):
    
    pi = query2mri(mri_data, query_points, b_ind)
    
    seg_arr = mri_data['mri_seg']
    body_mask = mri_data['mri_seg_BODY']
    
    # 
    if isinstance(seg_arr, torch.Tensor) and len(seg_arr.shape)==4:
        seg_arr = seg_arr.cpu().numpy()[0] # Remove batch dimention and convert to numpy
        body_mask = body_mask.cpu().numpy()[0]
        pi = pi[0]
    
    vals = np.unique(seg_arr)
    for val in vals:
        assert val in [0,1,2,3,4]

    body_mask = sample_grid(body_mask, pi, interp_order)
    occ = sample_grid(seg_arr, pi, interp_order)
    return occ, body_mask

def load_sdf_normal(mri_data, query_points, interp_order=0 ):
    
    normals = mri_data['mri_sdf_gradient']
    
    pi = query2mri(mri_data, query_points)
    
    val = sample_grid(normals.cpu().numpy(), pi, interp_order)
    return val


def sample_mri_pts(mri_data, body_only, dilate_body=False, use_mri_net=False):
    # If body_only is True, only sample points from the body and around, not in the whole MRI
    # return pts (N, 3) in metric space and the cooresponding pts coordinates pts_uv (N, 3) in mri space

    if body_only:
        # Dilate body volume to add 3 voxels on the outside
        if dilate_body:
            body = mri_data['mri_seg_BODY'][:,2:-2].clone() #remove the first and last noisy slices
            struct_shape = (4,4,4) 
            # print('structure shape:', struct_shape)
            structure = np.ones(struct_shape).astype(bool)
            body_dilated = ndimage.binary_dilation(body, structure)
            mask = body_dilated
        else:
            # import ipdb; ipdb.set_trace()
            mask = mri_data['mri_seg_BODY'][:,2:-2].clone()
    else:
        mri_size = mri_data['mri_size']
        mask = np.ones(mri_size.cpu().numpy().tolist())
       
    seg = mri_data['mri_seg'][np.where(mask > 0)]
    body_mask = mri_data['mri_seg_BODY'][np.where(mask > 0)]
    pts_uv = np.dstack(np.where(mask > 0))[0]
    pts = pts_uv.copy() 
    pts = mri_2_metric(mri_data, pts)
    pts = posed_2_unposed(mri_data, pts)

    # @varora
    if use_mri_net:
        mri_values = mri_data['mri_values'][np.where(mask > 0)]
    else:
        mri_values = None

    return pts, pts_uv, seg, body_mask, mri_values

def sample_tissue_pts(mri_data, channel_idx=None, mask=None, body_only=True):
    # If body_only is True, only sample points from the body and around, not in the whole MRI
    # mask: bool array of the size of the mri, only sample points in the mask
    # return pts (N, 3) in metric space and the cooresponding pts coordinates pts_uv (N, 3) in mri space
       
    if mask is None:
        assert channel_idx is not None, 'Please specify channel_idx or mask'
        body_mask = mri_data['mri_seg_BODY']
        tissue_arr = mri_data['mri_seg']
        mask = np.logical_and(tissue_arr==channel_idx, body_mask>0)
    
    pts_uv = np.dstack(np.where(mask > 0))[0]
    pts = pts_uv.copy() 
    pts = mri_2_metric(mri_data, pts)
    pts = posed_2_unposed(mri_data, pts)
    return pts


def center_on_voxel(pi, mri_data):
    
    pi = unposed_2_posed(mri_data, pi) # this undo the root translation     
    pi = metric_2_mri(mri_data, pi) # metric space to mri space
    pi = np.rint(pi) # center on voxel
    pi = mri_2_metric(mri_data, pi) # mri space to metric space
    pi = posed_2_unposed(mri_data, pi)
    return pi
    

def vis_create_pc(pts, color=(0.0, 1.0, 0.0), radius=0.005):

    import numpy as np
    import pyrender
    import trimesh
    if torch.is_tensor(pts):
        pts = pts.cpu().numpy()

    tfs = np.tile(np.eye(4), (pts.shape[0], 1, 1))
    tfs[:, :3, 3] = pts
    sm_in = trimesh.creation.uv_sphere(radius=radius)
    sm_in.visual.vertex_colors = color

    return pyrender.Mesh.from_trimesh(sm_in, poses=tfs)

    """
    import pyrender
    scene = pyrender.Scene(ambient_light=[.1, 0.1, 0.1], bg_color=[1.0, 1.0, 1.0])
    scene.add(pyrender.Mesh.from_trimesh(posed_mesh))
    scene.add(vis_create_pc(query_points[occ==1], color=(1., 0., 0.)))  # red - inside points
    scene.add(vis_create_pc(query_points[occ==0], color=(0., 1., 0.)))  # blue - outside points
    pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=False)
    """
    
def compute_discrete_sdf_gradient(mri_seg, mri_res, display=True):
    
    per_label_gradient = []
    for label_i in np.unique(mri_seg):
        
        mask = (mri_seg == label_i)
        # import ipdb; ipdb.set_trace()
        struct_shape = (2,2,2) 
        structure = np.ones(struct_shape).astype(bool)
        mask = ndimage.binary_opening(mask, structure)
        mask = ndimage.binary_closing(mask, structure)
       
        if display:
            pn = 5
            slice = mask.shape[0]//2 - 20
            slice_x = slice
            slice_y = mask.shape[1]//2 
            slice_z = mask.shape[2]//2 
            import matplotlib.pyplot as plt
            plt.figure()
            plt.subplot(1,pn,1)
            plt.imshow(mask[ slice,:, :])

        import scipy
        mask_dst_inside = scipy.ndimage.distance_transform_edt(mask, sampling=[1,1,10])
        mask_dst_outside = scipy.ndimage.distance_transform_edt(np.logical_not(mask), sampling=[1,1,10])
        
        # mask_dst_inside = scipy.ndimage.distance_transform_cdt(mask, metric='cityblock')
        # mask_dst_outside = scipy.ndimage.distance_transform_cdt(np.logical_not(mask), metric='cityblock')

        # mask_dst_inside = scipy.ndimage.distance_transform_bf(mask, sampling=[1,1,10])
        # mask_dst_outside = scipy.ndimage.distance_transform_bf(np.logical_not(mask), sampling=[1,1,10])
                
        # mask_dst = mask.astype(np.float32)
        
        mask_dst = mask_dst_inside - mask_dst_outside

        mask_dst = gaussian_filter(mask_dst, sigma=[3,3,1])

        gradient = np.gradient(mask_dst, 1,1,10) # list of gradients along each axis (x,y,z)
        
        if display:
            import matplotlib as mpl
            name = 'PiYG'
            plt.subplot(1,pn,2)
            plt.imshow(mask_dst[slice,:, :], cmap=mpl.colormaps[name])
            
            # Plot gradient
            # Slice x 
            plt.subplot(1,pn,3)
            plt.imshow(gradient[0][:, slice_y, :], cmap=mpl.colormaps[name])
            
            #slice y
            plt.subplot(1,pn,4)
            plt.imshow(gradient[1][slice_x , :, :], cmap=mpl.colormaps[name])
            
            #slice z
            plt.subplot(1,pn,5)
            plt.imshow(gradient[2][slice_x,:, :], cmap=mpl.colormaps[name])
        
        # Stack the gradients along x, y z in the last axis
        gradient = np.concatenate([g[...,None] for g in gradient], axis=-1) #  W, D, H, 3
        # gradient = gradient / (np.linalg.norm(gradient, axis=-1, keepdims=True)+0.001)
        
        per_label_gradient.append(gradient[None])
        
        #write figure to file
        import matplotlib.pyplot as plt
        plt.draw()
        plt.savefig(f'gradient_channel_{label_i}.png')
        
        # if display:
        #     plt.show() 
            
            
    gradient = np.concatenate(per_label_gradient, axis=0) # C, W, D, H, 3
    # import ipdb;ipdb.set_trace()
    # Move dimensions to one before last axis
    gradient = np.moveaxis(gradient, 0, -2) # W, D, H, C, 3
    # Normalize
    # gradient = gradient / np.linalg.norm(gradient, axis=-1, keepdims=True) # 
    return gradient
        
    
