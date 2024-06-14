import math
import pickle

import numpy as np
import smplx
import torch
from utils.pyrender_renderer import render_mesh_from_trimesh
from skimage.io import imsave

white = np.array([255, 255, 255])/255
gray = np.array([128, 128, 128])/255
black = np.array([0, 0, 0])/255
col_lt =  np.array((180, 17, 17))/255
col_at = np.array((245, 220, 90))/255
col_vat = np.array((140, 200, 220))/255
col_no = np.array((80, 80, 80))/255
col_bn = np.array((120, 120, 120))/255


# self.fov = math.pi/4
# self.im_shape = (150, 512)  

# T_back = np.array([0, 0, 1])  

# self.cam_pose = np.array([
#     [1.0,  0, 0, 0.25],
    
#     [0.0, 1.0,  0.0, 0.05],
    
#     [0.0,  0,    1,  3],
    
#     [0.0,  0.0,   0.0, 1.0]
# ])

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

R_back = rotation_matrix(np.array([0,1,0]), theta=math.pi)


def get_smpl_rot(pose):
    pose = pose[:3].clone().float().unsqueeze(0)
    R = smplx.lbs.batch_rodrigues(pose.reshape(-1, 3)) # B,3,3
    return R.squeeze(0).cpu().numpy()

def snapshot_mesh(mesh, mesh_path, R_cam, cam_pose, im_shape, fov):
    ''' Render the mesh and save the image with a path matching the mesh path'''
    mesh.vertices = np.dot(mesh.vertices , R_cam) 
    im = render_mesh_from_trimesh(mesh, cam_pose=cam_pose, im_shape=im_shape, fov=fov )
    save_path = mesh_path.replace('.obj', '.png')
    save_path = save_path.replace('meshes', 'images')
    imsave(save_path, im)
    print(f'Image saved as {save_path}')
          
          