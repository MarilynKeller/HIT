import os
import pickle
import random
import numpy as np
import trimesh
from psbody.mesh import Mesh

import hit.hit_config as cg


def weights2colors(weights):
    """Convert a matrix of skinning weights to a matrix of colors"""
    colors = color_gradient(weights.shape[1], scale=1, shuffle=True, darken=False, pastel=False, alpha=None, seed=2)
    verts_colors = weights[:,:,None] * colors

    verts_colors = verts_colors.sum(1)

    return verts_colors


def color_gradient(N, scale=1, shuffle=False, darken=False, pastel=False, alpha=None, seed=0):

    """Return a Nx3 or Nx4 array of colors forming a gradient

    :param N: Number of colors to generate
    :param scale: Scale of the gradient
    :param shuffle: Shuffle the colors
    :param darken: Darken the colors
    :param pastel: Make the colors pastel
    :param alpha: Add an alpha channel of value alpha to the colors
    """
    import colorsys
    if darken:
        V = 0.75
    else:
        V = 1

    if pastel:
        S = 0.5
    else:
        S = 1

    HSV_tuples = [((N-x) * 1.0 / (scale*N), S, V) for x in range(N)] # blue - grean - yellow - red gradient
    RGB_list = list(map(lambda x: list(colorsys.hsv_to_rgb(*x)), HSV_tuples))

    if shuffle:
        random.Random(seed).shuffle(RGB_list) #Seeding the random this way always produces the same shuffle
    
    RGB_array = np.array(RGB_list)
    
    if alpha is not None:
        RGB_array = np.concatenate([RGB_array, alpha * np.ones((N, 1))], axis=1)
    
    return RGB_array


import torch

# Add v2p as global variable
dict_part_name = {0: 'global',1: 'head',2: 'leftCalf',3: 'leftFingers',4: 'leftFoot',5: 'leftForeArm',
                  6: 'leftHand',7: 'leftShoulder',8: 'leftThigh',9: 'leftToes',10: 'leftUpperArm',
                  11: 'neck',12: 'rightCalf',13: 'rightFingers',14: 'rightFoot',15: 'rightForeArm',
                  16: 'rightHand',17: 'rightShoulder',18: 'rightThigh',19: 'rightToes',20: 'rightUpperArm',
                  21: 'spine',22: 'spine1',23: 'spine2'}
dict_part_name_kin_tree = {
                  0: 'global', 1: 'l_hip',2: 'r_hip',3: 'spine_1',4: 'r_knee',5: 'r_knee',
                  6: 'spine_2',7: 'l_ankle',8: 'r_ankle',9: 'spine_3',10: 'l_toe',
                  11: 'r_toe',12: 'neck',13: 'l_collar',14: 'r_collar',15: 'head',
                  16: 'l_shoulder',17: 'r_shoulder',18: 'l_elbow',19: 'r_elbow',20: 'l_wrist',
                  21: 'r_wrist',22: 'l_palm',23: 'r_palm'}
new_part_dict = {
                "legs": [2,4,9,12,14],
                "lower body": [8,18,21],
                "upper_body": [17,22,23],
                "hands and arms": [3,5,6,7,10,13,16,16],
                "head": [1,11],}

new_part_dict_kin_tree = {
        "legs": [4,5,7,8,10,11],
        "lower body": [0,1,2,3],
        "upper_body": [6,9,13,14,16,17],
        "hands and arms": [18,19,20,21,22,23],
        "head": [12,15],}


with open(cg.v2p, 'rb') as f:
    data = pickle.load(f)
    data = [list(dict_part_name.keys())[list(dict_part_name.values()).index(i)] for i in data]
    SMPL_V2P = torch.tensor(data, device='cuda:0')
    # TODO: change device

def canonical_x_bodypose():
    """Generate a canonical body pose with legs slightly open

    Returns:
        torch tensor [1,69]: body pose
    """
    body_pose = torch.zeros((1, 69),dtype=torch.float32)
    body_pose[:,2] =  torch.pi / 6 #Root is a separated parameter, so legs angles are at indices 0*3+2 and 1*3+2
    body_pose[:,5] =  -torch.pi / 6
    
    return body_pose


def get_gdna_bone_transfo(smpl, smpl_output):
    """Given the smpl obj used in gDNA, compute the bone transformation matrix for each bone as they did in gDNA

    Args:
        smpl (_type_): smpl obj used in gDNA
        smpl_output (_type_): smpl_output from my code

    Returns:
        _type_: tf_mats
    """

    pose_can_x = canonical_x_bodypose().to(smpl_output.body_pose.device).expand(smpl_output.body_pose.shape[0], -1)
    pose_gdna = smpl_output.body_pose #- pose_can_x
    smpl_output2 = smpl.forward(betas=smpl_output.betas,
                                    transl=smpl_output.transl,
                                    body_pose=pose_gdna,
                                    global_orient=smpl_output.global_orient,
                                    return_verts=True,
                                    return_full_pose=True)
    
    tf_mats = smpl_output2.T.clone()
    
    # import ipdb; ipdb.set_trace()
    cano_output = smpl.forward(betas=smpl_output.betas,
                                transl=None,
                                body_pose=pose_can_x,
                                global_orient=torch.zeros_like(smpl_output.global_orient),
                                return_verts=True,
                                return_full_pose=True)
    
    #    output_cano = { k+'_cano': v for k, v in output_cano.items() }
    # output.update(output_cano)

    tfc_cano = cano_output.T.clone()
    tfs_c_inv = tfc_cano.inverse()
    tf_mats = torch.einsum('bnij,bnjk->bnik', tf_mats, tfs_c_inv)
            
    # tf_mats[:, :, :3, 3] += smpl_output2.transl.unsqueeze(1)
    # smpl_tfs = tf_mats
    # smpl_tfs = bone_trans
    return tf_mats


def get_skinning_weights(points, smpl_vertices, smpl, free_verts=None):
    """ 
    Given points in world space, compute the skinning weights for each point
    points : [T, 3] numpy array
    free_verts : [6890, 3] tensor
    return : skinning weightd[T, 24] tensor, part_id [T] tensor
    
    """
    
    # is_batch_data = (len(points.shape) == 3)
    
    assert len(points.shape) == 2, 'points should be [T, 3]'
    assert isinstance(points, np.ndarray), 'points should be numpy array'
    if free_verts is not None:
        assert isinstance(free_verts, np.ndarray), 'free_verts should be numpy array'
    assert  isinstance(smpl_vertices, np.ndarray)

    # For each smpl vertex, gives the corresponding part
    with open(cg.v2p, 'rb') as f:
        data = pickle.load(f)
    part_list = sorted(set(data))
    dict_part={}
    for idx,ele in enumerate(part_list):
        dict_part[ele] = idx 
    part_num = np.array([ dict_part[v] for v in data])
    
    body_verts = smpl_vertices
    if free_verts is not None:
        body_verts = free_verts
        
    # import ipdb; ipdb.set_trace()
    mesh = trimesh.Trimesh(body_verts, smpl.faces, process=False)
    prox_query = trimesh.proximity.ProximityQuery(mesh)
    closest_vertex, closest_vertex_index = prox_query.vertex(points)
    skinning_weights = smpl.lbs_weights[closest_vertex_index].cpu().detach().numpy()
    part_id = np.argmax(skinning_weights, axis=1)
    # return skinning_weights, part_id

    # Find vertices rigged to the left thigh (Joint 1)
    lt_vert_pt = body_verts[974] # vertex on the external side of the left thigh
    rt_vert_pt = body_verts[4460] # vertex on the external side of the right thigh
    
    
    # List all the vertices that are associated with the wrong part
    # A vertex is wrongly associated with the left thigh if it is closer to the right thigh side than to the left thigh side
    mask_wrong_l = (part_id == 1) & (np.linalg.norm(points - lt_vert_pt, axis=-1) > np.linalg.norm(points - rt_vert_pt, axis=-1))
    mask_wrong_r = (part_id == 2) & (np.linalg.norm(points - lt_vert_pt, axis=-1) < np.linalg.norm(points - rt_vert_pt, axis=-1))

    debug=False
    if debug:
        from psbody.mesh import MeshViewer, MeshViewers
        from psbody.mesh.sphere import Sphere
        mv = MeshViewer()
        tigh_mask = (part_id==1).numpy() | (part_id==2)
        mv.set_static_meshes([Mesh(v=body_verts),                             
                            # Mesh(v=points[tigh_mask], f=[], vc=weights2colors(skinning_weights[tigh_mask])),
                            Mesh(v=points, f=[], vc=weights2colors(skinning_weights)),
                            # Mesh(v = points[(part_id == 1)], f=[], vc=np.array([0,1,0])),
                            # Mesh(v = points[(np.linalg.norm(points - lt_vert_pt, axis=-1) > np.linalg.norm(points - rt_vert_pt, axis=-1))], f=[], vc=np.array([1,0,0])), 
                            # Mesh(v = points[mask_wrong_l], f=[], vc=np.array([0,1,0])), 
                            Sphere(lt_vert_pt, 0.02).to_mesh(np.array([0,1,1])), 
                            # Sphere(rt_vert_pt, 0.02).to_mesh(np.array([0,1,1]))
                            ])
        import ipdb; ipdb.set_trace()
    
    # Correct the vertices wrongly associated with the left thigh
    # Inverse right and left thigh weights
    skinning_weights[mask_wrong_l, 2] = skinning_weights[mask_wrong_l, 1]
    skinning_weights[mask_wrong_l, 4] = skinning_weights[mask_wrong_l, 3]
    # Set the wrong limb weight to 0
    skinning_weights[mask_wrong_l, 1] = 0
    skinning_weights[mask_wrong_l, 3] = 0
    # Set the part id to the right part
    part_id[mask_wrong_l] = 2
    
    # Correct the vertices wrongly associated with the right thigh
    skinning_weights[mask_wrong_r, 1] = skinning_weights[mask_wrong_r, 2]
    skinning_weights[mask_wrong_r, 3] = skinning_weights[mask_wrong_r, 4]
    skinning_weights[mask_wrong_r, 2] = 0
    skinning_weights[mask_wrong_r, 4] = 0
    part_id[mask_wrong_r] = 1
    
    if debug:
        print(f"Corrected {mask_wrong_l.sum()} vertices wrongly associated with the left thigh")
        print(f"Corrected {mask_wrong_r.sum()} vertices wrongly associated with the right thigh")

    if debug:
        tigh_mask = (part_id==1).numpy() | (part_id==2).numpy()
        Mesh(v=points[tigh_mask], f=[], vc=weights2colors(skinning_weights[tigh_mask]).numpy()).show()
        
    return skinning_weights, part_id


def x_pose_like(body_pose):
    x_pose = torch.zeros_like(body_pose) 
    x_pose[:,2] =  torch.pi / 6 # Root is a separated parameter, so legs angles are at indices 0*3+2 and 1*3+2
    x_pose[:,5] =  -torch.pi / 6
    return x_pose

def get_template_verts(batch, smpl_body):
    global_orient0 = torch.zeros_like(batch['global_orient'])
    body_pose0 = x_pose_like(batch['body_pose'])
    betas0 = torch.zeros_like(batch['betas'])
    transl= torch.zeros_like(batch['transl'])
    smpl_output = smpl_body(global_orient=global_orient0, body_pose=body_pose0, betas=betas0, transl=transl, return_verts=True, return_full_pose=True)
    
    # smpl_output = smpl_body.forward_canonical(torch.zeros_like(batch['betas']))
    smpl_template_verts = smpl_output.vertices#[:,::10,:]
    return smpl_template_verts

def get_shaped_verts(batch, smpl_body):
    global_orient = torch.zeros_like(batch['global_orient'])
    body_pose = x_pose_like(batch['body_pose'])
    betas = batch['betas']
    transl= torch.zeros_like(batch['transl'])
    smpl_output = smpl_body(global_orient=global_orient, body_pose=body_pose, betas=betas, transl=transl, return_verts=True, return_full_pose=True)
    smpl_verts = smpl_output.vertices#[:,::10,:]
    
    # smpl_output = smpl_body.forward_canonical(batch['betas'])
    # smpl_verts = smpl_output.vertices#[:,::10,:]
    
    return smpl_verts

def psm(verts, smpl, values=None, indices=False, maxval=None, norm=False, display=True, skin_weights=None):
    '''Plot a smpl mesh with values as colors
    verts: [B,6890,3]
    smpl: mysmpl obj
    values [B,6890] or [B,6890,3] . The mesh will be colored according to these values
    indices: if True, the mesh will be colored according to the vertex indices. This enables to see the mesh topology.
    maxval: if values is an array [B,6890,3] of 3d offsets, the values will be normalized by maxval. If None, maxval is set to the max of the absolute values of values
    norm: if True, and values is an array [B,6890,3] of 3d offsets, the norm of the 3d displacements will be used to color the mesh
    '''
    
    mesh = Mesh(v=verts[0].detach().cpu().numpy(), f=smpl.faces)

    if indices is True:
        mesh.set_vertex_colors_from_weights(np.arange(mesh.v.shape[0]))
    elif values is not None:
        values = values[0].detach().cpu().numpy()
        if len(values.shape) == 1: 
            mesh.set_vertex_colors_from_weights(values)
            mesh.vc = mesh.vc/(np.max(mesh.vc)+1e-7)
        elif len(values.shape) == 2:
            if maxval is None:
                maxval = np.abs(values).max()
            if norm:
                values = np.linalg.norm(values, axis=1)
                mesh.set_vertex_colors_from_weights(values)
            else:
                mesh.vc = np.abs(values)/(maxval+1e-7)
        else:
            raise Exception(f'Unknown values shape, should be [6890] or [6890,3], got {values.shape}')
    elif skin_weights is not None:
        mesh.vc = weights2colors(skin_weights[0].detach().cpu().numpy())
        mesh.vc = mesh.vc/(np.max(mesh.vc)+1e-7)
    
    if display :      
        mesh.show()
    
    return mesh
    
    
def mpimesh2glb(mesh, path=None):
    # check that mesh has an attribut vc
    if hasattr(mesh, 'vc'):
        trimesh_mesh = trimesh.Trimesh(mesh.v, mesh.f, vertex_colors=mesh.vc, process=False)
    else:
        trimesh_mesh = trimesh.Trimesh(mesh.v, mesh.f, process=False)
    
    if path is None:
        folder = '/tmp/hit_training_meshes'
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f'mesh_{random.randint(0,100)}.glb')
    trimesh_mesh.export(path)
    return path
    
    
    
def t2m(mesh, show=True):
    '''Convert a trimesh mesh to a mpi mesh'''
    v = mesh.vertices
    f = mesh.faces
    m = Mesh(v=v, f=f)
    if show:
        m.show()
    return m

