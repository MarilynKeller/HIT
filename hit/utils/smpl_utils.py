import os
import pickle
import random
import time

import hit_config as cg
import numba
import numpy as np
import trimesh
from psbody.mesh import Mesh


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
    
    v = smpl_vertices
    if free_verts is not None:
        v = free_verts
        
    # import ipdb; ipdb.set_trace()
    mesh = trimesh.Trimesh(v, smpl.faces, process=False)
    prox_query = trimesh.proximity.ProximityQuery(mesh)
    closest_vertex, closest_vertex_index = prox_query.vertex(points)
    skinning_weights = smpl.lbs_weights[closest_vertex_index]
    part_id = torch.tensor(part_num[closest_vertex_index])
    
    # if isinstance(points, np.ndarray):
    #     import ipdb; ipdb.set_trace()
    #     points = points[None]
    #     if free_verts is not None:
    #         free_verts = free_verts[None]
    # else:
    #     points = points.cpu().numpy()
    
    # B = points.shape[0]
    # T = points.shape[1]
    # skinning_weights = torch.zeros((B, T, smplx_body.lbs_weights.shape[1]), dtype=torch.float32)
    # part_id = torch.zeros((B, T), dtype=torch.int32)
    # for bi in range(B):
    #     v = smpl_output.vertices[bi].cpu().numpy()
    #     if free_verts is not None:
    #         v = free_verts[bi].cpu().squeeze().numpy()
            
    #     mesh = trimesh.Trimesh(v, smplx_body.faces, process=False)
    #     prox_query = trimesh.proximity.ProximityQuery(mesh)
    #     closest_vertex, closest_vertex_index = prox_query.vertex(points[bi])
    #     skinning_weights_i = smplx_body.lbs_weights[closest_vertex_index]
    #     part_id_i = torch.tensor(part_num[closest_vertex_index])
        
    #     skinning_weights[bi] = skinning_weights_i
    #     part_id[bi] = part_id_i
        
        
    return skinning_weights, part_id


def get_skinning_weights_normals_based(points, smpl_vertices, smpl, free_verts=None):
    """
    Given points in world space, compute the skinning weights for each point
    points : [T, 3] numpy array
    free_verts : [6890, 3] tensor
    return : skinning weightd[T, 24] tensor, part_id [T] tensor

    """
    print("New sampling.....................................................................")

    # is_batch_data = (len(points.shape) == 3)

    assert len(points.shape) == 2, 'points should be [T, 3]'
    assert isinstance(points, np.ndarray), 'points should be numpy array'
    if free_verts is not None:
        assert isinstance(free_verts, np.ndarray), 'free_verts should be numpy array'
    assert isinstance(smpl_vertices, np.ndarray)
    # For each smpl vertex, gives the corresponding part
    with open('../v2p.pkl', 'rb') as f:
        data = pickle.load(f)
    part_list = sorted(set(data))
    dict_part = {}
    for idx, ele in enumerate(part_list):
        dict_part[ele] = idx
    part_num = np.array([dict_part[v] for v in data])

    v = smpl_vertices
    if free_verts is not None:
        v = free_verts

    # import ipdb; ipdb.set_trace()
    mesh = trimesh.Trimesh(v, smpl.faces, process=False, use_embree=True)

    prox_query = trimesh.proximity.ProximityQuery(mesh)
    _, closest_vertex_index = prox_query.vertex(points)
    part_id = part_num[closest_vertex_index]

    points_thighs_mask = (part_id==8) | (part_id==18)
    points_thighs = points[points_thighs_mask]
    part_id_thighs = part_id[points_thighs_mask]

    points_rest = points[~points_thighs_mask]
    closest_vertex_index_rest = closest_vertex_index[~points_thighs_mask]

    @numba.njit(cache=True, parallel=True)
    def process_batch_numba(points, mesh_vertices, mesh_vertex_normals, planar=False):
        """
        numba friendly implementation rigging points with cloest vertex subject to positive signed distance
        Args:
            points: point cloud points
            mesh_vertices: mesh vertices
            mesh_vertex_normals: mesh normals

        Returns:
            rigged_points
        """
        rigged_points = np.zeros(points.shape[0])
        n_query_points = 10000
        for i in range(0, len(points), n_query_points):
            batch = points[i:i + n_query_points]
            for idx in numba.prange(len(batch)):
                p = batch[idx]
                if planar:
                    #mask_planar = (mesh_vertices[:, 2] < (p[2] + 0.03)) & (mesh_vertices[:, 2] > (p[2] - 0.03))
                    #mask_planar = np.logical_and((mesh_vertices[:, 2] < (p[2] + 0.03)), (mesh_vertices[:, 2] > (p[2] - 0.03)))
                    mask_planar = np.logical_and( ( np.less(mesh_vertices[:, 2], (p[2] + 0.03)) ), np.greater( mesh_vertices[:, 2], (p[2] - 0.03) ) )
                    # vis_pc(mesh_vertices[(mesh_vertices[:,2] < (p[1] + 0.03)) & (mesh_vertices[:,2] > (p[1] - 0.03))])
                # (v-x)
                vertices_to_point_vectors = mesh_vertices - p
                # dot product with normals
                angles = np.sum(vertices_to_point_vectors * mesh_vertex_normals, axis=1)
                # mask
                if planar:
                    #mask = ((angles > 0) & (mask_planar))
                    mask = np.logical_and((angles > 0), (mask_planar))
                else:
                    mask = angles > 0
                #mask = ~mask
                mask = np.logical_not(mask)
                vertices_masked = mesh_vertices.copy()
                vertices_masked[mask, :] = np.nan
                p_to_v = vertices_masked - p
                # euclidean dist
                norm_p_to_v = np.sqrt((p_to_v ** 2).sum(-1))
                # dummy lar val for nan
                norm_p_to_v[norm_p_to_v == 0] = 100
                # closest vertex idx
                idx_v_rig_p = np.argmin(np.where(np.isnan(norm_p_to_v), np.inf, norm_p_to_v))
                rigged_points[i+idx] = idx_v_rig_p
        return rigged_points

    def process_batch(batch, mesh):
        rigged_batch = []
        v_rig_idx_batch = []
        for idx, p in enumerate(batch):
            #if idx > 10000:
            #    print()
            # (v-x)
            #mask_planar = (mesh.vertices[:, 2] < (p[2] + 0.03)) & (mesh.vertices[:, 2] > (p[2] - 0.03))
            #planar_vertices = np.ma.array(mesh.vertices, mask=np.repeat(mask_planar[:, None], 3, axis=1))
            vertices_to_point_vectors = mesh.vertices - p
            # vis_pc(mesh.vertices[(mesh.vertices[:,2] < (p[1] + 0.01)) & (mesh.vertices[:,2] > (p[1] - 0.01))])
            angles = np.einsum('ij,ij->i', vertices_to_point_vectors, mesh.vertex_normals)
            #mask = ((angles > 0) & (mask_planar))
            mask = angles > 0
            # masked values of True exclude the corresponding element from any computation in numpy.ma.array
            mask = ~mask
            vertices_masked = np.ma.array(mesh.vertices, mask=np.repeat(mask[:, None], 3, axis=1))
            p_to_v = vertices_masked - p
            norm_p_to_v = np.sqrt(np.ma.sum(p_to_v ** 2, axis=1))
            idx_v_rig_p = np.ma.argmin(norm_p_to_v)
            rigged_batch.append((idx_v_rig_p, p))
        return rigged_batch

    n_query_points = 10000
    use_numba = True
    debug = False
    if use_numba:
        mesh_vertices = np.asarray(mesh.vertices)
        mesh_vertex_normals = np.asarray(mesh.vertex_normals)
        start = time.time()
        closest_vertex_index_thighs = process_batch_numba(points_thighs, mesh_vertices, mesh_vertex_normals, planar=False)
        closest_vertex_index_thighs = closest_vertex_index_thighs.astype(int)
        closest_vertex_index = np.concatenate([closest_vertex_index_rest, closest_vertex_index_thighs])
        points = np.concatenate([points_rest, points_thighs])
        part_id = torch.tensor(part_num[closest_vertex_index]) #np.concatenate([part_id_rest, part_id_thighs])
        skinning_weights = smpl.lbs_weights[closest_vertex_index]
        end = time.time()
        print("Using Numba. Time: ", end - start)
    else:
        #mesh_tree = spatial.KDTree(mesh.vertices)
        print(f"Rigging.... No of tasks: {int(len(points)/n_query_points)}")
        start = time.time()
        if debug:
            results = process_batch(points, mesh)
        else:
            import joblib
            results = joblib.Parallel(n_jobs=-1, verbose=13)(
                joblib.delayed(process_batch)(points[i:i + n_query_points], mesh)
                for i in range(0, points.shape[0], n_query_points)
            )
        results = np.concatenate(results)
        closest_vertex_index, points = zip(*results)
        points = np.array(points)
        closest_vertex_index = np.array(closest_vertex_index)
        skinning_weights = smpl.lbs_weights[closest_vertex_index]
        part_id = torch.tensor(part_num[closest_vertex_index])
        end = time.time()
        print("Time: ", end - start)

    if debug:
        import open3d as o3d
        all = True
        if all:
            # Create a list to store the geometries
            geometries = []
            color_list = [
                [1.0, 0.0, 0.0],  # Red
                [0.0, 1.0, 0.0],  # Green
                [0.0, 0.0, 1.0],  # Blue
                [1.0, 1.0, 0.0],  # Yellow
                [1.0, 0.0, 1.0],  # Magenta
                [0.0, 1.0, 1.0],  # Cyan
                [0.5, 0.5, 0.5],  # Gray
                [0.8, 0.2, 0.2],  # Dark Red
                [0.2, 0.8, 0.2],  # Dark Green
                [0.2, 0.2, 0.8],  # Dark Blue
                [0.8, 0.8, 0.2],  # Dark Yellow
                [0.8, 0.2, 0.8],  # Dark Magenta
                [0.2, 0.8, 0.8],  # Dark Cyan
                [0.9, 0.6, 0.2],  # Orange
                [0.6, 0.9, 0.2],  # Light Green
                [0.2, 0.6, 0.9],  # Light Blue
                [0.9, 0.2, 0.6],  # Pink
                [0.6, 0.2, 0.9],  # Purple
                [0.2, 0.9, 0.6],  # Teal
                [0.8, 0.5, 0.2],  # Brown
                [0.5, 0.8, 0.2],  # Olive
                [0.2, 0.5, 0.8],  # Navy
                [0.9, 0.9, 0.2],  # Bright Yellow
                [0.9, 0.2, 0.9],  # Bright Magenta
                [0.2, 0.9, 0.9],  # Bright Cyan
                [0.5, 0.5, 0.8],  # Dark Gray
            ]
            # Loop through your point clouds
            for i in range(list(dict_part.values())[-1] + 1):
                # Create a point cloud geometry
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points[(part_id == i)])
                pcd.paint_uniform_color(color_list[i])
                # Add the geometry to the list
                geometries.append(pcd)
            o3d.visualization.draw_geometries(geometries)

            combined_pcd = o3d.geometry.PointCloud()
            for pcd, color in zip(geometries, color_list):
                pcd.paint_uniform_color(color)
                combined_pcd += pcd
            # Save the combined point cloud to a PLY file
            #o3d.io.write_point_cloud("/home/varora/Documents/thighfix/normals_closdist.ply", combined_pcd)

        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(points[(part_id == 8)])
        pcd1.paint_uniform_color([0, 1, 0])

        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(points[(part_id == 18)])  # + [0, 0.7, 0])
        pcd2.paint_uniform_color([1, 0, 0])

        pcd0 = o3d.geometry.PointCloud()
        pcd0.points = o3d.utility.Vector3dVector(points[(part_id == 0)])
        pcd0.paint_uniform_color([0, 0, 1])

        o3d.visualization.draw_geometries([pcd1, pcd2, pcd0])

    return skinning_weights, part_id


def get_skinning_weights_batch(points, smpl_output, smpl, free_verts=None):
    """ 
    Batch version of previous function 
    
    """
    
    # is_batch_data = (len(points.shape) == 3)
    
    assert len(points.shape) == 3, 'points should be [B,T, 3]'
    assert isinstance(points, np.ndarray), 'points should be numpy array'
    if free_verts is not None:
        assert isinstance(free_verts, np.ndarray), 'free_verts should be numpy array'
    assert smpl_output.vertices.shape[0] >= 1, 'smpl_output should be batch size 1'
    assert len(smpl_output.vertices.shape) == 3, 'smpl_output should have a batch dimension' 
    
    # For each smpl vertex, gives the corresponding part
    B = points.shape[0]
    assert points.shape[0] == smpl_output.vertices.shape[0]
    with open('../v2p.pkl', 'rb') as f:
            data = pickle.load(f)
    part_list = sorted(set(data))
    dict_part={}
    for idx,ele in enumerate(part_list):
        dict_part[ele] = idx 
    part_num = np.array([ dict_part[v] for v in data])
    
    skinning_weights_batch = []
    part_ids_batch = []
    for b in range(B):
        v = smpl_output.vertices[b].cpu().numpy()
        if free_verts is not None:
            v = free_verts
            
        # import ipdb; ipdb.set_trace()
        mesh = trimesh.Trimesh(v, smpl.faces, process=False)
        prox_query = trimesh.proximity.ProximityQuery(mesh)
        closest_vertex, closest_vertex_index = prox_query.vertex(points[b])
        skinning_weights = smpl.lbs_weights[closest_vertex_index]
        part_id = torch.tensor(part_num[closest_vertex_index])
        # Add to lists
        skinning_weights_batch.append(skinning_weights.unsqueeze(0))
        part_ids_batch.append(part_id.unsqueeze(0))
        
    return torch.cat(skinning_weights_batch, dim=0).to(device=smpl_output.vertices.device), torch.cat(part_ids_batch, dim=0).to(device=smpl_output.vertices.device)

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
            mesh.vc = mesh.vc/np.max(mesh.vc)
        elif len(values.shape) == 2:
            if maxval is None:
                maxval = np.abs(values).max()
            if norm:
                values = np.linalg.norm(values, axis=1)
                mesh.set_vertex_colors_from_weights(values)
            else:
                mesh.vc = np.abs(values)/maxval
        else:
            raise Exception(f'Unknown values shape, should be [6890] or [6890,3], got {values.shape}')
    elif skin_weights is not None:
        mesh.vc = weights2colors(skin_weights[0].detach().cpu().numpy())
        mesh.vc = mesh.vc/np.max(mesh.vc)
    
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

