import os
import random
import torch.nn.functional as F
from dataclasses import dataclass
from typing import NewType, Optional

import numpy as np
import torch
from hit.utils.smpl_utils import psm
from hit.smpl.smplx.body_models import SMPL
from hit.smpl.smplx.lbs import (batch_rigid_transform, batch_rodrigues,
                            blend_shapes, vertices2joints)

Tensor = NewType('Tensor', torch.Tensor)

@dataclass
class MySMPLOutput():
    vertices: Optional[Tensor] = None
    faces: Optional[Tensor] = None
    joints: Optional[Tensor] = None
    full_pose: Optional[Tensor] = None
    betas: Optional[Tensor] = None
    global_orient: Optional[Tensor] = None
    transl: Optional[Tensor] = None
    body_pose: Optional[Tensor] = None
    tfs: Optional[Tensor] = None
    weights: Optional[Tensor] = None
    


class MySmpl(torch.nn.Module):

    def __init__(self, model_path, gender='male', batch_size=1, **kwargs):
        super().__init__()
        
        self.nb_betas = 10
        self.nb_body_pose = 69
        
        self.B = batch_size

        self.smpl = SMPL(model_path=os.path.join(model_path, 'smpl'),
                         gender=gender,
                         batch_size=batch_size,
                         use_hands=False,
                         use_feet_keypoints=False,
                         dtype=torch.float32)

        self.faces = self.smpl.faces
        self.faces_tensor = self.smpl.faces_tensor
        self.lbs_weights = self.smpl.lbs_weights.float()
        self.gender=gender
        self.part_ids = torch.argmax(self.lbs_weights , axis=1)
        self.num_parts = max(self.part_ids)
        # self.transl = self.smpl.transl
        # self.global_orient = self.smpl.global_orient
        # self.betas = self.smpl.betas
        # self.pose_body = self.smpl.body_pose     
        
        # self.register_buffer('canonical_x_bodypose', self.x_cano(batch_size))
        self.canonical_x_bodypose = self.x_cano(batch_size)
        
        
    def x_cano(self, batch_size=1):
        body_pose = torch.zeros((batch_size, self.nb_body_pose), dtype=torch.float32)
        body_pose[:,2] =  torch.pi / 6 #Root is a separated parameter, so legs angles are at indices 0*3+2 and 1*3+2
        body_pose[:,5] =  -torch.pi / 6 
        return body_pose


    def forward(self, 
                betas,
                transl=None,
                body_pose=None,
                global_orient=None,
                displacement=None, 
                v_template=None, 
                absolute=False,
                **kwargs):
        
        """return SMPL output from params

        Args:
            smpl_params [B, 86]: smpl parameters [0-scale,1:4-trans, 4:76-thetas,76:86-betas]
            displacement [B, 6893] (optional): per vertex displacement to represent clothing. Defaults to None.

        Returns:
            verts: vertices [B,6893]
            tf_mats: bone transformations [B,24,4,4]
            weights: lbs weights [B,24]
        """
        
        self.B = betas.shape[0]
        device = betas.device
        
        # Check betas

        if betas is not None:
            assert betas.shape[1] == self.nb_betas, f'betas.shape[1] = {betas.shape[1]}, expected {self.nb_betas}'
        else:
            betas = torch.zeros((self.B, self.nb_betas),dtype=torch.float32).to(device)
        
        # Translation
        if transl is not None:
            assert transl.shape[1] == 3, f'transl.shape[1] = {transl.shape[1]}'
        else:
            transl = torch.zeros((self.B, 3),dtype=torch.float32).to(device)
            
        #body_pose
        if body_pose is not None:
            assert body_pose.shape[1] == self.nb_body_pose, f'body_pose.shape[1] = {body_pose.shape[1]}, expected {self.nb_body_pose}'
        else :
            body_pose = torch.zeros((self.B, self.nb_body_pose),dtype=torch.float32).to(device)
            
        #global_orient
        if global_orient is not None:
            assert global_orient.shape[1] == 3, f'global_orient.shape[1] = {global_orient.shape[1]}, expected 3'
        else:
            global_orient = torch.zeros((self.B, 3),dtype=torch.float32).to(device)
            
        # Check batch size
        assert betas.shape[0] == transl.shape[0] == body_pose.shape[0] == self.B

        if v_template is not None:
            betas = 0*betas

        smpl_output = self.smpl.forward(betas=betas,
                                        transl=transl,
                                        body_pose=body_pose,
                                        global_orient=global_orient,
                                        return_verts=True,
                                        v_template=v_template,
                                        displacement=displacement,
                                        return_full_pose=True)


        verts = smpl_output.vertices.clone()
        
        tf_mats = smpl_output.T.clone()
        tf_mats[:, :, :3, 3] += transl.unsqueeze(1)
        
        self.canonical_x_bodypose = self.canonical_x_bodypose.to(device)

        if absolute == False:
            # print(betas)
            body_posecanonical = self.canonical_x_bodypose.expand(self.B, -1).clone()
            output_cano = self.forward(betas=betas, body_pose=body_posecanonical, v_template=v_template, absolute=True)

            smpl_tfs_cano = output_cano.tfs
            
            try:
                # print(f"smpl_tfs_cano.shape = {smpl_tfs_cano.shape}")
                # import ipdb; ipdb.set_trace()
                tfs_c_inv = smpl_tfs_cano.inverse()
            except:
                # print('Warning: inverse of canonical tfs failed, using pseudo inverse')
                print('Warning: inverse of canonical tfs failed, values are', smpl_tfs_cano)
                raise Exception('Inverse of canonical tfs failed, values are', smpl_tfs_cano)
            
            tf_mats = torch.einsum('bnij,bnjk->bnik', tf_mats, tfs_c_inv)

        joints = smpl_output.joints.clone()
              
        output = MySMPLOutput(vertices= verts.float(),
                             faces=self.smpl.faces,
                             global_orient=global_orient,
                             body_pose=body_pose,
                             transl=transl,
                             joints= joints.float(),
                             betas=betas,
                             tfs=tf_mats, 
                             weights=smpl_output.weights.float(),
                             full_pose=smpl_output.full_pose,
                             )
        return output
    
    def forward_canonical(self, betas=None):
        if betas is None:
            betas = torch.zeros((1, self.nb_betas),dtype=torch.float32).to(self.canonical_x_bodypose.device)
        
        # import ipdb; ipdb.set_trace()
        body_posecanonical = self.canonical_x_bodypose.clone().expand(betas.shape[0], -1)
        if betas is not None:
            body_posecanonical = body_posecanonical.to(betas.device)
        output_cano = self.forward(betas=betas, body_pose=body_posecanonical, absolute=True)
        return output_cano
        
    def plot_parts(self):
        from psbody.mesh import Mesh
        smpl_template_verts = self.forward_canonical().vertices[0].cpu().numpy()
        part_ids = self.part_ids 
        import ipdb; ipdb.set_trace()

        part_colors = color_gradient(part_ids.shape[0], scale=1, shuffle=True, darken=False, pastel=False, alpha=None, seed=1)
        verts_color = part_colors[part_ids]
 
        m = Mesh(smpl_template_verts, self.faces, vc=verts_color)
        m.show()
        
    def psm(self, verts, values=None, indices=False, maxval=None, norm=False, display=False, skin_weights=None):
        smpl = self
        return psm(verts, smpl, values, indices, maxval, norm, display, skin_weights)
    
    def forward_shaped(self, betas, body_pose, global_orient=None):
        """ return SMPL vertices with applied shape and pose dependant blend shapes"""
        
        B = betas.shape[0]
        if global_orient is None:
            global_orient = torch.zeros((B, 3),dtype=torch.float32).to(betas.device)
                  
        full_pose = torch.cat([global_orient, body_pose], dim=1)
        shaped_vertices = v_shaped(betas, full_pose, self.smpl.v_template, self.smpl.shapedirs, self.smpl.posedirs, pose2rot=True, dtype=torch.float32)
        
        beta_zero = torch.zeros_like(betas)
        smpl_output = self.smpl.forward(betas=beta_zero,
                                        body_pose=self.canonical_x_bodypose.expand(B, -1),
                                        global_orient=global_orient,
                                        return_verts=True,
                                        v_template=shaped_vertices,
                                        return_full_pose=True)

        return smpl_output.vertices
    
    
    def pose_offsets(self, body_pose):
        
        batch_size = body_pose.shape[0]
        pose_feature = self.pose2features(body_pose)
        # (N x P) x (P, V * 3) -> N x V x 3
        pose_offsets = torch.matmul(pose_feature, self.smpl.posedirs).view(batch_size, -1, 3)

        return pose_offsets
    
    
    def pose2features(self, body_pose):
        global_orient = torch.zeros_like(body_pose[:, :3])
        full_pose = torch.cat([global_orient, body_pose], dim=1)
        
        fullpose = full_pose
        dtype = fullpose.dtype
        device = fullpose.device
        batch_size = fullpose.shape[0]
        ident = torch.eye(3, dtype=dtype, device=device)

        rot_mats = batch_rodrigues(
            fullpose.view(-1, 3), dtype=dtype).view([batch_size, -1, 3, 3])

        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        
        return pose_feature
    
    
    def forward_numpy(self, smpl_data, device='cpu'):
        # Run a SMPL forward pass to get the SMPL numpy body vertices from a numpy SMPL dictionary
        to_torch = lambda x: torch.from_numpy(x).float().to(device)
        
        poses_smpl = to_torch(smpl_data['poses'])
        trans_smpl = to_torch(smpl_data['trans'])
        betas_smpl = to_torch(smpl_data['betas'][:self.smpl.num_betas]).expand(trans_smpl.shape[0], -1)
        
        # Run a SMPL forward pass to get the SMPL body vertices
        smpl_output = self.smpl(betas=betas_smpl, body_pose=poses_smpl[:,3:], transl=trans_smpl, global_orient=poses_smpl[:,:3])
        verts = smpl_output.vertices
        return verts
    
    def compute_bone_trans(self, full_pose, posed_joints):
        """
        Code adapted from COAP https://github.com/markomih/COAP
        """
        # full_pose: B, K*3 or B,K,3
        # posed_joints: B, K*3 or B,K,3
        B = full_pose.shape[0]
        full_pose = full_pose.reshape(B, -1, 3)
        posed_joints = posed_joints.reshape(B, -1, 3)
        K = full_pose.shape[1]
        
        joint_mapper = torch.ones(K, dtype=torch.bool)
        # joint_mapper[self.merge_body_parts] = False
        mK = joint_mapper.shape[0]
        # posed_joints = posed_joints[:, :K]  # remove extra joints (landmarks for smplx)
        
        # torchgeometry.angle_axis_to_rotation_matrix(full_pose.view(-1, 3))[:, :3, :3]
        rot_mats = batch_rodrigues(full_pose.view(-1, 3)).view(B, K, 3, 3)  # B,K,3,3

        # fwd lbs to estimate absolute transformation matrices
        parents = self.smpl.parents.long()
        transform_chain = [rot_mats[:, 0]]
        for i in range(1, parents.shape[0]):
            if i == mK:
                break
            transform_chain.append(transform_chain[parents[i]] @ rot_mats[:, i])

        transforms = torch.stack(transform_chain, dim=1)
        abs_trans = torch.cat([
            F.pad(transforms.reshape(-1, 3, 3), [0, 0, 0, 1]),
            F.pad(posed_joints[:, :mK].reshape(-1, 3, 1), [0, 0, 0, 1], value=1)
        ], dim=2).reshape(B, mK, 4, 4)

        # remap joints
        # print(f'joint_mapper: {self.joint_mapper}')
        abs_trans = abs_trans.transpose(0, 1)[joint_mapper].transpose(0, 1)
        bone_trans = torch.inverse(abs_trans)
        return bone_trans
        
    def get_bbox_bounds_trans(self, vertices, bone_trans):
        """
        Code adapted from COAP https://github.com/markomih/COAP
        Args:
            vertices (torch.tensor): (B,V,3)
            bone_trans (torch.tensor): (B,K,4,4)
        Returns:
            tuple:
                bbox_min (torch.tensor): BxKx1x3
                bbox_max (torch.tensor): BxKx1x3
        """
        B, K = bone_trans.shape[0], bone_trans.shape[1]
        lbs_weights = self.tensor_to_numpy(self.lbs_weights).copy()
        max_part = lbs_weights.argmax(axis=1)
        
        valid_labels = list(range(K))
        tight_vert_selector = [ # (K, V_k)
            torch.where(torch.from_numpy(max_part == valid_labels[i]))[0] for i in range(K) 
        ]
        max_elem = max([v_sec.shape[0] for v_sec in tight_vert_selector])
        for b_part in range(len(tight_vert_selector)):
            vert_sec = torch.full((max_elem,), dtype=tight_vert_selector[b_part].dtype,
                                fill_value=tight_vert_selector[b_part][0].item())
            vert_sec[:tight_vert_selector[b_part].shape[0]] = tight_vert_selector[b_part]
            tight_vert_selector[b_part] = vert_sec
        
        tight_vert_selector = torch.stack(tight_vert_selector)  # (K, V_k)
        
        max_select_dim = tight_vert_selector.shape[1]

        part_vertices = torch.index_select(vertices, 1, tight_vert_selector.view(-1))  # (B, K*max_select_dim, 3)
        part_vertices = part_vertices.view(B, K, max_select_dim, 3)

        bone_trans = bone_trans.unsqueeze(2).expand(-1, -1, max_select_dim, -1, -1)

        local_part_vertices = (bone_trans @ F.pad(part_vertices, [0, 1], "constant", 1.0).unsqueeze(-1))[..., :3, 0]
        bbox_min = local_part_vertices.min(dim=-2, keepdim=True).values  # (B, K, 1, 3)
        bbox_max = local_part_vertices.max(dim=-2, keepdim=True).values  # (B, K, 1, 3)

        return bbox_min, bbox_max
    
    @staticmethod
    def tensor_to_numpy(tensor_vec):
        if torch.is_tensor(tensor_vec):
            return tensor_vec.detach().cpu().numpy()
        return tensor_vec
  

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


def v_shaped(betas, pose, v_template, shapedirs, posedirs, pose2rot=True, dtype=torch.float32):
    ''' Apply the shape and pose deb blend shape to the template mesh without applyting the transformation

        Parameters
        ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters
        pose : torch.tensor Bx(J + 1) * 3
            The pose parameters in axis-angle format
        v_template torch.tensor BxVx3
            The template mesh that will be deformed
        shapedirs : torch.tensor 1xNB
            The tensor of PCA shape displacements
        posedirs : torch.tensor Px(V * 3)
            The pose PCA coefficients
        J_regressor : torch.tensor JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        parents: torch.tensor J
            The array that describes the kinematic tree for the model
        lbs_weights: torch.tensor N x V x (J + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        pose2rot: bool, optional
            Flag on whether to convert the input pose tensor to rotation
            matrices. The default value is True. If False, then the pose tensor
            should already contain rotation matrices and have a size of
            Bx(J + 1)x9
        dtype: torch.dtype, optional

        Returns
        -------
        verts: torch.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: torch.tensor BxJx3
            The joints of the model
    '''

    batch_size = max(betas.shape[0], pose.shape[0])
    device = betas.device

    # Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    ident = torch.eye(3, dtype=dtype, device=device)

    if pose2rot:
        rot_mats = batch_rodrigues(
            pose.view(-1, 3), dtype=dtype).view([batch_size, -1, 3, 3])

        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1]) # 1,207
        # (N x P) x (P, V * 3) -> N x V x 3
        pose_offsets = torch.matmul(pose_feature, posedirs).view(batch_size, -1, 3)
    else:
        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
        rot_mats = pose.view(batch_size, -1, 3, 3)

        pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
                                    posedirs).view(batch_size, -1, 3)


    v_posed = pose_offsets + v_shaped

       
    return v_posed

    
if __name__ == '__main__':
    mysmpl = MySmpl(model_path='/ps/project/rib_cage_breathing/TML/Data/smplx_models')
    out = mysmpl.forward(torch.zeros((2,10)))
    out = mysmpl.forward(torch.zeros((4,10)))
    out = mysmpl.forward(torch.zeros((6,10)))
    print(out)
    
    betas = torch.zeros((1,10))
    body_pose = torch.randn((1,69))
    
    verts = mysmpl.forward_shaped(betas, body_pose)
    
    pose_offsets = mysmpl.pose_offsets(body_pose)
    mysmpl.psm(verts, pose_offsets, display=True)
    
    mysmpl.psm(verts, display=True)
    import ipdb; ipdb.set_trace()

    # mm.plot_parts()
    
    
