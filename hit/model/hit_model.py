import copy

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from utils.extract_pretrained_gdna import get_state_dict

from utils.smpl_utils import get_skinning_weights, weights2colors
from utils.tensors import cond_create
from model.deformer import ForwardDeformer, skinning
from model.helpers import expand_cond
from model.network import ImplicitNetwork
from model.generator import Generator
from skimage import measure
from hit.training.mri_sampling_utils import load_occupancy


class HITModel(torch.nn.Module):

    def __init__(self, train_cfg, smpl) -> None:
        super().__init__()
        self.train_cfg = train_cfg
        self.apply_compression = True
        self.smpl = smpl

        ###################### Load the configuration for all the submodules MLP networks
        
        lbs_network_conf = train_cfg.networks.lbs
        beta_network_conf = train_cfg.networks.beta # Beta displacement field config
        
        # Compressor config
        if self.train_cfg['compressor']:
            compressor_conf = train_cfg.networks.compression
        else:
            compressor_conf = None
            
        # Pose dep blend shapes config
        if self.train_cfg.pose_bs is True:
            pose_bs_conf = train_cfg.networks.pose_bs
        else:
            pose_bs_conf = None
            
        if self.train_cfg['forward_beta_mlp']:
            fwd_beta_conf = train_cfg.networks.beta_fwd
            self.fwd_beta = ImplicitNetwork(**fwd_beta_conf)
        else:
            self.fwd_beta = None
            

        # tissue occupancy prediction config
        tissue_cfg = train_cfg.networks.tissue
        tissue_cfg.update({'d_out': len(self.train_cfg.mri_labels)}) # This network outputs a class score of each tissue
        self.tissues_occ = ImplicitNetwork(**tissue_cfg)
        
        if self.train_cfg['use_generator'] == False:
            self.generator = None
        else:
            self.generator = Generator(64)
            
        if train_cfg['mri_values']:
            # Instanciate an MLP that predicts the MRI values
            mri_val_net_conf = train_cfg.networks.mri_val
            self.mri_val_net = ImplicitNetwork(**mri_val_net_conf)
        else:
            self.mri_val_net = None
        
        self.deformer = ForwardDeformer(lbs_network_conf=lbs_network_conf, 
                                        beta_network_conf=beta_network_conf,
                                        compressor_conf=compressor_conf,
                                        pose_bs_conf=pose_bs_conf,
                                        nb_root_init=train_cfg['root_init'],
                                        apply_compression=self.apply_compression)


            
    def initialize(self, pretrained=True, train_cfg = None, device=None, checkpoint_path=None):
        # self = COAPBodyModel(parametric_body, train_cfg)
        gender = self.smpl.gender
        if pretrained and checkpoint_path is None:
            model_type = self.model_type
            checkpoint = f'https://github.com/markomih/COAP/blob/dev/models/coap_{model_type}_{gender}.ckpt?raw=true'
            state_dict = torch.hub.load_state_dict_from_url(checkpoint, progress=True)
            self.load_state_dict(state_dict['state_dict'])
        elif checkpoint_path is not None:
            print('Using checkpoint from ', checkpoint_path)
            checkpoint = checkpoint_path
            state_dict = torch.load(checkpoint)
            self.load_state_dict(state_dict['state_dict'])    
        if train_cfg['to_train'] == 'pretrain':
            pass
            # parametric_body.hit.network.load_state_dict(get_state_dict(to_extract='network', source='gdna'))
            # parametric_body.hit.generator.load_state_dict(get_state_dict(to_extract='generator', source='gdna')  )      
            # self.deformer.disp_network.load_state_dict(get_state_dict(to_extract='deformer.disp_network', source='gdna')) # undo beta
            # self.deformer.lbs_network.load_state_dict(get_state_dict(to_extract='deformer.lbs_network', source='gdna')) # undo beta
        else:
            if train_cfg['load_pretrained_lbs'] == True:
                self.deformer.lbs_network.load_state_dict(get_state_dict(to_extract='deformer.lbs_network', source=f'pretrained_{gender}')) # undo beta           
            self.deformer.disp_network.load_state_dict(get_state_dict(to_extract='deformer.disp_network', source=f'pretrained_{gender}')) # undo beta
            if train_cfg['compressor'] and train_cfg['load_pretrained_compressor'] == True:
                self.deformer.compressor.load_state_dict(get_state_dict(to_extract='deformer.compressor', source=f'pretrained_compressor_{gender}')) # undo beta
                self.deformer.compressor.load_state_dict(get_state_dict(to_extract='deformer.compressor', source=f'pretrained_compressor_{gender}')) # undo beta
            
            if device is not None:
                self.smpl = self.smpl.to(device=device)


    def forward_rigged(self, betas, body_pose=None, global_orient=None, transl=None, do_compress=False, mise_resolution0=32, mise_depth=3, **kwargs):
        
        # smpl shaped in xpose
        # smpl = MySmpl(model_path=cg.smplx_models_path, modelgender=self.smpl.gender, device=betas.device)
        smpl = self.smpl
        smpl_output_xpose = smpl.forward(betas=betas, body_pose=smpl.x_cano().to(betas.device), global_orient=None, transl=None)  
         
        # smpl shaped and posed
        smpl_output = smpl.forward(betas=betas, body_pose=body_pose, global_orient=global_orient, transl=transl)  
        tfs = smpl_output.tfs 
            
        mesh_p_list = []
        mesh_c_list = []
        for ci, c_label in enumerate(self.train_cfg.mri_labels):
            if c_label != 'NO':
                mesh_s = self.extract_mesh(smpl_output_xpose, channel=ci, grid_res=64, 
                                        use_mise=True, mise_resolution0=mise_resolution0,
                                        mise_depth=mise_depth, batch=None, 
                                        template=False, 
                                        unposed = True, # The compression should be applied after posing
                                        color_mode='compression')[0]
                
                mesh_p = self.pose_unposed_tissue_mesh(mesh_s, smpl_output, do_compress=do_compress)
                
                # import ipdb; ipdb.set_trace()
                mesh_p_list.append(mesh_p)
                mesh_c_list.append(mesh_s)
        return mesh_p_list, mesh_c_list
    
    
    def extract_shaped_mesh(self, smpl_output, channel=1, grid_res=64, max_queries=None, use_mise=False, mise_resolution0=32, bound_by_smpl=False):
        
        body_pose = self.smpl.forward_canonical(betas= torch.zeros_like(smpl_output.betas)).body_pose
        smpl_output_xpose = self.smpl.forward(betas=smpl_output.betas, 
                                              body_pose=body_pose, 
                                              global_orient=None, transl=None)  
        
        mesh_s_list = self.extract_mesh(smpl_output_xpose, channel=channel, grid_res=64, 
                                        use_mise=True, mise_resolution0=mise_resolution0,
                                        mise_depth=3, batch=None, 
                                        template=False, 
                                        unposed = True, # The compression should be applied after posing
                                        color_mode='compression',
                                        bound_by_smpl=bound_by_smpl)
        return mesh_s_list
    
    
    def pose_unposed_tissue_mesh(self, mesh_s, smpl_output, do_compress=False):
        betas = smpl_output.betas
        body_pose = smpl_output.body_pose
        tfs = smpl_output.tfs 
        cond = cond_create(betas, body_pose, self.generator, self.smpl)
        x_s = torch.FloatTensor(np.array(mesh_s.vertices)).to(betas.device).unsqueeze(0)
        x_c = self.deformer.disp_network(x_s, cond) + x_s

        # skinning
        w = self.deformer.query_weights(x_c, {'latent': cond['lbs'], 'betas': cond['betas']*0})               
        xd = skinning(x_s[0], w[0], tfs, inverse=False)
        
        if do_compress:
            raise DeprecationWarning("This is wrong, do not use. Instead use the posed extraction")
            d_p = self.deformer.compressor(x_s, cond)
            xd = xd + d_p[0]
        
        mesh_p = trimesh.Trimesh(vertices=xd.detach().cpu().numpy(), faces=mesh_s.faces)
        return mesh_p
                
                
        
    def can_occ_query(self, pts_c, n_batch, n_dim, cond, both=False):
        
        if both:
            # occ_smpl_pd = self.network(
            #     pts_c.reshape((n_batch, -1, n_dim)), 
            #     cond={'latent': cond['latent']},
            #     mask=None,
            #     val_pad=-1000,
            #     return_feat=False,
            #     spatial_feat=True,
            #     normalize=True)
            pass
            
            if (self.train_cfg['to_train'] in ['pretrain', 'compression']):
                occ_smpl_pd = self.network(
                    pts_c.reshape((n_batch, -1, n_dim)), 
                    cond={'latent': cond['latent']},
                    mask=None,
                    val_pad=-1000,
                    return_feat=False,
                    spatial_feat=True,
                    normalize=True)
                occ_tissue_pd = occ_smpl_pd
            
            else:
                occ_tissue_pd = self.tissues_occ(
                    pts_c.reshape((n_batch, -1, n_dim)), 
                    cond={'latent': cond['latent']},
                    mask=None,
                    val_pad=-1000,
                    return_feat=False,
                    spatial_feat=True,
                    normalize=True)
                
                occ_smpl_pd = None
                
            # if self.train_cfg['mask_smpl']:
            # if True:
            #     import ipdb; ipdb.set_trace()
            #     occ_tissue_pd = torch.sigmoid(occ_tissue_pd)
            #     occ_smpl_pd = torch.sigmoid(occ_smpl_pd) 
            #     is_outside = (occ_smpl_pd < 0.5).float()
            #     c0 = 1-occ_smpl_pd
            #     c_no = occ_smpl_pd[...,0] 
            #     C = occ_tissue_pd.shape[-1]
            #     occ_tissue_pd_outside = occ_tissue_pd.clone()
            #     for ci in range(1, C):
            #         import ipdb; ipdb.set_trace()
            #         occ_tissue_pd_outside[..., ci] = occ_tissue_pd[..., ci] + (c_no/(C-1)) - (c0/(C-1))
            #     occ_tissue_pd_outside[..., 0] = c0
                
            #     occ_tissue_pd[is_outside.bool()] = occ_tissue_pd_outside[is_outside.bool()]   
            
            return occ_smpl_pd, occ_tissue_pd
        
        if self.train_cfg['to_train'] == 'pretrain':
            occ_pd = self.network(
                pts_c.reshape((n_batch, -1, n_dim)), 
                cond={'latent': cond['latent']},
                mask=None,
                val_pad=-1000,
                return_feat=False,
                spatial_feat=True,
                normalize=True) # B, T*n_init, 
        else: 
            occ_pd = self.tissues_occ(
                pts_c.reshape((n_batch, -1, n_dim)), 
                cond={'latent': cond['latent']},
                mask=None,
                val_pad=-1000,
                return_feat=False,
                spatial_feat=True,
                normalize=True)
            

        return occ_pd

    def canonicalize_from_similar(self, x_p, xx_c, smpl_tfs, cond, undo_shape=True):
        """
        xf_p : Points to unpose
        xv_p : Posed points to get w from
        smpl_tfs : Bone transforms of the posed smpl
        undo_shape : if true, undo beta
        return :  unposed points x_c 
        
        
        """
        
        B = x_p.shape[0]
        T = x_p.shape[1]
        
        w = self.deformer.query_weights(xx_c, {'latent': cond['lbs'], 'betas': cond['betas']})
        
        x_p_flat = x_p.reshape(-1, 3)
        tfs = expand_cond(smpl_tfs, x_p)
        tfs_flat = tfs.reshape(-1, tfs.shape[-3], 4, 4)
        w_flat = w.reshape(-1, w.shape[-1])  
        
        # import ipdb; ipdb.set_trace()
        x_s_flat = skinning(x_p_flat, w_flat, tfs_flat, inverse=True) 
        x_s = x_s_flat.reshape(B, T, 3)
        
        if undo_shape:
            x_c = self.deformer.disp_network(x_s, cond) + x_s 
            return x_c
        else:
            return x_s
            
                        
    def query(self, points, smpl_output, part_id=None, skinning_weights=None, is_inside=None, signed_dist=None, ret_intermediate=False, eval_mode=False, template=False, unposed=False, apply_compression=True):
        """
        Args:
            points (torch.tensor): Query points of shape [B,T,3]
            template: The input pc is in the SMPL template space
            smpl_output: object with the following attributes 'betas', 'body_pose', 'full_pose', 'get', 'global_orient', 'items', 'joints', 'keys', 'n_free', 'transl', 'v_free', 'v_shaped', 'values', 'vertices']
           
        This function defines the following variables: 
        B : batch size
        K : number of parts
        T : number of query 3D points
        """

        
        T = points.shape[1]
        n_dim = 3
        B = points.shape[0]
        n_batch = B

        pts_d = points
        mask = None
        cond = cond_create(smpl_output.betas, smpl_output.body_pose, None, self.smpl)             
        smpl_tfs = smpl_output.tfs

        if mask is None:
            mask = torch.ones( (n_batch, T), device=pts_d.device, dtype=torch.bool)

        #jumphere
        if template:
            pts_c = pts_d
            
            
            occ_smpl_pd, occ_pd = self.can_occ_query(pts_c, n_batch, n_dim, cond, both=True)
            
            occ_pd2 = occ_pd.squeeze(-1)
            # occ_smpl_pd = occ_smpl_pd.squeeze(-1)
            pts_c2 = pts_c 
            
        elif unposed:    
            
            pts_c = self.deformer.query_cano(points, 
                                            {'betas': cond['betas'],
                                             'thetas': cond['thetas']}, 
                                            mask=None)
            
            occ_smpl_pd, occ_pd = self.can_occ_query(pts_c, n_batch, n_dim, cond, both=True) # B, T*n_init, C
            
            occ_pd2 = occ_pd.squeeze(-1)
            # occ_smpl_pd = occ_smpl_pd.squeeze(-1)
            pts_c2 = pts_c 
  
        else:
            assert not skinning_weights is None, "Skinning weights are required now for unposing. Edit code if you want to find multiple roots"

            pts_c, others = self.deformer.forward(pts_d,
                                            {'betas': cond['betas'],
                                             'thetas': cond['thetas'],
                                            'latent': cond['lbs']},
                                            tfs=smpl_tfs,
                                            mask=mask,
                                            eval_mode=eval_mode,
                                            part_id = part_id,
                                            skinning_weights=skinning_weights)
            
            # test to offset can space by a constant
            #pts_c = pts_c + 0.1 #try1

            occ_smpl_pd, occ_tissue = self.can_occ_query(pts_c, n_batch, n_dim, cond, both=True)# B, T*n_init, C
                   
            C = occ_tissue.shape[-1]
            occ_tissue = occ_tissue.reshape(n_batch, T, -1, C) 
            # occ_smpl_pd = occ_smpl_pd.reshape(n_batch, T, -1, 1)
            
            if C == 1:         
                occ_pd, idx_c = occ_tissue.max(dim=2)
                pts_c = torch.gather(pts_c, 2, idx_c.unsqueeze(-1).expand(-1,-1, 1, pts_c.shape[-1])).squeeze(2)
            else:
                
                # if part_id is not None: 
                #     import ipdb; ipdb.set_trace()
                    
                # import ipdb; ipdb.set_trace()
                # unposing strategy
                # unposing_strategy = 'smpl'
                # unposing_strategy = 'tissues'
                
                # occ_smpl_pd, idx_c = occ_smpl_pd.max(dim=2)
                # if unposing_strategy == 'tissues':
                #     import ipdb; ipdb.set_trace()
                #     occ_tissue_masked = occ_tissue
                #     occ_tissue_masked[...,0] = -1e6
                #     occ_tissue_masked, _ = occ_tissue_masked.max(dim=-1)
                #     occ_tissue_masked = occ_tissue_masked.unsqueeze(-1)
                #     occ_smpl_pd, idx_c = occ_tissue_masked.max(dim=2)
                    
                    
                
                # pts_c = torch.gather(pts_c, 2, idx_c.unsqueeze(-1).expand(-1,-1, 1, pts_c.shape[-1])).squeeze(2) # B,T,3
                if pts_c.shape[2] != 1:
                    print(f"WARNING : pts_c.shape[2] != 1 : {pts_c.shape[2]}. The unposing should give a single point.")
                    import ipdb; ipdb.set_trace()
                assert pts_c.shape[2] == 1
                pts_c = pts_c[:,:,0] 
                
                # import ipdb; ipdb.set_trace()
                if self.deformer.compressor is None or not self.apply_compression or not apply_compression or self.train_cfg['skip_compression'] :
                    # import ipdb; ipdb.set_trace()
                    assert occ_tissue.shape[2] == 1
                    occ_pd = occ_tissue[:,:,0]
                    # occ_pd = torch.gather(occ_tissue, 2, idx_c.unsqueeze(-1).expand(-1,-1, 1, occ_tissue.shape[-1])).squeeze(2) # B,T,C
                    # occ_pd = torch.gather(occ_tissue, 2, idx_c.unsqueeze(-1).expand(-1,-1, 1, occ_tissue.shape[-1])).squeeze(2) # B,T,C
                else:
                    xf_c = pts_c
                    xf_p = pts_d
                    
                    #try 2
                    # pts_c = pts_c + 0.1
                    # _, occ_pd = self.can_occ_query(pts_c, n_batch, n_dim, cond, both=True) # B, T, C
                    # import ipdb; ipdb.set_trace()
                    
                    #try 3
                    # import ipdb; ipdb.set_trace()
                            
                                                    
                    #---------------------global
                    # # hypothesis: skinning weights are the same for xp and xf
                    # d = self.deformer.compressor(xf_c, cond)
                    # xv_p = xf_p+d

                    # xv_c = self.canonicalize_from_similar(xv_p, xf_c, smpl_tfs, cond)
                    # d_c = xv_c - xf_c
                    # pts_c = pts_c - d_c # This is quite close to the previous pts_c #*4 to emphize
                    
                    
                    # #---------------------global learned in xf_c
                    # d_p = self.deformer.compressor(xf_c, cond)
                    # xv_p = xf_p + d_p
                    # xv_c = self.canonicalize_from_similar(xv_p, xf_c, smpl_tfs, cond)
                    
                    # # import ipdb; ipdb.set_trace()
                    # pts_c = xv_c
                    
                    #---------------------global learned in xf_s
                    # import  ipdb; ipdb.set_trace()
                    xf_s = self.canonicalize_from_similar(xf_p, xf_c, smpl_tfs, cond, undo_shape=False)
                    # do not backpropagate through the compression
                    if self.train_cfg['no_comp_grad']:
                        with torch.no_grad():
                            d_p = self.deformer.compressor(xf_s, cond)
                    else:
                        d_p = self.deformer.compressor(xf_s, cond)

                    xv_p = xf_p + d_p
                    xv_c = self.canonicalize_from_similar(xv_p, xf_c, smpl_tfs, cond)
                    # xv_c = self.deformer.disp_network(xf_s, cond) + xf_s 
                    
                    # import ipdb; ipdb.set_trace()
                    pts_c = xv_c
                                
                    #------------------------local
                    
                    # d = self.deformer.compressor(xf_c, cond)
                    # xv_c = xf_c + d
                    # # import ipdb; ipdb.set_trace()
                    # pts_c = xv_c
                    # ----------------------------------                           
                        
                    _, occ_pd = self.can_occ_query(pts_c, n_batch, n_dim, cond, both=True) # B, T, C                   

                
                
                if self.train_cfg['smpl_mask']:
                    import ipdb; ipdb.set_trace()
                    occ_pd = occ_pd * F.softmax(occ_smpl_pd, dim=-1)
                
                
            if C == 1:
                occ_pd = occ_pd.squeeze(-1)
                occ_smpl_pd = occ_smpl_pd.squeeze(-1)
                
            occ_pd2 = occ_pd
            pts_c2 = pts_c 
        
            
        weights = self.deformer.query_weights(pts_c2, {'latent': cond['lbs'], 'betas': cond['betas']})    
            
        output = {'pred_occ':occ_pd2, 'pts_c':pts_c2, 'smpl_occ': occ_smpl_pd,  'weights':weights}
        
        return output


    

    @staticmethod
    def batchify_smpl_output(smpl_output):
        b_smpl_output_list = []
        batch_size = smpl_output.vertices.shape[0]
        for b_ind in range(batch_size):
            b_smpl_output_list.append(copy.copy(smpl_output))
            for key in b_smpl_output_list[-1].__dict__.keys(): # callling __dict__.keys() is necessary to support v_free
                
                # print(key)
                val = getattr(smpl_output, key)
                if torch.is_tensor(val):
                    val = val[b_ind:b_ind+1].clone()
                setattr(b_smpl_output_list[-1], key, val)
        return b_smpl_output_list
    
    @torch.no_grad()
    def pts_distances(self, smpl_output, points):
        
        if isinstance(smpl_output.vertices, torch.Tensor):
            verts_numpy = smpl_output.vertices.squeeze().cpu().numpy()
        else:
            
            verts_numpy = smpl_output.vertices
        mesh = trimesh.Trimesh(verts_numpy, self.smpl.faces, process=False)
        from leap.tools.libmesh import check_mesh_contains
        is_inside = check_mesh_contains(mesh, points).astype(float)
        
        # Compute distances from points to mesh surface using trimesh
        if isinstance(points, torch.Tensor):
            points_numpy = points.squeeze().cpu().numpy()
        else:
            points_numpy = points
        proximity = trimesh.proximity.ProximityQuery(mesh)
        signed_dist = proximity.signed_distance(points_numpy)
        
        is_inside = torch.FloatTensor(is_inside).to(smpl_output.vertices.device)
        signed_dist = torch.FloatTensor(signed_dist).to(smpl_output.vertices.device)
        
        return is_inside, signed_dist
    
    def query_gt(self, query_points, batch, b_ind=None):
        
        gt_occ = load_occupancy(batch, query_points.cpu().numpy(), interp_order=0, b_ind=b_ind)[0] # because load_occupancy returns [occ, body_mask]
        return torch.FloatTensor(gt_occ).to(query_points.device)
            
    @torch.no_grad()
    def extract_mesh(self, smpl_output, channel=1, grid_res=64, max_queries=None, use_mise=True, mise_resolution0=32,
                     mise_depth=3, batch=None, template=False, unposed=False, color_mode='lbs', bound_by_smpl=False):
        

        if max_queries is None:
            max_queries = int(self.train_cfg['max_queries'])
        scale = 1.1  # padding

        act = lambda x: torch.log(x/(1-x+1e-6)+1e-6)  # revert sigmoid
        level = 0.5
        occ_list = []

        verts = smpl_output.vertices
        B = verts.shape[0]
        device = verts.device
        part_colors = self.get_part_colors()
        b_smpl_output_list = self.batchify_smpl_output(smpl_output)
        
        if template == True:
            # All the meshes are the same, only extract the first one
            B = 1
            b_smpl_output_list = [b_smpl_output_list[1]]

        b_min, b_max = verts.min(dim=1).values, verts.max(dim=1).values  # B,3
        gt_center = ((b_min + b_max)*0.5).cpu()  # B,3
        gt_scale = (b_max - b_min).max(dim=-1, keepdim=True).values.cpu()  # (B,1)
        gt_scale_gpu, gt_center_gpu = gt_scale.to(device), gt_center.to(device)
        
        mesh = trimesh.Trimesh(smpl_output.vertices.detach().cpu().squeeze().numpy(), self.smpl.faces, process=False)      
        
        if use_mise:
            from leap.tools.libmise import MISE
            value_grid_list = []
            for b_ind in range(B):
                mesh_extractor = MISE(mise_resolution0, mise_depth, level)
                points = mesh_extractor.query()
                print('Running marching cube...')
                while points.shape[0] != 0:
                    grid3d = torch.FloatTensor(points).to(device)
                    grid3d = scale*(grid3d/mesh_extractor.resolution - 0.5).reshape(1, -1, 3)  # [-0.5, 0.5]*scale
                    grid3d = grid3d*gt_scale_gpu[b_ind].reshape(1, 1, 1) + gt_center_gpu[b_ind].reshape(1, 1, 3)

                    # check occupancy for sampled points
                    occ_hats = []
                    for pts in torch.split(grid3d, max_queries, dim=1):
                        
                        # import ipdb; ipdb.set_trace()
                        skinning_weights, part_id = get_skinning_weights(pts[0].cpu().numpy(), 
                                                                         b_smpl_output_list[b_ind].vertices[0].cpu().numpy(), 
                                                                         self.smpl)
                        skinning_weights = torch.FloatTensor(skinning_weights).to(device)
                        
                        if self.train_cfg.use_precomputed_dist:
                            is_inside, signed_dist = self.pts_distances(b_smpl_output_list[b_ind], pts.squeeze(0).cpu().numpy())
                        else:
                            is_inside, signed_dist = None, None
 
                        if batch is not None: 
                            # Fetch GT occupancy
                            pred = self.query_gt(pts.to(device=device), batch, b_ind)
                            occ_tissue = (pred == channel).float()
                            occ_hats.append(occ_tissue.cpu())
                            
                        else:   
                            # import ipdb; ipdb.set_trace() 
                            output = self.query(pts.to(device=device), b_smpl_output_list[b_ind], 
                                          ret_intermediate=False, eval_mode=True, 
                                          template=template, unposed=unposed,
                                          part_id=part_id, skinning_weights=skinning_weights)
                            if channel == -1:
                                # import ipdb; ipdb.set_trace()
                                pred = output['smpl_occ'].squeeze(-1)
                            else:
                                pred = output['pred_occ']
                            
                            if len(self.train_cfg.mri_labels) > 1 and channel != -1: 
                                
                                import torch.nn.functional as F
                                occ_all_tissue = F.softmax(pred, dim=-1) 
                                
                                # occ_smpl = output['smpl_occ']
                                # occ_all_tissue = torch.cat((occ_all_tissue, occ_smpl), dim=-1)
                                
                                sorted_idx = torch.argsort(occ_all_tissue, dim=-1, descending=True)
                                                   
                                best_idx = sorted_idx[:,:,0]
                                best_non_channel_idx = best_idx
                                best_non_channel_idx[best_idx==channel] = sorted_idx[:,:,1][best_idx==channel]
                                
                                channel_score = occ_all_tissue[..., channel]
                                best_non_channel_score = occ_all_tissue.gather(-1, best_non_channel_idx.unsqueeze(-1)).squeeze(-1)
                                
                                assert (best_non_channel_idx==channel).sum() == 0
                                occ_tissue = channel_score / (channel_score + best_non_channel_score)
                                occ = occ_tissue.cpu().squeeze(0)

                            else:
                                occ = torch.sigmoid((pred.cpu().squeeze(0)))
                                
                            # if bound_by_smpl:
                            if False:
                                # import ipdb; ipdb.set_trace()
                                from leap.tools.libmesh import \
                                    check_mesh_contains

                                # Add normal offset to mesh 
                                # is_inside = check_mesh_contains(mesh, pts.cpu().numpy()[0]).astype(np.float32)
                                mesh.vertices = mesh.vertices + mesh.vertex_normals * 0.01
                                is_inside = check_mesh_contains(mesh, pts.cpu().numpy()[0]).astype(np.float32)
                                
                                
                                occ[is_inside==0] = 0
                                
                            # if self.train_cfg['filter_outside']:
                            if False:
                                # import ipdb; ipdb.set_trace()
                                from leap.tools.libmesh import \
                                    check_mesh_contains
                                is_inside = check_mesh_contains(mesh, pts.cpu().numpy()[0]).astype(np.float32)
                                # occ = torch.FloatTensor(is_inside).to(device='cpu') * occ
                                
                                smpl_score = torch.sigmoid(output['smpl_occ'].cpu().squeeze(0))
                                # occ[is_inside==0] = smpl_score[is_inside==0]
                                import ipdb; ipdb.set_trace()
                                # occ[is_inside==0] = torch.min(smpl_score, occ)[is_inside==0]
                                
                                no_score = occ_all_tissue[..., 0]
                                C = occ_all_tissue.shape[-1]
                                corr_score = occ + no_score/(C-1) - (1-smpl_score/(C-1))
                                
                            occ_hats.append(occ)
                    # print('Done.')
                    
                    values = torch.cat(occ_hats, dim=0).numpy().astype(np.float64)
                    # sample points again
                    mesh_extractor.update(points, values)
                    points = mesh_extractor.query()

                
                value_grid_list.append(mesh_extractor.to_dense())
            value_grid = np.stack(value_grid_list)
            grid_res = mesh_extractor.resolution
        else:
            raise NotImplementedError
            grid3d = self.create_meshgrid3d(grid_res, grid_res, grid_res)  # range [0,G]
            grid3d = scale*(grid3d/grid_res - 0.5).reshape(1, -1, 3)  # (1,D*H*W,3) in range [-0.5, +0.5]*scale
            for grid_queries in torch.split(grid3d, max_queries//B, dim=1):
                pts = grid_queries.expand(B, -1, -1).to(device)*gt_scale_gpu.unsqueeze(1) + gt_center_gpu.unsqueeze(1)
                pred = self.query(pts.to(device=device), b_smpl_output_list)
                occ_tissue = act(F.softmax(pred, dim=-1))
                occ_tissue = occ_tissue[..., channel]
                
                occ_list.append(occ_tissue.cpu())  # B,N
            value_grid = torch.cat(occ_list, dim=1).reshape(B, grid_res, grid_res, grid_res).numpy()  # B,D,H,W

        # extract meshes
        mesh_list = []
        
        exception_error = ''
        for b_ind in range(B):
            try:
                verts, faces, normals, values = measure.marching_cubes(volume=value_grid[b_ind], gradient_direction='ascent', level=level)
            except Exception as e:
                exception_error = e
                print(f'Marching cubes failed for one of the batch item, no mesh will be returned: {exception_error}')
                return []

            # vertices to world space
            verts = scale*(verts/(grid_res-1) - 0.5)
            verts = verts*gt_scale[b_ind].item() + gt_center[b_ind].cpu().numpy()
            
            if template == True:
                # color verts 
                cond = cond_create(betas=smpl_output.betas[0].unsqueeze(0))
                verts_batched = torch.tensor(verts).to(smpl_output.betas.device).unsqueeze(0)
                if color_mode == 'compression' and self.train_cfg['compressor']:
                    
                    # Marching cube conmpression
                    comp = self.deformer.compressor(verts_batched, {'betas': cond['betas']*0})[0].cpu().numpy()
                
                    # # Template compression
                    # comp = self.deformer.compressor(smpl_output.vertices, {'betas': cond['betas']*0})[0].cpu().numpy()
                    # cols = np.abs(comp) / np.abs(comp).max()
                    # mesh = trimesh.Trimesh(smpl_output.vertices[0].cpu().numpy(), self.smpl.faces, vertex_colors=cols, process=False)
                    # mesh.export('/tmp/comp_mesh_template.ply')   
                    
                    # cols = comp / torch.linalg.norm(comp, axis=-1).unsqueeze(-1)
                    cols = np.abs(comp) / np.abs(comp).max()
                    mesh = trimesh.Trimesh(verts, faces, vertex_colors=cols, process=False)
                    mesh.export('/tmp/comp_mesh_mc.ply')    
                    
                    
                else:
                    w_pd = self.deformer.query_weights(verts_batched, {'latent': cond['lbs'], 'betas': cond['betas']*0})
                    cols = weights2colors(w_pd[0].cpu().numpy())
                    mesh = trimesh.Trimesh(verts, faces, vertex_colors=cols, process=False)
                    mesh.export('/tmp/lbs_mesh.ply')

                
            else:
                mesh = trimesh.Trimesh(verts, faces, process=False)
            # color meshes
            # vertex_colors = self.color_points(torch.from_numpy(verts).reshape(1, -1, 3).to(device), b_smpl_output_list[b_ind], max_queries)[0]
            mesh_list.append(mesh)
        return mesh_list

    @torch.no_grad()
    def color_points(self, pts, smpl_output=None, max_queries=100000):
        part_colors = self.get_part_colors()
        K = self.smpl.num_parts
        B = pts.shape[0]
        part_colors = part_colors[:K+1]
        part_colors[-1, :] = 0  # set bg to black

        pts_colors = []
        for _v in torch.split(pts, max_queries//B, dim=1):
            output = self.query(_v, smpl_output, ret_intermediate=True, eval_mode=True)
            part_pred = output['pred_occ'][1]
            label = part_pred['part_occupancy'].argmax(dim=1)  # B,K,V -> B,V
            label[part_pred['all_out']] = K  # all mlps say outside
            inds = label.reshape(-1).cpu().numpy()
            pts_colors.append(part_colors[inds].reshape((B, -1, 3)))

        pts_colors = np.concatenate(pts_colors, 1)
        return pts_colors

    @staticmethod
    def create_meshgrid3d(
        depth: int,
        height: int,
        width: int,
        device=torch.device('cpu'),
        dtype=torch.float32,
    ) -> torch.Tensor:
        """ Generate a coordinate grid in range [-0.5, 0.5].

        Args:
            depth (int): grid dim
            height (int): grid dim
            width (int): grid dim
        Return:
            grid tensor with shape :math:`(1, D, H, W, 3)`.
        """
        xs = torch.linspace(0, width, width, device=device, dtype=dtype)
        ys = torch.linspace(0, height, height, device=device, dtype=dtype)
        zs = torch.linspace(0, depth, depth, device=device, dtype=dtype)
        return torch.stack(torch.meshgrid([xs, ys, zs]), dim=-1).unsqueeze(0)  # 1xDxHxWx3


    @staticmethod
    def get_part_colors():
        return np.array([
            [ 8.94117647e-01,  1.01960784e-01,  1.09803922e-01],
            [ 2.15686275e-01,  4.94117647e-01,  7.21568627e-01],
            [ 3.01960784e-01,  6.86274510e-01,  2.90196078e-01],
            [ 5.96078431e-01,  3.05882353e-01,  6.39215686e-01],
            [ 1.00000000e+00,  4.98039216e-01,  3.58602037e-16],
            [ 1.00000000e+00,  1.00000000e+00,  2.00000000e-01],
            [ 6.50980392e-01,  3.37254902e-01,  1.56862745e-01],
            [ 9.68627451e-01,  5.05882353e-01,  7.49019608e-01],
            [ 6.00000000e-01,  6.00000000e-01,  6.00000000e-01],
            [ 1.00000000e+00,  9.17647059e-01,  8.47058824e-01],
            [ 4.94117647e-01, -3.58602037e-16,  1.84313725e-01],
            [ 7.92156863e-01,  7.29411765e-01,  3.72549020e-01],
            [ 5.25490196e-01,  4.78431373e-01,  3.58602037e-16],
            [ 9.29411765e-01,  6.50980392e-01,  5.76470588e-01],
            [ 5.05882353e-01,  4.03921569e-01,  4.15686275e-01],
            [ 7.80392157e-01,  3.56862745e-01,  4.43137255e-01],
            [ 6.86274510e-01,  5.37254902e-01,  3.52941176e-01],
            [ 6.27450980e-01,  8.23529412e-02,  0.00000000e+00],
            [ 1.00000000e+00,  4.35294118e-01,  3.80392157e-01],
            [ 7.37254902e-01, -7.17204074e-16,  2.82352941e-01],
            [ 1.00000000e+00,  9.09803922e-01,  5.52941176e-01],
            [ 1.00000000e+00,  7.84313725e-02,  4.15686275e-01],
            [ 7.72549020e-01,  7.52941176e-01,  6.66666667e-01],
            [ 5.68627451e-01,  2.62745098e-01,  2.98039216e-01],
            [ 0,  0,  0],
        ])
