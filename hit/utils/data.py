import gzip
import pickle
import pickle as pkl

import numpy as np
import torch


def build_smpl_torch_params(param_dict, device, model_type='smpl'):
    def to_tensor(x, device):
        if torch.is_tensor(x):
            return x.to(device=device)
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x.astype(np.float32)).to(device=device)
        return x
    
    torch_param = {key: to_tensor(val, device) for key, val in param_dict.items()}
    
    # for visualization -- disable global rotation and translation as they do not influence self-penetration
    for zero_key in ['global_orient', 'transl', 'left_hand_pose', 'right_hand_pose']:
        if zero_key in torch_param:
            torch_param[zero_key][:] = 0.0

    if model_type == 'smpl':
        smpl_body_pose = torch.zeros((1, 69), dtype=torch.float, device=device)
        if 'body_pose' in torch_param:
            smpl_body_pose[:, :63] = torch_param['body_pose']
            torch_param['body_pose'] = smpl_body_pose
        else: 
            # import ipdb; ipdb.set_trace()
            torch_param['betas'] = torch_param['betas'][None, :10]
            torch_param['body_pose']= torch_param['pose'][None,3:]
            torch_param['global_orient'] = torch_param['pose'][None, 0:3]
            torch_param['transl'] = torch_param['trans'][None, 0:3]
            
    for key in ['betas', 'body_pose', 'global_orient', 'transl']:
        assert key in torch_param, f'{key} not in torch_param'
        # if not key in torch_param:
        #     import ipdb; ipdb.set_trace()

    return torch_param
   

def load_smpl_data(pkl_path, device):
    with open(pkl_path, 'rb') as f:
        param_dict = pickle.load(f)
        
    return build_smpl_torch_params(param_dict, device)
 
 
def load_smpl_data_from_dataset(dataroot, gender, split, index, device):
    from hit.training.dataloader_mri import get_split_files
    path = get_split_files(dataroot, gender, split)[index]
    # import ipdb; ipdb.set_trace()
    if path.endswith('.pkl'):
        data = pkl.load(open(path, 'rb'))
    elif path.endswith('.gz'):
        with gzip.open(path, "rb") as f:
            data = pickle.load(f)
    data_smpl = data['smpl_dict']
    
    return build_smpl_torch_params(data_smpl, device, model_type='smpl')    