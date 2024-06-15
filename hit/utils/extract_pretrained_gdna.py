""" Extract he pretraining of specific modules of gDNA to use it in another model"""
import os
import torch
import hit.hit_config as cg

def get_state_dict(to_extract = 'deformer', source=None):
    
    assert source is not None, 'Please specify source'
    assert source in ['gdna', 'pretrained_male', 'pretrained_female', 'pretrained_compressor_male', 'pretrained_compressor_female'], f'Unknown source {source}'

    root = os.getcwd() 
    if source == 'gdna':
        path = os.path.join(root, '../../Data/gDNA_pretrained/pretrained/renderpeople/last.ckpt')
    elif source == 'pretrained_male':
        path = cg.pretrained_male_smpl
    elif source == 'pretrained_female':
        path = cg.pretrained_female_smpl
    elif source == 'pretrained_compressor_male':
        path = os.path.join(cg.trained_models_folder, 'cf/ckpts/last.ckpt') #global shapespace
    elif source == 'pretrained_compressor_female':
        path = os.path.join(cg.trained_models_folder, 'cfm/ckpts/last.ckpt')
        

        


    else:
        raise NotImplementedError

    checkpoint = torch.load(path)
    checkpoint['state_dict'].keys()
    
    state_dict = {}

    for k, v in checkpoint['state_dict'].items():
        # import pdb; pdb.set_trace()
        if k.startswith(to_extract):
            k_new = k.replace(to_extract+'.', '')
            # print(k, k_new)
            state_dict[k_new] = v          

    return state_dict

if __name__ == '__main__':
    get_state_dict()
    get_state_dict('generator')