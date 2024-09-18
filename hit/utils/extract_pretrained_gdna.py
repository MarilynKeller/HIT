""" Extract he pretraining of specific modules of gDNA to use it in another model"""
import os
import torch
import hit.hit_config as cg

def get_state_dict(to_extract = 'deformer', source=None):
    
    assert source is not None, 'Please specify source'
    assert source in ['pretrained_male', 'pretrained_female'], f'Unknown source {source}'

    root = os.getcwd() #'/home/mkeller2/data2/Code/hit/learning/pretrained/'
    if source == 'pretrained_male':
        path = cg.pretrained_male_smpl
    elif source == 'pretrained_female':
        path = cg.pretrained_female_smpl
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