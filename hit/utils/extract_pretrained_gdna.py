""" Extract he pretraining of specific modules of gDNA to use it in another model"""
import os
import torch
import hit.hit_config as cg

def get_state_dict(to_extract = 'deformer', source=None):
    
    assert source is not None, 'Please specify source'
    assert source in ['gdna', 'pretrained_male', 'pretrained_female', 'pretrained_compressor_male', 'pretrained_compressor_female'], f'Unknown source {source}'

    # import ipdb; ipdb.set_trace()
    root = os.getcwd() #'/home/mkeller2/data2/Code/hit/learning/pretrained/'
    if source == 'gdna':
        path = os.path.join(root, '../../Data/gDNA_pretrained/pretrained/renderpeople/last.ckpt')
    # elif source == 'pretrained':
    #     # path = root + '../learning/pretrained/beta_disp_pretrainedG_male/ckpts/model-epoch=5399-val_accuracy=0.951843.ckpt'
    #     path = os.path.join(root, '../pretrained/beta_disp_pretrained_G_male_v2/last.ckpt')
    # elif source == 'pretrained_3C':
    #     path = os.path.join('/is/cluster/work/mkeller2/Data/hit/trained_models/pretrain_fromscratch_wcan1e6/ckpts/last.ckpt')
    elif source == 'pretrained_male':
        # path = os.path.join(cg.trained_models_folder,  'ptest6_256/ckpts/last.ckpt')
        path = os.path.join(cg.trained_models_folder,  'ptest7_lbs_fem_geoinit_v2/ckpts/last.ckpt')
    elif source == 'pretrained_female':
        # path = os.path.join(cg.trained_models_folder, 'pfm4/ckpts/last.ckpt')
        path = os.path.join(cg.trained_models_folder, 'ptest7_lbs_geoinit/ckpts/last.ckpt')
    elif source == 'pretrained_compressor_male':
        # path = os.path.join('/is/cluster/work/mkeller2/Data/hit/trained_models/12_compression_test_perbeta_B/ckpts/last.ckpt') # posed
        # path = os.path.join('/is/cluster/work/mkeller2/Data/hit/trained_models/12_compression_testBD/ckpts/last.ckpt') #local
        # path = os.path.join('/is/cluster/work/mkeller2/Data/hit/trained_models/v7_compression_posedB/ckpts/last.ckpt') #local
        # path = os.path.join('/is/cluster/work/mkeller2/Data/hit/trained_models/v7_compression_posed_shapespace/ckpts/last.ckpt') #local
        # path = os.path.join('/is/cluster/work/mkeller2/Data/hit/trained_models/v7_compression_posedC/ckpts/last.ckpt') #global shapespace
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