import argparse
import os
import time
import numpy as np

from hit.utils.metrics import mri_data_to_percentage
from hit.training.dataloader_mri import MRIDataset
import hit.hit_config as cg

if __name__ == '__main__':
    # parse an index value from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--index', type=int, default=[0], nargs='+', help='Index of the subjects to visualize')
    parser.add_argument('-t','--tissues', type=str, choices=['LT', 'AT'])
    parser.add_argument('-s','--split', type=str, choices=['train', 'test', 'val'], default='train')
    parser.add_argument('-D','--display', action='store_true')
    parser.add_argument('-g', '--gender', type=str, choices=['female', 'male'], default='female')
    parser.add_argument('-c', '--can_only', action='store_true')
    parser.add_argument('--all', action='store_true', help='Load the entire dataset. Otherwise, only the index i is loaded.')
    parser.add_argument('--time', action='store_true', help='Time the loading of the dataset.')
    parser.add_argument('--trans', action='store_true', help='Print the translation of the root.')
    parser.add_argument('--recache', action='store_true', help='Recache the dataset.')
    args = parser.parse_args()

    args = parser.parse_args()

    config_file = os.path.join(cg.package_directory, 'configs/config.yaml')

    # load default config
    with open(config_file, 'r') as f:
        from hydra import compose, initialize
        with initialize(version_base=None, config_path="../hit/configs"):
            cfg = compose(config_name="config")
        
    smpl_cfg, data_cfg, train_cfg = cfg['smpl_cfg'], cfg['data_cfg'], cfg['train_cfg']
    
    # Update the config
    smpl_cfg['gender'] = args.gender
    
    if args.recache:
        for split in ['test', 'val', 'train']:
            data_cfg['force_recache'] = True
            args.all = True
            dl = MRIDataset.from_config(smpl_cfg, data_cfg, train_cfg, split=split)
            print(f'Split {split} recached for gender {args.gender}')

    # Show a data sample
    t1 = time.perf_counter()
    if args.all:
        dl = MRIDataset.from_config(smpl_cfg, data_cfg, train_cfg, split=args.split)
    else:
        dl = MRIDataset.from_config(smpl_cfg, data_cfg, train_cfg, split=args.split, filter_indices= args.index)
    t2 = time.perf_counter()
    print(f'Time to load dataset of one subject: {t2-t1:.2f}s')

    
    # Check the values of the translation
    # dl = MRIDataset.from_config(smpl_cfg, data_cfg, train_cfg, split=args.split)
    # all_t = []
    # for i in range(len(dl)):
    #     # import ipdb; ipdb.set_trace()
    #     t = dl[i]['root_trans']
    #     all_t.append(t)
    #     print(f'Index {i}: trans = {t}')
    # t_mean = sum(all_t)/len(all_t)
    # print(f'Average t: {sum(all_t)/len(all_t)}')
    # print('Variance of t along x, y, z:')
    # for i, axis in enumerate(['x', 'y', 'z']):
    #     print(f'{axis}: {torch.var(torch.Tensor([t[i] for t in all_t]))}')
    # diff = torch.vstack([abs(t-t_mean) for t in all_t])
    # max_indices = diff.argmax(axis=0)
    # print(f'Max diff to mean: {diff[max_indices]*100} cm')
    # subj = [dl[k]['seq_names'] for k in max_indices]
    # print(f'Max diff to mean subjects: {subj}')
    
    # dl = MRIDataset.from_config(smpl_cfg, data_cfg, train_cfg, split=args.split)
    
    # Time loading the dataset
    if args.time:
        dt_list = []
        for i in range(len(dl)):
            t1 = time.perf_counter()
            d = dl[i]
            t2 = time.perf_counter()
            dt_list.append(t2-t1)
        print(f'Average time to get item: {sum(dt_list)/len(dt_list):.2f}s') 
        print(f'Total time to get all items: {sum(dt_list):.2f}s')  

    for i in range(len(args.index)):
        # time the operation precisely
        t0 = time.perf_counter()
        data = dl[i]
        t1 = time.perf_counter()
        # format the results nicely
        print(f'Time to load sample {i}: {t1-t0:0.4f} seconds')
        
        if args.trans:
            t = dl[i]['root_trans']
            print(f'translation: {t}')
            
        if args.display:
            dl.display_sample(i, can_only=args.can_only)
            
    # Compute average percentage of each tissue - deprecated
    # percentage_list = []
    # for i in range(len(dl)):
    #     labels = ['NO', 'LT', 'AT', 'BT']
    #     data = dl[i]
    #     perc = mri_data_to_percentage(data, labels, mask_with_inside=True)
    #     percentage_list.append(perc)
    # import ipdb; ipdb.set_trace()
    # percentage_array = np.array(percentage_list)
    # mean_perc = np.mean(percentage_array, axis=0)
    # mean_perc_str = [f'{perc:.2f}' for perc in mean_perc]
    # print(f'Mean percentage of each tissue: {mean_perc_str}')
        

for k, v in data.items():
    shape = None
    value = None
    try:
        shape = v.shape
    except:
        pass
    
    if shape is None:
        value = v
    print(f'{k}: {type(v)} {shape} {value}')
