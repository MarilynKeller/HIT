
import collections
import glob
import gzip
import os
import pickle
import pickle as pkl
import sys
import time

import numpy as np
import pyrender
import torch
import torch.nn.functional as F
import tqdm
import trimesh
from datasets import load_dataset

from hit.model.mysmpl import MySmpl
from hit.training.mri_sampling_utils import (compute_discrete_sdf_gradient, sample_mri_pts, vis_create_pc)
from hit.utils.smpl_utils import get_skinning_weights
import hit.hit_config as cg

# all the MRI will be padded with zero to this size
MRI_SHAPE = (256, 256, 128) #(512,512,128) for mpi
BODY_ONLY = False # When the whole MRI is sampled, only sample voxels in the body mask and around (dilatation is used)
SMPL_KEYS = ['betas', 'body_pose', 'global_orient', 'transl']

if BODY_ONLY:
    raise NotImplementedError(f'TODO: implement the composition computation ')

def pad_mri_z_vector(mri_vector):
    # pad a vecor to size MRI_SHAPE[2] and repeat the last slice value
    assert mri_vector.shape[0] < MRI_SHAPE[2], f'Input vector is larger than the target size MRI_SHAPE[2]. Make MRI_SHAPE[2] bigger to fix.'
    arr = mri_vector
    z_pad = MRI_SHAPE[2] - arr.shape[0]
    padded = F.pad(input=arr, pad=(0, 0, 0, z_pad), mode='constant', value=0)
    padded[arr.shape[1]:, :] = arr[-1, :]
    return padded

def sample_uniform_from_max(shape, max_val, padding):
    # Sample uniformly
    max_val = max_val + padding
    return torch.rand(shape).to(max_val.device) * max_val * 2 - max_val 

def pad_mri(mri_data):
    
    size_err = f'Input vector of size{mri_data.shape} is larger than the target size MRI_SHAPE {MRI_SHAPE}. Make MRI_SHAPE bigger to fix.'
    mri_data = mri_data[:MRI_SHAPE[0], :MRI_SHAPE[1], :MRI_SHAPE[2]]

    arr = mri_data
    x_pad = MRI_SHAPE[0] - arr.shape[0]
    y_pad = MRI_SHAPE[1] - arr.shape[1]
    z_pad = MRI_SHAPE[2] - arr.shape[2]

    padded = F.pad(input=arr, pad=(0, z_pad, 0, y_pad, 0, x_pad), mode='constant', value=0)
    return padded

def pad_gradient(gradient_data):
    '''Pad a  CxWxDxHx3x3 array to the MRI_SHAPE'''
    size_err = f'Input vector of size{gradient_data.shape} is larger than the target size MRI_SHAPE {MRI_SHAPE}. Make MRI_SHAPE bigger to fix.'
    
    # If the mri is just one slice too big, we remove the last slice. We don't want to double the MRI_SHAPE array size just for that.
    margin = 1 
    assert gradient_data.shape[0] < MRI_SHAPE[0]+margin, size_err
    assert gradient_data.shape[1] < MRI_SHAPE[1]+margin, size_err
    assert gradient_data.shape[2] < MRI_SHAPE[2]+margin, size_err  
    
    gradient_data = gradient_data[:, :MRI_SHAPE[0], :MRI_SHAPE[1], :MRI_SHAPE[2], :]

    arr = gradient_data
    x_pad = MRI_SHAPE[0] - arr.shape[0]
    y_pad = MRI_SHAPE[1] - arr.shape[1]
    z_pad = MRI_SHAPE[2] - arr.shape[2]

    padded = F.pad(input=arr, pad=(0,0, 0,0, 0, z_pad, 0, y_pad, 0, x_pad), mode='constant', value=0)
    return padded


def list_preprocessed_files(data_root, genders: list):
    """ List all preprocessed files in the data root. Return only the genders listed in genders. """
    
    assert os.path.isdir(data_root), f"Data root {data_root} does not exist."
    for g in genders:
        assert g in ['male', 'female']
    assert genders, f'No gender listed to fetch the dataset'
    paths = []
    labels = []
    li = 0
    for dataset in ['cds']:
        for gender in genders:
            folder = os.path.join(data_root, dataset, gender)
            os.path.isdir(folder), f"Folder {folder} does not exist."
            file_list = os.listdir(folder)
            path_list = [os.path.join(folder, filename) for filename in file_list]
            paths.extend(path_list)
            labels.extend([li]*len(path_list))
            li += 1
            
    paths.sort()
    return paths

def get_split_files(data_root, gender, split):
    paths = glob.glob(os.path.join(data_root, gender, split, '*.gz'))
    return paths

def _get_split_files(data_root, gender, split):
    data_version = cg.data_version
    print(f'\n Loading splits for data version {data_version} for gender {gender} \n')
    if data_version == 'v4':
        split_file = os.path.join('./splits', f'split_mri_{gender}_{split}.txt')
    else:
        split_file = os.path.join(f'./splits{data_version}', f'split_mri_{gender}_{split}.txt')

    assert os.path.exists(split_file), f'Split file {split_file} does not exist, you can create it with generate_splits.py'

    paths = []
    with open(split_file, 'r') as f:
        for line in f:
            # print(data_root, line.strip())
            paths.append(os.path.join(data_root, line.strip()))

    return paths

def print_splits_files(data_root):
    for gender in ['female', 'male']:
        for split in ['train', 'test', 'val']:
            paths = get_split_files(data_root, gender, split)
            print(f'Number of {split} samples for {gender}: {len(paths)}')
            for path in paths:
                print('\t', path)
  
# @varora
# function to normalize the mri values by min max
def process_mri_values(mri_values: np.ndarray, normalize=False):
    if normalize:
        # min-max normalization
        mri_values = (mri_values - np.min(mri_values)) / (np.max(mri_values) - np.min(mri_values))
    mri_values = mri_values.astype(np.float32)
    return mri_values


class MRIDataset(torch.utils.data.Dataset):

    @torch.no_grad()
    def __init__(self, smpl_cfg, data_cfg, train_cfg, smpl_data, split):
        super().__init__()
        self.smpl_cfg = smpl_cfg
        self.data_cfg = data_cfg
        self.train_cfg = train_cfg

        self.smpl_data = smpl_data
        self.split = split  # changes the batch nb of vertices

        self.data_keys = list(smpl_data.keys())
        self.smpl = MySmpl(model_path=cg.smplx_models_path, gender=smpl_cfg['gender'])

        self.faces = self.smpl.faces.copy()
        
        self.synthetic = data_cfg.synthetic
        
        self.bbox_padding = 1.125

        if not self.synthetic:
            # find index of a subject with fat for overfitting if needed
            self.lowest_b1_subject_idx = self.find_lowest_b1()
            name = self.smpl_data['seq_names'][self.lowest_b1_subject_idx]
            print(f'Lowest b1 subject idx: {self.lowest_b1_subject_idx}, name : {name}')

            self.highest_b1_subject_idx = self.find_highest_b1()
            name = self.smpl_data['seq_names'][self.highest_b1_subject_idx]
            print(f'Highest b1 subject idx: {self.highest_b1_subject_idx}, name : {name}')

            self.__len__()

            self.can_points_dictionary = self.sample_can_space()  # We do this once for all the dataset
            self.can_hands_dictionary = self.sample_can_hands()  # We do this once for all the dataset

    @classmethod
    @torch.no_grad()
    def from_config(cls, smpl_cfg, data_cfg, train_cfg, split='train', filter_indices=None, force_recache=False):
        """Load the dataset from the packaged data.
        param smpl_cfg: SMPL configuration
        param data_cfg: Data configuration
        param train_cfg: Training configuration
        param split: Which split to load
        param filter_indices (str of integers): If not None, only load the subjects with those indices
        """
        # load data
        gender = smpl_cfg['gender']

        # Paths for caching the created data dictionary data_stacked
        if data_cfg['use_gzip']:
            cache_name = f'{gender}_{split}.gz'
            open_fct = gzip.open
        else:
            cache_name = f'{gender}_{split}.pkl'
            open_fct = open
        cache_folder = cg.packaged_data_folder
        cache_path = os.path.join(cache_folder, cache_name)

        if train_cfg.mri_values == True:
            varying_size_keys = ['mri_points', 'mri_values', 'mri_coords', 'mri_occ', 'skinning_weights',
                                 'part_id',
                                 'body_mask', 'mri_data_packed']
        else:
            varying_size_keys = ['mri_points', 'mri_coords', 'mri_occ', 'skinning_weights', 'part_id',
                                 'body_mask',
                                 'mri_data_packed']

        def hf_to_hit(hf_data_sample):
            """
            convert hf data to hit data
            :param hf_data_sample:
            :return: hit_data_sample
            """
            hit_data_sample = {}
            mri_seg_dict = {}
            smpl_dict = {}
            # gender
            hit_data_sample['gender'] = hf_data_sample['gender']
            # mri_seg
            hit_data_sample['mri_seg'] = np.array(hf_data_sample['mri_seg']).transpose(1,2,0)
            # mri_labels
            hit_data_sample['mri_labels'] = hf_data_sample['mri_labels']
            # mri_seg_dict
            mri_seg_dict['BODY'] = np.array(hf_data_sample['body_mask'])
            hit_data_sample['mri_seg_dict'] = mri_seg_dict
            # resolution
            hit_data_sample['resolution'] = np.array(hf_data_sample['resolution'])
            # center
            hit_data_sample['center'] = np.array(hf_data_sample['center'])
            # smpl_dict
            smpl_dict['verts'] = np.array(hf_data_sample['smpl_dict']['verts'])
            smpl_dict['verts_free'] = np.array(hf_data_sample['smpl_dict']['verts_free'])
            smpl_dict['faces'] = np.array(hf_data_sample['smpl_dict']['faces'])
            smpl_dict['pose'] = np.array(hf_data_sample['smpl_dict']['pose'])
            smpl_dict['betas'] = np.array(hf_data_sample['smpl_dict']['betas'])
            smpl_dict['trans'] = np.array(hf_data_sample['smpl_dict']['trans'])
            hit_data_sample['smpl_dict'] = smpl_dict
            # dataset_name
            hit_data_sample['dataset_name'] = hf_data_sample['dataset_name']
            hit_data_sample['subject_ID'] = hf_data_sample['subject_ID']
            return hit_data_sample

        def cache_condition(train_cfg):
            # return True
            if data_cfg.synthetic is True:
                return False, 'synthetic data'
            if data_cfg.subjects != 'all':
                return False, 'not subjects all'
            if filter_indices is not None:
                return False, 'has filter_indices'
            return True, ''

        def get_deep_size(obj, seen=None):
            """Recursively find the memory footprint of a Python object."""
            if seen is None:
                seen = set()
            obj_id = id(obj)
            if obj_id in seen:
                return 0
            seen.add(obj_id)
            size = sys.getsizeof(obj)
            if isinstance(obj, dict):
                size += sum([get_deep_size(k, seen) for k in obj.keys()])
                size += sum([get_deep_size(v, seen) for v in obj.values()])
            elif isinstance(obj, (list, tuple, set)):
                size += sum([get_deep_size(i, seen) for i in obj])
            elif isinstance(obj, np.ndarray):
                size += obj.nbytes
            elif isinstance(obj, torch.Tensor):
                size += obj.element_size() * obj.numel()
            elif hasattr(obj, '__dict__'):
                size += get_deep_size(obj.__dict__, seen)
            elif hasattr(obj, '__slots__'):
                size += sum(get_deep_size(getattr(obj, slot), seen) for slot in obj.__slots__)
            return size

        def chunk_dataset_stacked(data_dict, num_chunks):
            chunked_dicts = [{} for _ in range(num_chunks)]
            chunk_size = len(next(iter(data_dict.values()))) // num_chunks
            last_chunk_size = len(next(iter(data_dict.values()))) % num_chunks
            last_chunk_dict = {}
            for key, value_list in data_dict.items():
                for i in range(num_chunks):
                    chunked_dicts[i][key] = value_list[i * chunk_size:(i + 1) * chunk_size]
                last_chunk_dict[key] = value_list[-last_chunk_size:]
            chunked_dicts.append(last_chunk_dict)
            return chunked_dicts

        def read_and_stack_chunks(chunk_files, varying_size_keys):
            # init
            with open(chunk_files[0], 'rb') as f:
                chunk_data = pickle.load(f)
            dataset_list = {k: [] for k in chunk_data.keys()}

            for chunk_file in tqdm.tqdm(chunk_files):
                with open(chunk_file, 'rb') as f:
                    chunk_data = pickle.load(f)
                for key, value_list in chunk_data.items():
                    for i in value_list:
                        dataset_list[key].append(i)
            # Stack the lists into tensors
            data_stacked = {}
            for key, val in dataset_list.items():
                if isinstance(val[0], torch.Tensor) and key not in varying_size_keys:
                    data_stacked[key] = torch.stack(val, dim=0)
                else:
                    data_stacked[key] = val
            return data_stacked

        def delete_old_cache_files(directory, extension='*.gz'):
            """Delete old cache files with the specified extension from the directory."""
            # Construct the full path with the extension
            search_path = os.path.join(directory, extension)
            # Use glob to find all files matching the pattern
            files = glob.glob(search_path)
            # Iterate over the files and remove them
            for file_path in files:
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except OSError as e:
                    print(f"Error deleting {file_path}: {e}")

        do_cache, reason = cache_condition(train_cfg)
        do_cache = do_cache and not force_recache

        n_chunks = 10
        if split == 'train':
            n_chunks = 20

        cache_dir = os.path.join(os.path.dirname(cache_path), 'dataset_cache', f'{gender}', f'{split}')
        os.makedirs(cache_dir, exist_ok=True)

        def check_cache(n_chunks, path):
            cache_files = glob.glob(os.path.join(path, '*.gz'))
            n_cache_files = len(cache_files)
            if n_cache_files == n_chunks or n_cache_files == n_chunks + 1:
                return True
            else:
                return False

        cache_exists = check_cache(n_chunks=n_chunks,  path=cache_dir)

        if not cache_exists:
            do_cache = True
            reason = 'Chunks dont exist or number of chunk mismatch. Fresh cache will be created.'
            delete_old_cache_files(cache_dir)

        if force_recache:
            reason = 'force recache set to True in the dataloader'

        data_root = cg.packaged_data_folder

        paths = get_split_files(data_root, gender, split)
        hf = data_cfg.huggingface
        if hf:
            print(f'Dowlnoading dataset {split} from huggingface.')
            paths = load_dataset("varora/hit", name=gender, split=split)
        if do_cache:
            if cache_exists and not data_cfg['force_recache']:  # and False:
                print(f'Loading cached dataset chunks for {gender} {split} from {cache_dir}')
                cache_files = glob.glob(os.path.join(cache_dir, '*.gz'))
                t1 = time.perf_counter()
                data_stacked = read_and_stack_chunks(cache_files, varying_size_keys)
                t2 = time.perf_counter()
                # todo Check the shape of stached mri data here
                nb_subjects = data_stacked['betas'].shape[0]
                print(f'Loaded {nb_subjects} subjects data from cache in {t2 - t1:.2f}s')
                return MRIDataset(smpl_cfg, data_cfg, train_cfg, data_stacked, split)
            else:
                print(
                    f'No cached dataset {split} found at {cache_path}. A cache will be created when the dataset is done loading.')
        else:
            print(f'Not caching dataset {split} because {reason}')

        if data_cfg.subjects == 'first':
            paths = [paths[0]]

        if data_cfg.subjects == 'two':
            paths = [paths[0], paths[1]]

        if filter_indices is not None:
            paths = [paths[i] for i in filter_indices]

        if len(paths) < 5:
            print(f'Loading dataset {split} with {len(paths)} samples:')

        synthetic = data_cfg.synthetic

        if not synthetic:
            print(f'Loading dataset {split} with {len(paths)} samples')
        else:
            print(f'Loading one subject to create synthetic dataset.')

        smpl = MySmpl(model_path=cg.smplx_models_path, gender=smpl_cfg['gender'])
        num_body_joints = smpl.smpl.NUM_BODY_JOINTS

        dataset_list = collections.defaultdict(list)
        print('Loading dataset into memory')
        for pi, path in tqdm.tqdm(enumerate(paths)):
            subj_data = {}

            if hf:
                data = hf_to_hit(path)
            else:
                if path.endswith('.pkl'):
                    data = pkl.load(open(path, 'rb'))
                elif path.endswith('.gz'):
                    with gzip.open(path, "rb") as f:
                        data = pickle.load(f)
                assert data is not None, f'Could not load {path}'

            # Extract smpl data from the package
            gender = data['gender']
            betas = torch.Tensor(data['smpl_dict']['betas'][:smpl.nb_betas])
            pose = torch.Tensor(data['smpl_dict']['pose'])
            trans = torch.Tensor(data['smpl_dict']['trans'])
            verts_free = data['smpl_dict']['verts_free'] - data['smpl_dict']['trans']
            verts = data['smpl_dict']['verts'] - data['smpl_dict']['trans']

            assert len(pose.shape) == 1, f'Pose shape as an unexpected batch dimension {pose.shape}'

            # Fill the dataset fields
            subj_data['betas'] = betas
            subj_data['global_orient'] = pose[:3]
            subj_data['transl'] = torch.zeros_like(
                trans)  # The functions to transform MRI points to metric space already take into account the translation
            subj_data['root_trans'] = trans  # For unposing the mri
            subj_data['body_verts'] = verts
            subj_data['body_verts_free'] = verts_free
            subj_data['body_pose'] = pose[3:3 + num_body_joints * 3]
            # Unpose the hands
            subj_data['body_pose'][-6:] = 0

            if data_cfg.body_normals is True:
                body_normals = trimesh.Trimesh(vertices=verts_free, faces=data['smpl_dict']['faces'],
                                               process=False).vertex_normals
                body_normals.setflags(
                    write=True)  # the output of trimesh is read-only, make it writable to avoid warning from pytorch
                body_normals = torch.Tensor(body_normals)
                subj_data['body_normals'] = body_normals

            subj_data['global_orient_init'] = subj_data['global_orient']  # For back compatibility

            # Name of the subject
            if hf:
                seq_name = data['subject_ID']
            else:
                seq_name = os.path.splitext(os.path.basename(path))[0]
            subj_data['seq_names'] = seq_name

            # For backward compatibility with export_visuals function
            subj_data['frame_ids'] = f'{pi:06d}'

            # MRI data
            # print(data['mri_seg'].shape)
            for k in range(3):
                # The MRIs will be padded to size MRI_SHAPE for batch support, check that they are not too big
                assert data['mri_seg'].shape[k] <= MRI_SHAPE[
                    k], f"mri_seg shape {data['mri_seg'].shape} is larger than {MRI_SHAPE}. Update MRI_SHAPE file to support this MRI size."

            # We pad those so that they can be stacked per batch by the dataloader
            subj_data['mri_resolution'] = pad_mri_z_vector(torch.FloatTensor(data['resolution']))
            subj_data['mri_center'] = pad_mri_z_vector(torch.FloatTensor(data['center']))
            subj_data['mri_size'] = torch.IntTensor(data['mri_seg'].shape)

            # Merge VAT and LT
            mri_labels_dict = data['mri_labels']  # Dictionary which for each label gives the value in the mri_seg array
            mri_seg = data['mri_seg']
            mri_seg[mri_seg == mri_labels_dict['VAT']] = mri_labels_dict['LT']

            # Since we remove VAT, we need to update the labels. Bones was 4 and becomes 3
            if 'BONE' in mri_labels_dict:
                mri_seg[mri_seg == mri_labels_dict['BONE']] = 3  # Manually set this to 3
                if mri_seg.max() != 3:
                    print(f'Wrong max value for mri_seg: {mri_seg.max()}')
                assert mri_seg.max() == 3, f'Wrong max value for mri_seg: {mri_seg.max()}'
            else:
                assert mri_seg.max() == 2

            # Compute discrete sdf gradient
            if data_cfg.use_gradient:
                mri_gradient = compute_discrete_sdf_gradient(data['mri_seg'], data['resolution'])  # W, D, H, C, 3
                subj_data['mri_sdf_gradient'] = pad_gradient(torch.FloatTensor(mri_gradient))
                # dataset['sdf'].append(pad_gradient(torch.FloatTensor(mri_gradient)).unsqueeze(0))

            # Remove VAT from the mri_labels
            mri_labels = {}
            mri_labels.update({key: data['mri_labels'][key] for key in data['mri_labels'] if key != 'VAT'})

            if len(train_cfg.mri_labels) == 1:
                C = train_cfg.mri_labels[0]
                C_id = mri_labels[C]
                mri_seg = (mri_seg[:, :, C_id] == 1)

            # if data_cfg['use_bone'] == False:
            #     data['mri_seg'][data['mri_seg']==4]  0

            subj_data['mri_seg'] = pad_mri(torch.FloatTensor(data['mri_seg']))
            # subj_data['mri_seg_NO'] = pad_mri(torch.FloatTensor(data['mri_seg_dict']['NO']))
            subj_data['mri_seg_BODY'] = pad_mri(torch.FloatTensor(data['mri_seg_dict']['BODY']))

            # @varora
            if train_cfg.mri_values == True:
                # get normalized mri values
                mri_values_processed = process_mri_values(data['mri_values'], normalize=True)
                # pad the mri values
                subj_data['mri_values'] = pad_mri(torch.FloatTensor(mri_values_processed))

            # @varora
            # get the mri values for sampled points
            mri_points, mri_coords, gt_occ, body_mask, mri_values = sample_mri_pts(subj_data,
                                                                                   body_only=True,
                                                                                   dilate_body=data_cfg.get(
                                                                                       'dilate_body_mask', False),
                                                                                   use_mri_net=train_cfg.get(
                                                                                       'mri_values'))

            # TODO: add a flag to use the normals version
            skinning_weights, part_id = get_skinning_weights(mri_points, subj_data['body_verts_free'], smpl,
                                                             free_verts=None)

            # subj_data_cleaned = {}
            # subj_data_cleaned['mri_points'] = mri_points
            # subj_data_cleaned['mri_coords'] = mri_coords
            # subj_data_cleaned['mri_occ'] = gt_occ
            # subj_data_cleaned['skinning_weights'] = skinning_weights
            # subj_data_cleaned['part_id'] = part_id
            # subj_data_cleaned['body_mask'] = body_mask
            # [print(val.shape) for key, val in subj_data_cleaned.items()]

            # import ipdb; ipdb.set_trace()

            mri_data_topack = {}
            mri_data_topack['mri_points'] = mri_points  # (nb_mri_pts x 3)
            mri_data_topack['mri_coords'] = mri_coords  # ...
            mri_data_topack['mri_occ'] = gt_occ[..., None]
            mri_data_topack['part_id'] = part_id[..., None]
            mri_data_topack['body_mask'] = body_mask[..., None]
            mri_data_topack['skinning_weights'] = skinning_weights
            # @varora
            # add mri values to stack only if train_cfg['mri_values'] is Trueq
            if train_cfg.mri_values == True:
                print('Adding mri values to stack')
                mri_data_topack['mri_values'] = mri_values[..., None]

            mri_points_nb = mri_points.shape[0]  # around 1e6
            print(f'Number of MRI points: {mri_points_nb}')
            stacked_data = np.concatenate([val for key, val in mri_data_topack.items()], axis=1)  # (1217671, 33)
            # stacked_data = None

            subj_data_cleaned = {}
            subj_data_cleaned['mri_data_packed'] = stacked_data
            subj_data_cleaned['mri_data_shape0'] = mri_points_nb
            subj_data_cleaned['mri_data_shape1'] = [val.shape[0] for key, val in mri_data_topack.items()]
            # subj_data_cleaned['mri_data_keys'] = mri_data_topack.keys()

            subj_data_cleaned['betas'] = betas
            subj_data_cleaned['global_orient'] = pose[:3]
            subj_data_cleaned['transl'] = torch.zeros_like(
                trans)  # The functions to transform MRI points to metric space already take into account the translation
            subj_data_cleaned['body_verts'] = verts
            subj_data_cleaned['body_verts_free'] = verts_free
            subj_data_cleaned['body_pose'] = subj_data['body_pose']

            if split == 'test':
                subj_data_cleaned['root_trans'] = subj_data['root_trans']
                subj_data_cleaned['mri_resolution'] = subj_data['mri_resolution']
                subj_data_cleaned['mri_center'] = subj_data['mri_center']
                subj_data_cleaned['mri_size'] = subj_data['mri_size']
                subj_data_cleaned['mri_seg'] = subj_data['mri_seg']
                subj_data_cleaned['mri_seg_BODY'] = subj_data['mri_seg_BODY']

            subj_data_cleaned['global_orient_init'] = subj_data['global_orient']  # For back compatibility

            # Name of the subject

            subj_data_cleaned['seq_names'] = subj_data['seq_names']

            nb_add = 1
            if len(paths) < 4 and not data_cfg.synthetic:
                # In case there is only one subject, artificially increase the number of subjects
                if split == 'train':
                    nb_add = 128 // len(
                        paths)  # Add each subject that number of time to simulate a bigger dataset and have meaningfull epochs
                else:
                    nb_add = 12 // len(paths)

            for _ in range(nb_add):
                for key, val in subj_data_cleaned.items():
                    if isinstance(val, np.ndarray):
                        val = torch.FloatTensor(val)
                    dataset_list[key].append(val)

            if data_cfg.synthetic:
                # Load just one subject to know the sizes of the data
                break

        # import ipdb; ipdb.set_trace()
        # max_nb_mri_pts = np.max(dataset_list['mri_points_nb'])

        # Stack the lists into tensors
        data_stacked = {}
        for key, val in dataset_list.items():
            if isinstance(val[0], torch.Tensor) and key not in varying_size_keys:
                data_stacked[key] = torch.stack(val, dim=0)
            else:
                data_stacked[key] = val

        print(f"Size of stacked data for {gender} {split} = {get_deep_size(data_stacked)/1e9} GBs")

        if do_cache:
            print(f'Caching dataset {gender} {split} to {cache_dir} in {n_chunks} chunks')

            chunk_dicts_list = chunk_dataset_stacked(data_stacked, n_chunks)
            cache_paths = [ os.path.join(cache_dir, f"chunk_{i}.gz") for i in range(len(chunk_dicts_list)) ]

            #N_subjects_chunks = np.array([len(chunk_dicts_list[i]['mri_seg_BODY']) for i in range(len(chunk_dicts_list))]).sum()
            #N_subjects = len(next(iter(data_stacked.values())))
            #assert N_subjects == N_subjects_chunks, "Data truncated. Check chunking code."

            for idx, p in enumerate(cache_paths):
                with open_fct(p, 'wb') as f:
                    t1 = time.perf_counter()
                    pickle.dump(chunk_dicts_list[idx], f)
                    mem = get_deep_size(chunk_dicts_list[idx])
                    N_subjects = len(next(iter(chunk_dicts_list[idx].values())))
                    t2 = time.perf_counter()
                    print(f'Saving chunk-{idx}\n'
                          f'No of subjects: {N_subjects}\n'
                          f'Caching footprint: {(2*mem)/1e9} GBs\n'
                          f'Path: {p}\n'
                          f'Time taken: {t2 - t1:.2f}s\n\n')

        return MRIDataset(smpl_cfg, data_cfg, train_cfg, data_stacked, split)

    @torch.no_grad()
    def __getitem__(self, idx, return_smpl=False, get_whole_mri=False):

        t1 = time.perf_counter()
        if self.synthetic:
            return self._getitem_synthetic()

        # t1 = time.perf_counter()
        subj_data = {key: self.smpl_data[key][idx] for key in self.smpl_data.keys()}

        for key in ['betas', 'body_pose', 'global_orient', 'transl']:
            if not key in subj_data:
                raise ValueError(f'{key} not in subj_data')
            assert key in subj_data, f'{key} not in torch_param'

        if self.train_cfg['comp0_out']:
            mri_points = subj_data['mri_data_packed'][:, 0:3]
            body_mask = subj_data['mri_data_packed'][:, 8]
            mri_out_pts = mri_points[body_mask == False]
            idx = np.random.randint(0, mri_out_pts.shape[0], 6000)
            mri_out_pts_sampled = mri_out_pts[idx]
            subj_data.update(mri_out_pts=mri_out_pts_sampled)

        # if self.data_cfg.subjects == 'lowest_b1':
        #     idx = self.lowest_b1_subject_idx
        #     name = self.smpl_data['seq_names'][idx]
        # print(f'Lowest b1 subject idx: {self.lowest_b1_subject_idx}, name : {name}')

        # We need to compute smpl output for this subject
        # t1 = time.perf_counter()
        # smpl_data_batched = {key: subj_data[key][None] for key in SMPL_KEYS}
       
        # t2 = time.perf_counter()
        # print(f'Get item A {idx} took {t2-t1:.2f}s')

        # smpl_data = {key: val.squeeze(0) if torch.is_tensor(val) else val[0] for key, val in smpl_data.items()}  # remove B dim

        # import ipdb; ipdb.set_trace()
        # if return_smpl:
        #     subj_data.update({'smpl_output': smpl_output})
        if self.train_cfg['to_train'] != 'compression':  # We do not need to sample pts for the compression
            # import ipdb; ipdb.set_trace()
            if self.split == 'test' or get_whole_mri is True:
                subj_data.update(self.sample_whole_mri(subj_data))

            else:
                if self.data_cfg['sampling_strategy'] == 'mri':
                    # @varora
                    # updated to sample whole mri
                    subj_data = self.sample_whole_mri(subj_data, nb_points=self.data_cfg['n_pts_mri'])

                else:
                    raise DeprecationWarning('Needs to precompute SMP L, which takes time so avoid it')

                    if self.data_cfg['sampling_strategy'] == 'per_part':
                        subj_data.update(self.sample_points(smpl_output, subj_data))
                    elif self.data_cfg['sampling_strategy'] == 'per_tissue':
                        subj_data.update(self.sample_given_number(subj_data, smpl_output))
                    elif self.data_cfg['sampling_strategy'] == 'boundary':
                        subj_data.update(self.sample_boundary(smpl_output, subj_data))
                    elif self.data_cfg['sampling_strategy'] == 'local':
                        subj_data.update(self.sample_tissue_per_part(smpl_output, subj_data))
                    else:
                        raise NotImplementedError(
                            f"Sampling strategy {self.data_cfg['sampling_strategy']} not implemented")

                if self.data_cfg['sample_can_points']:
                    # subsample can_points_dictionary
                    idx_rand = np.random.randint(0, self.can_points_dictionary['can_points'].shape[0], 6000)
                    can_points_dict = {key: val[idx_rand] for key, val in self.can_points_dictionary.items()}
                    subj_data.update(can_points_dict)

                if self.data_cfg['sample_can_hands']:
                    subj_data.update(self.can_hands_dictionary)
        else:
            subj_data = {key: val for key, val in subj_data.items() if
                         key not in ['mri_data_packed', 'mri_data_shape1', 'mri_data_keys', 'mri_data_shape0']}

            # t2 = time.perf_counter()
        # print(f'Get item {idx} took {t2-t1:.2f}s')
        # print(subj_data.keys())
        return subj_data

    def sample_whole_mri(self, subj_data, nb_points=None):

        stacked_data = subj_data['mri_data_packed']
        mri_points_nb = subj_data['mri_data_shape0']
        mri_data_shape1 = subj_data['mri_data_shape1']
        # mri_data_keys = subj_data['mri_data_keys']

        # import ipdb; ipdb.set_trace()
        # to_ret = dict(mri_points=subj_data['mri_points'],
        #               mri_occ=subj_data['mri_occ'],
        #               mri_coords=subj_data['mri_coords'],
        #               body_mask = subj_data['body_mask'],
        #               part_id = subj_data['part_id'],
        #               skinning_weights = subj_data['skinning_weights'])
        # to_ret = dict(mri_points=subj_data['mri_points'],
        #               mri_occ=subj_data['mri_occ'][..., None],
        #               mri_coords=subj_data['mri_coords'],
        #               body_mask = subj_data['body_mask'][..., None],
        #               part_id = subj_data['part_id'][..., None],
        #               skinning_weights = subj_data['skinning_weights'])

        # [print(val.shape) for key, val in to_ret.items()]
        # stacked_data = np.concatenate([val for key, val in to_ret.items()], axis=1)

        # This could be optimised by stacking the upper arrays
        t1 = time.perf_counter()
        if nb_points is not None:
            # sample random points inside the body
            # idx_bodymask = np.where(subj_data['body_mask']==1)[0] # 0.16 s
            # # import ipdb; ipdb.set_trace()
            idx_to_keep = np.random.randint(0, mri_points_nb, nb_points)  # 0.05 (0.22)
            # # idx_to_keep = np.random.choice(idx_bodymask.shape[0], nb_points, replace=False) # 3.35s
            # to_ret = {key: val[idx_bodymask][idx_to_keep] for key, val in to_ret.items()}# 3sec
            stacked_data = stacked_data[idx_to_keep]  # 3sec
            # print(f'Number of MRI points to evaluate: {to_ret["points"].shape[0]}')
        t2 = time.perf_counter()
        # print(f'Time to sample {nb_points} points: {t2-t1:.5f}s')

        # import ipdb; ipdb.set_trace()
        # todo retrieve the last row to be mnri value and put it in key `mri_values`
        # @varora
        # retrieve the last row to be mri values and put it in key `mri_values`

        # import ipdb; ipdb.set_trace()
        if subj_data['mri_data_packed'].shape[1] > 33:
            if self.train_cfg.mri_values is True:
                to_ret = dict(mri_points=stacked_data[:, :3],
                              mri_coords=stacked_data[:, 3:6],
                              mri_occ=stacked_data[:, 6],
                              part_id=stacked_data[:, 7],
                              body_mask=stacked_data[:, 8],
                              skinning_weights=stacked_data[:, 9:33],
                              mri_values=stacked_data[:, 33:])
            else:
                to_ret = dict(mri_points=stacked_data[:, :3],
                              mri_coords=stacked_data[:, 3:6],
                              mri_occ=stacked_data[:, 6],
                              part_id=stacked_data[:, 7],
                              body_mask=stacked_data[:, 8],
                              skinning_weights=stacked_data[:, 9:33])
        else:
            to_ret = dict(mri_points=stacked_data[:, :3],
                          mri_coords=stacked_data[:, 3:6],
                          mri_occ=stacked_data[:, 6],
                          part_id=stacked_data[:, 7],
                          body_mask=stacked_data[:, 8],
                          skinning_weights=stacked_data[:, 9:])

        assert to_ret['skinning_weights'].shape[
                   1] == 24, f'Wrong number of skinning weights: {to_ret["skinning_weights"].shape[1]}'

        not_ret_keys = ['mri_data_packed', 'mri_data_shape1', 'mri_data_keys', 'mri_data_shape0']
        to_ret.update({key: val for key, val in subj_data.items() if key not in not_ret_keys})

        return to_ret

    def sample_can_space(self):

        nb_outside_pts = self.data_cfg['nb_points_canspace']
        nb_skin_pts = self.data_cfg['n_skin_pts']

        smpl_can = self.smpl.forward_canonical()
        can_vertices = smpl_can.vertices.detach().cpu().numpy()
        can_mesh = trimesh.Trimesh(can_vertices.squeeze(), self.faces, process=False)

        # Uniform points
        max_val = can_vertices.max() + self.data_cfg['uniform_sampling_padding']
        uniform_pts = torch.rand((nb_outside_pts, 3)) * max_val * 2 - max_val  # Sample uniformly
        # uniform_pts = center_on_voxel(uniform_pts.cpu().numpy(), smpl_data) # Align to voxel grid

        # import ipdb; ipdb.set_trace()

        # Skin surface points
        if nb_skin_pts > 0:
            points, faces = can_mesh.sample(nb_skin_pts, return_index=True)
            normals = can_mesh.face_normals[faces]

            surface_offset_min = self.data_cfg['surface_offset_min']
            surface_offset_max = self.data_cfg['surface_offset_max']
            offset = surface_offset_min + (surface_offset_max - surface_offset_min) * np.abs(
                np.random.randn(nb_skin_pts, 1))
            surface_points = points + offset * normals
            surface_points = surface_points.astype(np.float32)
            # surface_points = center_on_voxel(surface_points.cpu().numpy(), smpl_data)

            can_pts = np.concatenate((uniform_pts, surface_points), axis=0)
        else:
            can_pts = uniform_pts

        # from leap.tools.libmesh import check_mesh_contains
        # can_occ = check_mesh_contains(can_mesh, can_pts).astype(np.float32)
        # can_occ = can_mesh.contains(can_pts)
        # gt_occ, body_mask = load_occupancy(smpl_data, can_pts, interp_order=0)

        # import ipdb; ipdb.set_trace()
        from pysdf import SDF
        f = SDF(can_mesh.vertices, can_mesh.faces);  # (num_vertices, 3) and (num_faces, 3)
        can_occ = f.contains(can_pts)

        return dict(can_points=can_pts, can_occ=can_occ)

    def sample_can_hands(self):

        nb_pts = self.data_cfg['n_points_hands']

        smpl_can = self.smpl(betas=torch.zeros(1, 10).to(self.smpl.smpl.betas.device))
        can_vertices = smpl_can.vertices.detach().cpu().numpy()
        can_mesh = trimesh.Trimesh(can_vertices.squeeze(), self.faces, process=False)

        bone_trans = self.smpl.compute_bone_trans(smpl_can.full_pose, smpl_can.joints)  # 1,24,4,4
        bbox_min, bbox_max = self.smpl.get_bbox_bounds_trans(smpl_can.vertices,
                                                                        bone_trans)  # (B, K, 1, 3) [can space]
        n_parts = bbox_max.shape[1]

        #### Sample points inside local boxes

        bbox_size = (bbox_max - bbox_min).abs() * self.bbox_padding - 1e-3  # (B,K,1,3)
        bbox_center = (bbox_min + bbox_max) * 0.5
        bb_min = (bbox_center - bbox_size * 0.5)  # to account for padding

        ##### Sample points uniformly in the body bounding boxes
        uniform_points = bb_min + torch.rand((1, n_parts, nb_pts, 3)) * bbox_size  # [0,bs] (B,K,N,3)
        abs_transforms = torch.inverse(bone_trans)  # B,K,4,4
        uniform_points = (abs_transforms.reshape(1, n_parts, 1, 4, 4).repeat(1, 1, nb_pts, 1, 1) @ F.pad(uniform_points,
                                                                                                         [0, 1],
                                                                                                         "constant",
                                                                                                         1.0).unsqueeze(
            -1))[..., :3, 0]

        if self.data_cfg.sample_can_toes is False:
            hand_pts = uniform_points[0, 20:24]
        else:
            toes_pts = uniform_points[0, 10:12]
            hand_pts = torch.cat((toes_pts), dim=1)
        can_pts = hand_pts.reshape(-1, 3).numpy()
        # can_occ = can_mesh.contains(can_pts)

        # from leap.tools.libmesh import check_mesh_contains
        # can_occ = check_mesh_contains(can_mesh, can_pts).astype(np.float32)

        # gt_occ, body_mask = load_occupancy(smpl_data, can_pts.cpu().numpy(), interp_order=0)

        # import ipdb; ipdb.set_trace()
        from pysdf import SDF
        f = SDF(can_mesh.vertices, can_mesh.faces);  # (num_vertices, 3) and (num_faces, 3)
        can_occ = f.contains(can_pts)

        return dict(hands_can_points=can_pts, hands_can_occ=can_occ)

    def display_sample(self, idx, color_style='by_class', can_only=False):

        assert color_style in ['by_class', 'by_skinning']
        
        nb_beta = 10

        data = self.__getitem__(idx, return_smpl=True)
        points = data['mri_points']
        occ = data['mri_occ']

        if self.data_cfg['sample_can_points']:
            can_points = data['can_points']
            can_occ = data['can_occ']

        if self.data_cfg['sample_can_hands']:
            hands_can_points = data['hands_can_points']
            hands_can_occ = data['hands_can_occ']

        if color_style == 'by_skinning':
            skinning_weights = data['skinning_weights'][:]
            from utils.smpl_utils import weights2colors
            skinning_color = weights2colors(skinning_weights)

        print(f"Number of points sampled: {points.shape[0]}")

        print(f'Visualizing sample {data["seq_names"]}')
        if points.shape[0] > 50000:
            points = points[:-1:500]
            occ = occ[:-1:500]  # Don't display all the points otherwise it will crash

        # mri_points, mri_coords = sample_mri_pts(data, body_only=BODY_ONLY)
        # mri_points = mri_points[0:-1:100]
        # print(f'Number of MRI points displayed: {mri_points.shape[0]}')

        # smpl_output = self.smpl(**data)
        # smpl_verts = smpl_output.vertices.squeeze().numpy()
        # smpl_verts = data['body_verts'].numpy()
        free_verts = data['body_verts_free'].numpy()
        # smpl_mesh = trimesh.Trimesh(smpl_verts, self.faces, process=False)
        free_verts_mesh = trimesh.Trimesh(free_verts, self.faces, process=False)

        for occ_val in [0, 1, 2]:
            assert np.shape(points[occ == 0])[0] > 0, f'No sampled points with occ value {occ_val}'

        # Print stats
        print(f'Number of points sampled: {points.shape[0]}')
        print(
            f'Number of points with occ value NO: {np.shape(points[occ == 0])[0]}, {np.shape(points[occ == 0])[0] / points.shape[0] * 100:.2f}%')
        print(
            f'Number of points with occ value LT: {np.shape(points[occ == 1])[0]}, {np.shape(points[occ == 1])[0] / points.shape[0] * 100:.2f}%')
        print(
            f'Number of points with occ value AT: {np.shape(points[occ == 2])[0]}, {np.shape(points[occ == 2])[0] / points.shape[0] * 100:.2f}%')
        print(
            f'Number of points with occ value BONE: {np.shape(points[occ == 3])[0]}, {np.shape(points[occ == 3])[0] / points.shape[0] * 100:.2f}%')

        scene = pyrender.Scene(ambient_light=[.1, 0.1, 0.1], bg_color=[1.0, 1.0, 1.0])
        if color_style == 'by_class':
            scene.add(vis_create_pc(points[occ == 0], color=(0., 0., 0.)))  # NO
            scene.add(vis_create_pc(points[occ == 1], color=(1., 0., 0.)))  # LT
            scene.add(vis_create_pc(points[occ == 2], color=(1., 1., 0.)))  # AT
            scene.add(vis_create_pc(points[occ == 3], color=(0., 0., 1.)))  # BONE
        elif color_style == 'by_skinning':
            m = pyrender.Mesh.from_points(points, colors=skinning_color)
            scene.add(m)
        else:
            raise NotImplementedError(f'Color style {color_style} not implemented')
        # scene.add(vis_create_pc(mri_points, color=(0., 0., 0.), radius=0.005))
        # scene.add(pyrender.Mesh.from_trimesh(smpl_mesh, smooth=False, wireframe=True))
        scene.add(pyrender.Mesh.from_trimesh(free_verts_mesh, smooth=False, wireframe=True))

        if not can_only:
            pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=False)

        if self.data_cfg['sample_can_points']:
            scene = pyrender.Scene(ambient_light=[.1, 0.1, 0.1], bg_color=[1.0, 1.0, 1.0])
            scene.add(vis_create_pc(can_points[can_occ == 0], color=(1., 0., 0.)))
            scene.add(vis_create_pc(can_points[can_occ == 1], color=(0., 1., 0.)))

            can_vertices = self.smpl.forward_canonical(
                betas=torch.zeros(1,nb_beta)).vertices.detach().cpu().numpy()
            can_mesh = trimesh.Trimesh(can_vertices.squeeze(), self.faces, process=False)

            scene.add(pyrender.Mesh.from_trimesh(can_mesh, smooth=False, wireframe=True))
            pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=False)

        if self.data_cfg['sample_can_hands']:
            # hands_can_points=can_pts, hands_can_occ=can_occ
            scene = pyrender.Scene(ambient_light=[.1, 0.1, 0.1], bg_color=[1.0, 1.0, 1.0])
            scene.add(vis_create_pc(hands_can_points[hands_can_occ == 0], color=(1., 0., 0.)))
            scene.add(vis_create_pc(hands_can_points[hands_can_occ == 1], color=(0., 1., 0.)))

            can_vertices = self.smpl.forward_canonical(betas=torch.zeros(1,nb_beta)).vertices.detach().cpu().numpy()
            can_mesh = trimesh.Trimesh(can_vertices.squeeze(), self.faces, process=False)

            scene.add(pyrender.Mesh.from_trimesh(can_mesh, smooth=False, wireframe=True))
            pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=False)

    def find_lowest_b1(self):
        all_beta_1 = [self.smpl_data['betas'][i][1] for i in range(len(self))]
        subj_idx = np.array(all_beta_1).argmin()
        return subj_idx

    def find_highest_b1(self):
        all_beta_1 = [self.smpl_data['betas'][i][1] for i in range(len(self))]
        subj_idx = np.array(all_beta_1).argmax()
        return subj_idx

    @torch.no_grad()
    def _getitem_synthetic(self, return_smpl=False):

        # We add a modulo in get_item so that we can use a batch size bigger than the dataset size.
        # dataset[batch_idx] = dataset[batch_idx % len(dataset)]
        # print(f'idx: {idx}')

        subj_data = {key: self.smpl_data[key][0] for key in self.smpl_data.keys()}

        mri_pose = subj_data['body_pose'].clone()
        mri_global_orient = subj_data['global_orient'].clone()

        # Set all the SMPL params to zero
        subj_data['betas'] = np.zeros_like(subj_data['betas'])
        subj_data['body_pose'] = np.zeros_like(subj_data['body_pose'])
        subj_data['global_orient'] = np.zeros_like(subj_data['global_orient'])
        subj_data['transl'] = np.zeros_like(subj_data['transl'])
        subj_data['global_orient_init'] = np.zeros_like(subj_data['global_orient_init'])

        nb_beta = subj_data['betas'].shape[0]
        nb_pose = subj_data['body_pose'].shape[0]

        xpose = self.smpl.canonical_x_bodypose[0].cpu().numpy()
        random_pose = np.random.randn(nb_pose) * 3.14 / 8
        random_betas = (np.random.rand(nb_beta) - 0.5) * 4
        mri_pose = mri_pose.cpu().numpy()

        if self.data_cfg.synt_style == 'random':
            subj_data['body_pose'] = xpose
            subj_data['betas'] = random_betas.astype(np.float32)
            
        if self.data_cfg.synt_style == 'fixed':
            subj_data['body_pose'] = xpose
            subj_data['betas'] = np.zeros(nb_beta).astype(np.float32)
            subj_data['betas'][0:2] = 2

        if self.data_cfg.synt_style == 'random_per_joint':
            subj_data['betas'] = random_betas.astype(np.float32)

            subj_data['body_pose'] = np.zeros(nb_pose).astype(np.float32)
            random_param_idx = np.random.randint(0, nb_pose)
            subj_data['body_pose'][random_param_idx] = np.random.randn(1) * 3.14
            # print(subj_data['body_pose'][random_param_idx] )

        
        # subj_data['body_verts'] = smpl_output.vertices

       
        # subj_data = {key: val.squeeze(0) if torch.is_tensor(val) else val[0] for key, val in subj_data.items()}  # remove B dim

        # subj_data.update({'v_shaped': v_shaped.squeeze(0)})
        # subj_data.update(self.sample_points(smpl_output, subj_data))

        # if return_smpl:
        #     subj_data.update({'smpl_output': smpl_output})
        return subj_data

    def __len__(self):
        if self.split == 'val':
            if self.synthetic:
                return 8
            else:
                return self.smpl_data['betas'].shape[0]
        else:
            if self.synthetic:
                return 128  # Artificial number since the data are generated on the fly
            else:
                return self.smpl_data['betas'].shape[0]