# HIT

Official repo of the 2024 paper: *HIT: Estimating Internal Human Implicit Tissues from the Body Surface*

[[paper]](https://hit.is.tue.mpg.de/media/upload/4978.pdf) [[project page]](https://hit.is.tue.mpg.de/)

<img src="assets/teaser.png" width="500">

## Table of Contents
<!-- outline -->
- [What is HIT ?](#what-is-hit-)
- [Content of this repo](#content-of-this-repo)
- [Installation](#installation)
- [Dataset](#dataset)
- [Demos](#demos)
- [Training HIT](#training)
- [Testing HIT](#testing)

## What is HIT ?
HIT is a neural network that learns to infer the internal tissues of the human body from its surface. The input is a 3D body represented as SMPL parameters (a shape vector β and a pose vector θ) and the output is an implicit function, that given a point in space, returns the probability of this point being part the following tissue:
- lean tissue (muscules, organs) (LT)
- adipose tissue (subcutaneous) (AT)
- bone (BT)
- empty space (inside the lungs, outside the body) (E)

The implicit HIT function can also be used to generate the 3D mesh of the tissues using marching cube.

The figure below illustrates our approach. Given a posed body shape (β, θ), and a 3D point xm, HIT first canonicalizes this point to the corresponding location xc inside an average template body, and then predicts the class of the tissue at location xc.

<img src="assets/pipeline.png" width="700">

## Content of this repo

This repo contains the code to train and test HIT on the HIT dataset (available on our project page). It also contains some demo code to infer the tissue meshes for a given SMPL body.


# Installation
```shell
git clone https://github.com/MarilynKeller/HIT (This will be he final link eventually)
cd HIT
```

#### Set up a virtual environment
```shell
conda create -n hit python=3.9
conda activate hit
```

#### Install packages
Check CUDA toolkit version
```shell
nvcc --version
```
Install torch depending on CUDA toolkit version. See: https://pytorch.org/get-started/previous-versions/ . For 11.8:
```shell
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
```

Install relevant packages
```shell
pip install -r requirements.txt
pip install  git+https://github.com/facebookresearch/pytorch3d.git 
pip install  git+https://github.com/MPI-IS/mesh.git
pip install  git+https://github.com/mattloper/chumpy
```

The **LEAP** package is used to create the ground truth occupancy and for visualization.
```shell
cd learning
mkdir external
cd external 
git clone https://github.com/neuralbodies/leap.git
cd leap
python setup.py build_ext --inplace
pip install -e .
```

Then install the HIT package
```shell
cd HIT
pip install -e .
```

#### SMPL model files
Download the SMPL model from https://smpl.is.tue.mpg.de/ and update the path `smplx_models_path` in `hit_config.py` to the proper path.
The folder hierarchie should be the following:
```
${smplx_models_path}
├── smpl
│   ├── SMPL_FEMALE.pkl
│   └── SMPL_MALE.pkl
│   └── SMPL_NEUTRAL.pkl
```

# Dataset

Download the HIT dataset from the Download tab at [https://hit.is.tue.mpg.de/].
This dataset contains data for 157 males and 241 females. For each subject it contains:

```yaml
  'gender': "gender of the subject",
  'mri_seg': "annotated array with the labels 0,1,2,3",
  'mri_labels': "dictionary of mapping between label integer and name",
  'mri_seg_dict': "a dictionary of individual masks of the different tissues (LT, AT, BT, ...)",
  'resolution': "per slice resolution in meters",
  'center': "per slice center, in pixels",
  'smpl_dict': "dictionary containing all the relevant SMPL parameters of the   subject alongwith mesh faces and vertices ('verts': original fit, 'verts_free': compressed fit")
```

In the `hit_config.py` file, set the path to the dataset folder as `packaged_data_folder`.

## Check the data

**Caching**
The first loading of the dataset requires sampling which takes time. We do this once and then cache the result for further fast loading.
To forces the recaching of the dataset for a gender, run:
```shell
python ../tutorials/load_data.py -s train -i 0 --gender male --all --recache  
```

**Vizualize the data**
You can visualize a subject of the dataset using:

```shell
python demos/load_data.py -s train -i 0 --gender female -D
```

This shows the tight SMPL fit to the subject and the MRI points sampled for one iteration of the training, colored according to their ground truth label:
- black: E (empty)
- red: LT (lean tissue)
- yellow: AT (adipose tissue)
- blue: BT (bone tissue)

<img src="assets/data_vizu.png" width="500">

After closing the window, this will show the points sampled in the SMPL canonical space for one iteration of the training.
In red are the points outside the SMPL template mesh, in green the points inside.

<img src="assets/data_can_vizu.png" width="500">

-------------------

# Demos

## Pretrained models

**! The pretrained models are not available yet, we will release them in the comming weeks.**
<!-- 
You can download the pretrained models checkpoints for male and female from the Download tab at [https://hit.is.tue.mpg.de/].
In the `hit_config.py` file, set the path to the pretrained models in `...`. -->

## Infer the tissue meshes for the SMPL template
```shell
python demos/infer_smpl.py  --exp_name=rel_male --to_infer smpl_file
```

## Infer the tissue meshes given a SMPL body as .pkl 
```shell
python demos/infer_smpl.py  --exp_name=rel_male --to_infer smpl_file --target_body assets/sit.pkl
``` 

# Training 

To load the training dataset in memory, you will need at least TODO GB of RAM for the male dataset and 62 GB for the female dataset.

## Training parameters
The default training parameters are in ```config.yaml```. The project uses hydra to load this config file.
Each parameter of this file can be overwriten through command line arguments.

## Monitoring the training

The training is logged on Weight and Biasis. You can set the wandb entity and project name in `hit_config.py`.

## Launch training

First you need to pretrain the submodules on generated SMPL meshes to learn the LBS and inverse beta fields for each gender. Here we give the commands for 'male'

```shell
python hit/train.py exp_name=pretrained_smpl  data_cfg.synt_style=random  train_cfg.to_train=pretrain  batch_size=8 trainer.check_val_every_n_epoch=5 lr=0.0001 train_cfg.lambda_betas=50 data_cfg.synthetic=True train_cfg.networks.lbs.geometric_init=true trainer.max_epochs=1000 smpl_cfg.gender=male 
```

Once this trained for mal and female, you can train HIT for this gender:
```shell
python hit/train.py exp_name=hit_male smpl_cfg.gender=male
```

Here `hit_male` is the name of the experiment. This training will be logged and saved in a folder with this name. Note that if you launch a new training with the same name, the last checkpoint with this name will be loaded. 


# Testing

Coming soon.



# Acknoledgments

We thank the authors of the [COAP](https://github.com/markomih/COAP) and [gDNA](https://github.com/xuchen-ethz/gdna) for their codebase. HIT is built on top of these two projects.

# Citation

If you use this code, please cite the following paper:

```
@inproceedings{Keller:CVPR:2024,
  title = {{HIT}: Estimating Internal Human Implicit Tissues from the Body Surface},
  author = {Keller, Marilyn and Arora, Vaibhav and Dakri, Abdelmouttaleb and Chandhok, Shivam and 
  Machann, Jürgen and Fritsche, Andreas and Black, Michael J. and Pujades, Sergi},   
  booktitle = {Proceedings IEEE/CVF Conf.~on Computer Vision and Pattern Recognition (CVPR)},
  month = jun,
  year = {2024},
  month_numeric = {6}}
```

## Contact

For more questions, please contact hit@tue.mpg.de

For commercial licensing, please contact ps-licensing@tue.mpg.de
