import os

import hit_config as cg
from model.hit_model import HITModel
from utils.exppath import Exppath
from model.mysmpl import MySmpl
from omegaconf import OmegaConf


class HitLoader():
    
    def __init__(self, checkpoint, cfg):
        self.checkpoint = checkpoint
        self.cfg = cfg
        self.device = 'cuda:0'

    @classmethod
    def from_expname(cls, exp_name, cfg={}, wdboff=False):
        
        # Find folder and checkpoint
        exp_path = Exppath(exp_name, local=False)
        
        train_folder = exp_path.find_train_folder()
        
        # retrive the checkpoint with highest validation loss : model-epoch=0189-val_accuracy=0.758575.ckpt 
        checkpoint = exp_path.get_best_checkpoint()
        
        # get parameters from config file in the parent folder of the checkpoint
        cfg_file = os.path.join(train_folder, 'config.yaml')
        print(f"Loading config file {cfg_file}")

        assert os.path.exists(cfg_file), f"Config file {cfg_file} does not exist"
        assert os.path.exists(checkpoint), f"Checkpoint {checkpoint} does not exist"
            
        # Load config 
        
        if os.path.isfile(cfg_file):
            with open(cfg_file, 'r') as f:
                cfg_checkpoint = OmegaConf.load(f)
            cfg = cfg_checkpoint
             
        return cls(checkpoint, cfg)
            
    def load(self):
        
        self.smpl = MySmpl(model_path=cg.smplx_models_path, gender=self.cfg.smpl_cfg.gender).to(self.device)
        self.hit_model = HITModel(train_cfg=self.cfg.train_cfg, smpl=self.smpl).to(self.device)
        self.hit_model.initialize(checkpoint_path=self.checkpoint, train_cfg = self.cfg.train_cfg)
        

        
