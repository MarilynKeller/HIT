import glob
import os
import wandb
import hit.hit_config as cg

class Exppath:
    
    def __init__(self, exp_name, local=True):
        """ If find_train_folder is False, the run will be looked for locally, otherwise it will be looked for 
        both locally and on the cluster."""
        
        self.exp_name = exp_name
        self.local = local

    @property      
    def local_train_folder(self):
        return os.path.join(cg.trained_models_folder, self.exp_name)
    
    @property
    def cluster_train_folder(self):
        return os.path.join(cg.cluster_trained_models_folder, self.exp_name)
      
    def find_train_folder(self):
        # if self.local == True:
        train_folder = os.path.join(cg.trained_models_folder, self.exp_name)
        # else:
        #     if self.is_experiment_on_cluster and self.is_experiment_local:
        #         raise ValueError('Both local and remote wandb runs found')
        #     elif self.is_experiment_on_cluster:
        #         train_folder = os.path.join(cg.cluster_trained_models_folder, self.exp_name)
        #     elif self.is_experiment_local:
        #         train_folder = os.path.join(cg.trained_models_folder, self.exp_name)
        #     else:
        #         raise ValueError(f'No existing wandb run found for experiment {self.exp_name}, returning the local path')
        print(f'Found wandb run for experiment {self.exp_name} in {train_folder}')
        return train_folder
            
    @property
    def is_experiment_on_cluster(self):
        cluster_wandb_run_folders = os.path.join(cg.cluster_trained_models_folder, self.exp_name, 'wandb/run*')
        cluster_run_folders = glob.glob(cluster_wandb_run_folders)
        if len(cluster_run_folders) > 0:
            return True

    @property
    def is_experiment_local(self):
        local_wandb_run_folders = self.local_train_folder+   '/wandb/run*'
        local_run_folders = glob.glob(local_wandb_run_folders)
        if len(local_run_folders) > 0:
            return True

    def get_last_run_id(self):
        """ Return the id of the last wandb run for a given experience name. Look locally only.
        return False if no run was found."""
        train_folder = self.find_train_folder()
        run_file_pattern = os.path.join(train_folder, 'wandb/run*')
        runs = glob.glob(run_file_pattern)
        if len(runs) == 0:
            return False
            
        run_folders_sorted = sorted(runs, key=os.path.getmtime, reverse=True)
        id = run_folders_sorted[0].split('/')[-1].split('-')[-1]
        return id
            
 
    def get_wandb_logger(self, wdboff=False):
        """ Load the last wandb run for a given experience name. 
        This functions list all the runs for a given experience name locally and on the cluster,
        finds the id of the last run, and returns the corresponding logger, instanciated with wandb.init
        """
        train_folder = self.find_train_folder()
        id = self.get_last_run_id()
        
        if wdboff:
            wandb_mode = 'disabled'
        else:
            wandb_mode = 'online'
        print(f"Resuming wandb logging for run {id}")
        logger = wandb.init(name=self.exp_name, project='hit', id=id, dir=train_folder, resume='must', mode=wandb_mode)
        return logger

    def get_best_checkpoint(self):
        """Returns the path to the best checkpoint for a given experience name.
        This function list ckpts both locally and on the cluster to retrive the checkpoint 
        with highest validation loss : model-epoch=0189-val_accuracy=0.758575.ckpt"""
        train_folder = self.find_train_folder()
        ckpt_folder = os.path.join(train_folder, 'ckpts')
        ckpt_files = glob.glob(os.path.join(ckpt_folder, 'model*.ckpt'))
        if len(ckpt_files) == 0:
            raise ValueError(f'No checkpoint found in {ckpt_folder}')
        key = lambda x: float(x.split('val_accuracy=')[1].split('.ckpt')[0]) # sort the checkpoints files by validation loss  
        checkpoint = sorted(ckpt_files, key=key)[-1] # Take the checkpoint with the highest validation loss
        return checkpoint
    

    def get_last_checkpoint(self):
        train_folder = self.find_train_folder()
        ckpt_folder = os.path.join(train_folder, 'ckpts')
        ckpt_files = glob.glob(os.path.join(ckpt_folder, 'last.ckpt'))
        if len(ckpt_files) == 0:
            raise ValueError(f'No last.ckpt checkpoint found in {ckpt_folder}')
        return ckpt_files[0]