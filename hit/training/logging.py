import os
import wandb


def log_canonical_meshes(hit_trainer, batch):
        for li in hit_trainer.train_cfg.mri_labels:
            if li == 'NO':
                # Do not export canonical mesh for empty tissues (lungs and outside)
                continue
            else:
                channel_index = hit_trainer.train_cfg['mri_labels'].index(li)
                can_mesh = hit_trainer.hit.generate_canonical_mesh(batch, channel_index)
                if not (can_mesh is None):
                    can_mesh_path = os.path.join(hit_trainer.logger.save_dir, 'val', 'can_mesh.glb')
                    os.makedirs(os.path.dirname(can_mesh_path), exist_ok=True)
                    can_mesh.export(can_mesh_path)
                    wandb.log({f"val_meshes/can_mesh_{li}": wandb.Object3D(can_mesh_path)})   
        
def log_slices(hit_trainer, batch, values=["occ", "sw", "beta", "fwd_beta", "comp"]):
    
    smpl_output = hit_trainer.smpl(**batch, return_verts=True, return_full_pose=True)
    
    # Horizontal slices
    # for z0 in [-0.2, -0.4, 0, 0.3] :
    z0=-0.2
    images = hit_trainer.hit.evaluate_slice(batch, 
                                            smpl_output, 
                                            z0=z0, 
                                            axis='z', 
                                            values=values, 
                                            res=0.001)

    for value, image in zip(values, images):
        wandb.log({f"slicesZ/{value}_z={z0}": wandb.Image(image)})
    
    # Frontal slices
    z0=0
    images = hit_trainer.hit.evaluate_slice(batch, 
                                            smpl_output, 
                                            z0=z0, 
                                            axis='y', 
                                            values=values, 
                                            res=0.005)
    for value, image in zip(values, images):
        wandb.log({f"slicesY/{value}_z={z0}": wandb.Image(image)})

    
    # if not self.train_cfg['to_train'] == 'compression':      
    #     sl.plot_slice_levelset(disp_np, values=values_np, to_plot=False)
    #     im_path = os.path.join('/tmp', f'disp_{z0:.2f}.png')
    #     plt.savefig(im_path)
    #     plt.close()
    #     print(f'Saved disp image to {im_path}')  
    
    
    # if self.train_cfg['compressor']:
    #     d = self.hit.deformer.compressor(xc_batch, cond_create(torch.zeros_like(batch['betas'])))
    #     d_np = d.detach().cpu().numpy()[0]
    #     sl.plot_slice_levelset(d_np, values=d_np, to_plot=False, iscompression=True)
    #     im_path = os.path.join('/tmp', f'compression_{z0:.2f}.png')
    #     plt.savefig(im_path)
    #     plt.close()
    #     print(f'Saved disp image to {im_path}')  
    #     wandb.log({f"slices_comp/compression_{z0}": wandb.Image(im_path)})
                