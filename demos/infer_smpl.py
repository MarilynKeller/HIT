"""Given a SMPL parameters, infer the tissues occupancy"""

import argparse
import os
import torch
import trimesh

from hit.utils.model import HitLoader
from hit.utils.data import load_smpl_data
import hit.hit_config as cg

def main():
    
    parser = argparse.ArgumentParser(description='Infer tissues from SMPL parameters')
    
    parser.add_argument('--to_infer', type=str, default='smpl_template', choices=['smpl_template', 'smpl_file'], 
                        help='Whether to infer from a SMPL template or a SMPL file')
    parser.add_argument('--exp_name', type=str, default='rel_male',
                        help='Name of the checkpoint experiment to use for inference' 
                        ) #TODO change to checkpoint path
    parser.add_argument('--target_body', type=str, default='assets/sit.pkl', 
                        help='Path to the SMPL file to infer tissues from')
    parser.add_argument('--out_folder', type=str, default='output',
                        help='Output folder to save the generated meshes')
    parser.add_argument('--device', type=str, default='cuda:0', choices=['cuda:0', 'cpu'],
                        help='Device to use for inference')
    parser.add_argument('--ckpt_choice', type=str, default='best', choices=['best', 'last'],
                        help='Which checkpoint to use for inference')
    parser.add_argument('--output', type=str, default='meshes', choices=['meshes', 'slices'], 
                        help='Form of the output (either extract a mesh of each tissue, either extract slices of each tissue)')
    parser.add_argument('--betas', help="List of the 2 first SMPL betas to use for inference", nargs='+', type=float, default=[0.0, 0.0])
    # to enter this parameter, use the following syntax: --betas 0.0 0.0
    
    args = parser.parse_args()

    exp_name = args.exp_name
    target_body = args.target_body
    ckpt_choice = args.ckpt_choice
    device = torch.device(args.device)
    
    out_folder = os.path.join(args.out_folder, f'{exp_name}_{ckpt_choice}')
    
    # Create a data dictionary containing the SMPL parameters 
    if args.to_infer == 'smpl_template':
        data = {}
        data['global_orient'] = torch.zeros(1, 3).to(device) # Global orientation of the body
        data['body_pose'] = torch.zeros(1, 69).to(device) # Per joint rotation of the body (21 joints x 3 axis)
        data['betas'] = torch.zeros(1, 10).to(device) # Shape parameters, values should be between -2 and 2
        data['betas'][0,0] = args.betas[0]
        data['betas'][0,1] = args.betas[1]
        data['transl'] = torch.zeros(1, 3).to(device) # 3D ranslation of the body in meters
    else:
        assert target_body.endswith('.pkl'), 'target_body should be a pkl file'
        assert os.path.exists(target_body), f'SMPL file "{target_body}" does not exist'
        data = load_smpl_data(target_body, device)

        
    # Create output folder
    os.makedirs(out_folder, exist_ok=True)
    
    # Load HIT model
    hl = HitLoader.from_expname(exp_name, ckpt_choice=ckpt_choice)
    hl.load()
    hl.hit_model.apply_compression = False

    # Run smpl forward pass to get the SMPL mesh
    smpl_output = hl.smpl(betas=data['betas'], body_pose=data['body_pose'], global_orient=data['global_orient'], trans=data['transl'])
    
    if args.output == 'slices':
        # Extract slices of each tissue
        # values = ["occ", "sw", "beta"] # Values to extract slices from
        values = ["occ","sw", "beta", "comp"] # Values to extract slices from
        for axis, z0 in zip(['y', 'z', 'z'], [0.0, 0.0, -0.4]):
            images = hl.hit_model.evaluate_slice(data, smpl_output, z0=z0, axis=axis, values=values, res=0.002)
            # Save the slice images
            for value, image in zip(values, images):
                image.save(os.path.join(out_folder, f'{value}_{axis}={z0}.png'))
            print(f'Saved slice {os.path.abspath(out_folder)}')
        print(f'Slices saved in {os.path.abspath(out_folder)}')
        
    elif args.output == 'meshes':
        # Extract the mesh 
        extracted_meshes, _ = hl.hit_model.forward_rigged(data['betas'], 
                                                                body_pose=data['body_pose'], 
                                                                global_orient=data['global_orient'], 
                                                                transl=data['transl'],
                                                                mise_resolution0=64)
        # Extracted meshes are in the form of a list of 3 trimesh objects corresponding to the 3 tissues 'LT', 'AT', 'BT'
        # LT : Lean Tissue (muscle and organs, merged with the visceral and intra-muscular fat)
        # AT : Adipose Tissue (subcutaneous fat)
        # BT : Bone Tissue (long bones, we only predict the femur, radius-ulna, tibia and fibula)
        
        smpl_mesh = trimesh.Trimesh(vertices=smpl_output.vertices[0].detach().cpu().numpy(), faces=hl.smpl.faces)
        
        # Save all the meshes
        smpl_mesh.export(os.path.join(out_folder, 'smpl_mesh.obj'))
        extracted_meshes[0].export(os.path.join(out_folder, 'LT_mesh.obj'))
        extracted_meshes[1].export(os.path.join(out_folder, 'AT_mesh.obj'))
        extracted_meshes[2].export(os.path.join(out_folder, 'BT_mesh.obj'))
        
        print(f'Meshes saved in {os.path.abspath(out_folder)}')
    else:
        raise ValueError(f'Unknown output type {args.output}')


if __name__ == '__main__':
    main()