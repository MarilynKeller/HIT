import torch
from torch.autograd import grad


def tensor_linspace(start, end, steps=10):
    # source  https://github.com/zhaobozb/layout2im/blob/master/models/bilinear.py#L246
        #!/usr/bin/python
    #
    # Copyright 2018 Google LLC
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #      http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    """
    Vectorized version of torch.linspace.
    Inputs:
    - start: Tensor of any shape
    - end: Tensor of the same shape as start
    - steps: Integer
    Returns:
    - out: Tensor of shape start.size() + (steps,), such that
      out.select(-1, 0) == start, out.select(-1, -1) == end,
      and the other elements of out linearly interpolate between
      start and end.
    """
    assert start.size() == end.size()
    view_size = start.size() + (1,)
    w_size = (1,) * start.dim() + (steps,)
    out_size = start.size() + (steps,)

    start_w = torch.linspace(1, 0, steps=steps).to(start)
    start_w = start_w.view(w_size).expand(out_size)
    end_w = torch.linspace(0, 1, steps=steps).to(start)
    end_w = end_w.view(w_size).expand(out_size)

    start = start.contiguous().view(view_size).expand(out_size)
    end = end.contiguous().view(view_size).expand(out_size)

    out = start_w * start + end_w * end
    return out
  



        
def cond_create(betas, body_pose=None, generator=None, smpl=None):
    
    assert betas.shape[1] == 10
    
    if body_pose is not None:
      assert body_pose.shape[1] == 69
    
    B = betas.shape[0]
    cond = {}

    if body_pose is not None:
        cond['thetas'] = body_pose/torch.pi
        if False:
            pose_feature = smpl.pose2features(body_pose)
            cond['thetas'] = pose_feature
      
    cond['betas'] = betas/10.
    cond['betas'][:,0] = -cond['betas'][:,0] # This is just to leverage the pretrained neutral smpl model, should be removed eventually
    
    z_shape = torch.zeros((B, 64)).cuda()  
    cond['lbs'] = z_shape
    if generator is not None:           
      cond['latent'] = generator(z_shape)
    else:
      cond['latent'] = None
    return cond
  

def gradient(inputs, outputs, create_graph=True, retain_graph=True):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=create_graph,
        retain_graph=retain_graph,
        only_inputs=True)[0]#[:, -3:]
    return points_grad
  

def _eikonal_loss(nonmnfld_grad, mnfld_grad, eikonal_type='abs'):
    # Compute the eikonal loss that penalises when ||grad(f)|| != 1 for points on and off the manifold
    # shape is (bs, num_points, dim=3) for both grads
    # Eikonal 
    if nonmnfld_grad is not None and mnfld_grad is not None:
        all_grads = torch.cat([nonmnfld_grad, mnfld_grad], dim=-2)
    elif nonmnfld_grad is not None:
        all_grads = nonmnfld_grad
    elif mnfld_grad is not None:
        all_grads = mnfld_grad
    
    if eikonal_type == 'abs':
        eikonal_term = ((all_grads.norm(2, dim=2) - 1).abs()).mean()
    else:
        eikonal_term = ((all_grads.norm(2, dim=2) - 1).square()).mean()
    
    return eikonal_term


def eik_loss(nonmnfld_pred, nonmnfld_points):
    
    nonmnfld_grad = gradient(nonmnfld_points, nonmnfld_pred)
    eikonal_term = _eikonal_loss(nonmnfld_grad, mnfld_grad=None, eikonal_type='abs')
    return eikonal_term
    
