"""Examples of using pyrender for viewing and offscreen rendering.
"""
import math

import pyglet

pyglet.options['shadow_window'] = False
import os

import numpy as np
import trimesh
from pyrender import (DirectionalLight, Mesh, MetallicRoughnessMaterial, Node,
                      OffscreenRenderer, PerspectiveCamera, PointLight,
                      Primitive, RenderFlags, Scene, SpotLight, Viewer)


def render_mesh_from_mpimesh(mesh, **kwargs):
    
    mesh = trimesh.Trimesh(vertices=mesh.v, faces=mesh.f)
    return render_mesh_from_trimesh([mesh], **kwargs)

def render_mesh_from_path(mesh_path, **kwargs):
 
     # Fuze trimesh
    mesh = trimesh.load(mesh_path)
    # fuze_trimesh.visual.vertex_colors = [200, 200, 200, 255]
    # fuze_trimesh.visual.vertex_colors = [ 145 , 252, 255]
    mesh.visual.vertex_colors = [ 255 , 255, 255]
    if color is not None:
        # import ipdb; ipdb.set_trace()
        mesh.visual.vertex_colors = list((color*255).astype(np.uint8)) + [255]
    
    return render_mesh_from_trimesh([mesh], **kwargs)
    
    
def render_mesh_from_trimesh(mesh, **kwargs):
    mesh = Mesh.from_trimesh(mesh)
    return render_mesh([mesh],  **kwargs)

def render_meshes_from_trimesh(meshes, **kwargs):
    pyrender_meshes = []
    for mesh in meshes:
        pyrender_meshes.append(Mesh.from_trimesh(mesh))
    return render_mesh(pyrender_meshes,  **kwargs)


def render_mesh(meshes, fov=math.pi/3, im_shape=(100,100), cam_pose=None, color=None, transparent=True):

    #==============================================================================
    # Mesh creation
    #==============================================================================

    #------------------------------------------------------------------------------
    # Creating textured meshes from trimeshes
    #------------------------------------------------------------------------------

 

    # boxf_trimesh = trimesh.creation.box(extents=1.92*np.ones(3))
    # fuze_mesh = Mesh.from_trimesh(boxf_trimesh, smooth=False)

    #==============================================================================
    # Light creation
    #==============================================================================

    # key_light = DirectionalLight(color=np.ones(3), intensity=5.0)
    # fill_light = SpotLight(color=np.ones(3), intensity=10.0,
    #                 innerConeAngle=np.pi/16, outerConeAngle=np.pi/6)
    key_light = PointLight(color=np.array([1., 0.90, 0.85]), intensity=30.0)
    fill_light = PointLight(color=np.array([0.8, 0.9, 1]), intensity=5.0)
    rim_light = PointLight(color=np.array([1., 1., 1.]), intensity=50.0)

    #==============================================================================
    # Camera creation
    #==============================================================================

    # cam = OrthographicCamera(xmag=0.5*fov[1], ymag=0.5*fov[1])
    aspectRatio = im_shape[0]/im_shape[1]
    # aspectRatio = 1
    cam = PerspectiveCamera(yfov=fov, aspectRatio=aspectRatio)
    if cam_pose is None:
        cam_pose = np.array([
            [1.0,  0, 0, 0],
            [0.0, 1.0,           0.0,           0.0],
            [0.0,  0,       1, 3],
            [0.0,  0.0,           0.0,          1.0]
        ])

    #==============================================================================
    # Scene creation
    #==============================================================================

    scene = Scene(ambient_light=np.array([0.01, 0.01, 0.015, 1.0]),  bg_color=0 * np.array([1.0, 1.0, 1.0, 0.0]))

    #==============================================================================
    # Adding objects to the scene
    #==============================================================================

    #------------------------------------------------------------------------------
    # By manually creating nodes
    #------------------------------------------------------------------------------
    for mesh in meshes:
        node = Node(mesh=mesh, translation=np.array([0, 0, 0]))
        scene.add_node(node)

    key_light_node = Node(light=key_light, translation=2*np.array([-0.7,0.8,1]))
    scene.add_node(key_light_node)

    fill_light_node = Node(light=fill_light, translation=2*np.array([1, -0.5, 0.5]))
    scene.add_node(fill_light_node)

    rim_light_node = Node(light=rim_light, translation=2*np.array([0.3, 1,-1.5]))
    scene.add_node(rim_light_node) 

    # scene.set_pose(direc_l_node, np.array([-1,1,0])
    # direc_l_node.translation = np.array([-1,1,0])
    # key_light_node.translation = np.array([0, 0, 0]
    # spot_l_node = scene.add(spot_l, pose=cam_pose)


    # # #==============================================================================
    # # # Using the viewer with a pre-specified camera
    # # #==============================================================================
    cam_node = scene.add(cam, pose=cam_pose)
    # v = Viewer(scene)

    #==============================================================================
    # Rendering offscreen from that camera
    #==============================================================================

    r = OffscreenRenderer(viewport_width=im_shape[0], viewport_height=im_shape[1])
    if transparent:
        color, depth = r.render(scene, flags=RenderFlags.RGBA)
    else:
        color, depth = r.render(scene)

    return color




    r.delete()

if __name__ == "__main__":
    mesh_path = '/ps/project/rib_cage_breathing/Code/skeleton/datas/body_mesh/skel_mean.ply'
    color = render_mesh(mesh_path)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(color)
    plt.show()
