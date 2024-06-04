import torch
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
)
from pytorch3d.structures import Meshes

import torch
import numpy as np

import matplotlib.pyplot as plt

plt.switch_backend("agg")
from matplotlib.gridspec import GridSpec


def get_colors_from_diff_pc(diff_pc, min_error, max_error, color="coolwarm"):
    """
    Adapted from https://github.com/papagina/MeshConvolution/blob/master/code/GraphAE/graphAE_test.py
    Gives the color of each vertex given its positional error.
    """
    b, n = diff_pc.shape
    colors = np.zeros((b, n, 3))
    mix = (diff_pc - min_error) / (max_error - min_error)
    mix = np.clip(mix, 0, 1)  # point_num
    cmap = plt.cm.get_cmap(color)
    colors = cmap(mix)[:, :, 0:3]
    return colors


def get_colors_from_meshes(groundtruth_mesh, prediction, min_error, max_error):
    """
    Gives the color of each vertex given the prediction and the groundtruth mesh.
    """
    groundtruth_mesh = groundtruth_mesh - torch.mean(
        groundtruth_mesh, axis=1, keepdims=True
    )
    prediction = prediction - torch.mean(prediction, axis=1, keepdims=True)
    diff_mesh = (
        100 * torch.norm(groundtruth_mesh - prediction, dim=-1).detach().cpu().numpy()
    )
    colored_mesh = get_colors_from_diff_pc(
        diff_mesh,
        min_error,
        max_error,
    )
    return colored_mesh


def renderer(vertices, faces, device, colors=None, rot=True):
    """
    Adapted from Pose-NDF: https://github.com/garvita-tiwari/PoseNDF/blob/main/experiments/exp_utils.py
    Renders a mesh given the vertices and the mesh topology.
    Optionally we can app a color map with colors.
    For some datasets like 3DPW, the mesh needs to be rotated before rendering it.
    """
    R, T = look_at_view_transform(2.0, 0, 0)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    raster_settings = RasterizationSettings(
        image_size=256, blur_radius=0.0, faces_per_pixel=1, bin_size=-1
    )

    lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights),
    )
    # create mesh from vertices
    if colors is None:
        verts_rgb = torch.ones_like(vertices)  # (1, V, 3)
    else:
        verts_rgb = colors
    textures = TexturesVertex(verts_features=verts_rgb.to(device))

    if rot:
        Rx = (
            torch.from_numpy(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))
            .float()
            .to(device)
        )
        vertices = vertices.to(device).reshape(-1, 3) @ Rx
    vertices = vertices.reshape(-1, 6890, 3)

    vertices = vertices - torch.mean(vertices, axis=1, keepdims=True)

    meshes = Meshes(
        vertices,
        faces.unsqueeze(0).repeat(len(vertices), 1, 1).to(vertices),
        textures=textures,
    )
    images = renderer(meshes)
    return images[:, :, :, :-1]


def plot_meshes(
    meshes,
    faces,
    device,
    show: bool = True,
    save: str = None,
    rot: bool = False,
    colors=None,
):
    """
    Plot a set of meshes.
    """
    images = renderer(meshes, faces, device, rot=rot, colors=colors)
    fig = plt.figure(figsize=(10, 10))
    if len(meshes) == 16:
        nrows = 4
        ncols = 4
    elif len(meshes) == 4:
        nrows = 2
        ncols = 2
    else:
        ncols = len(meshes)
        nrows = 1

    gs = GridSpec(ncols=ncols, nrows=nrows)
    i = 0
    for line in range(nrows):
        for col in range(ncols):
            ax = fig.add_subplot(gs[line, col])
            if images[i].shape[0] == 1:
                ax.imshow(images[i][0, :, :].cpu().detach().numpy())
            else:
                ax.imshow(images[i].cpu().detach().numpy())
            plt.axis("off")
            i = i + 1
    if show:
        plt.show()
    if save is not None:
        plt.savefig(save)
        plt.close()
