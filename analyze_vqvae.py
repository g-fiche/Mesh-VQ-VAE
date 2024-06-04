"""
Code for visualizations in the latent space such as seeing which indices are responsible for each body part.
"""

from mesh_vq_vae import (
    MeshVQVAE,
    FullyConvAE,
    DatasetMeshDisentangled,
    set_seed,
    plot_meshes,
    get_colors_from_meshes,
)
import hydra
from omegaconf import DictConfig
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-p", "--path", type=str, help="Path to save the output", default="analyze"
)

args = parser.parse_args()
path = args.path


@hydra.main(
    config_path="config_autoencoder",
    config_name="config_medium",
    version_base=None,
)
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())

    set_seed(0)

    device = torch.device("cuda")

    ref_bm_path = "body_models/smplh/neutral/model.npz"
    ref_bm = np.load(ref_bm_path)
    faces = torch.from_numpy(ref_bm["f"].astype(np.int32))

    """Data"""
    data_path = cfg.dataset
    data = DatasetMeshDisentangled(folder=data_path.train_path)
    dataloader = DataLoader(data, batch_size=cfg.train.batch, shuffle=True)

    """ Model """
    fully_conv_ae = FullyConvAE(cfg.modelconv, test_mode=False)
    mesh_vqvae = MeshVQVAE(fully_conv_ae, **cfg.model).to(device)
    pytorch_total_params = sum(
        p.numel() for p in mesh_vqvae.parameters() if p.requires_grad
    )
    mesh_vqvae.load(cfg.checkpoint)
    print(f"Mesh-VQ-VAE: {pytorch_total_params}")

    # This segmentation was determined manually by visualizing the variation induced by the random modification of each index.
    segmentation = dict()
    segmentation["head"] = [0, 1, 2, 3, 12, 29, 30, 31, 48]
    segmentation["left_arm"] = [8, 11, 13, 14, 15, 16, 17, 18, 19, 20, 23]
    segmentation["right_arm"] = [39, 40, 41, 42, 43, 44, 45, 46, 47, 49]
    segmentation["left_leg"] = [6, 7, 10, 26, 27, 28]
    segmentation["right_leg"] = [34, 36, 50, 51, 52, 53]
    segmentation["buste"] = [4, 5, 9, 21, 22, 24, 25, 32, 33, 35, 37, 38]

    os.mkdir(path)

    for sample in dataloader:
        # Get meshes and their latent representations in the form of index sequences (l) and continuous (z).
        x1 = sample["x1"].to(device)
        l1 = mesh_vqvae.get_codebook_indices(x1)
        z1 = mesh_vqvae.encoder(x1)
        x2 = sample["x2"].to(device)
        l2 = mesh_vqvae.get_codebook_indices(x2)
        x3 = sample["x3"].to(device)
        l3 = mesh_vqvae.get_codebook_indices(x3)
        z3 = mesh_vqvae.encoder(x3)
        plot_meshes(
            x1,
            faces,
            device=device,
            show=False,
            save=f"{path}/orig_1.png",
        )
        plot_meshes(
            x2,
            faces,
            device=device,
            show=False,
            save=f"{path}/orig_2.png",
        )
        plot_meshes(
            x3,
            faces,
            device=device,
            show=False,
            save=f"{path}/orig_3.png",
        )

        # Interpolation between meshes with different poses and shapes
        os.mkdir(os.path.join(path, "interpolation"))
        for i in range(20):
            z13 = ((19 - i) / 20) * z1 + ((i + 1) / 20) * z3
            l13 = mesh_vqvae.vq_vae.get_codebook_indices(z13)
            x13 = mesh_vqvae.decode(l13)
            plot_meshes(
                x13,
                faces,
                device=device,
                show=False,
                save=f"{path}/interpolation/{i}.png",
            )

        # Visualize the effect of modifiying randomly each index
        os.mkdir(os.path.join(path, "indices"))
        for part in range(54):
            l1_temp = l1.clone().detach()
            l1_temp[:, part] = torch.randint(0, 512, (1,)).to(device)
            x1_pred = mesh_vqvae.decode(l1_temp)
            x1_pred = x1_pred - torch.mean(x1_pred, axis=1, keepdims=True)
            colored_mesh = get_colors_from_meshes(
                x1,
                x1_pred,
                0,
                5,
            )
            plot_meshes(
                x1_pred,
                faces,
                colors=torch.from_numpy(colored_mesh).float(),
                device=device,
                show=False,
                save=f"{path}/indices/pred_{part}.png",
            )

        # Once the segmentation is defined (manually), check that modifying each body part does not have an impact on others
        os.mkdir(os.path.join(path, "parts"))
        for part, indices in segmentation.items():
            l1_temp = l1.clone().detach()
            l1_temp[:, indices] = torch.randint(0, 512, (len(indices),)).to(device)
            x1_pred = mesh_vqvae.decode(l1_temp)
            x1_pred = x1_pred - torch.mean(x1_pred, axis=1, keepdims=True)
            colored_mesh = get_colors_from_meshes(
                x1,
                x1_pred,
                0,
                5,
            )
            plot_meshes(
                x1_pred,
                faces,
                colors=torch.from_numpy(colored_mesh).float(),
                device=device,
                show=False,
                save=f"{path}/parts/pred_{part}.png",
            )

        # Exchange body parts between M1 and M2, which have the same body shape.
        os.mkdir(os.path.join(path, "exchange_pose"))
        for part, indices in segmentation.items():
            l1_temp = l1.clone().detach()
            l1_temp[:, indices] = l2[:, indices]
            x1_pred = mesh_vqvae.decode(l1_temp)
            x1_pred = x1_pred - torch.mean(x1_pred, axis=1, keepdims=True)
            colored_mesh = get_colors_from_meshes(
                x1,
                x1_pred,
                0,
                10,
            )
            plot_meshes(
                x1_pred,
                faces,
                colors=torch.from_numpy(colored_mesh).float(),
                device=device,
                show=False,
                save=f"{path}/exchange_pose/pred_{part}.png",
            )

        # Exchange body parts between meshes with the same pose.
        os.mkdir(os.path.join(path, "exchange_shape"))
        for part, indices in segmentation.items():
            l2_temp = l2.clone().detach()
            l2_temp[:, indices] = l3[:, indices]
            x2_pred = mesh_vqvae.decode(l2_temp)
            x2_pred = x2_pred - torch.mean(x2_pred, axis=1, keepdims=True)
            colored_mesh = get_colors_from_meshes(
                x2,
                x2_pred,
                0,
                10,
            )
            plot_meshes(
                x2_pred,
                faces,
                colors=torch.from_numpy(colored_mesh).float(),
                device=device,
                show=False,
                save=f"{path}/exchange_shape/pred_{part}.png",
            )
        break


if __name__ == "__main__":
    main()
