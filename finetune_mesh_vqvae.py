"""
Code for training the Mesh-VQ-VAE.
"""

from mesh_vq_vae import (
    MeshVQVAE,
    MeshVQVAE_Train,
    FullyConvAE,
    DatasetMeshFromSmpl,
    DatasetMeshTest,
    set_seed,
)
import hydra
from omegaconf import DictConfig
import os
import numpy as np
import torch


@hydra.main(
    config_path="config_autoencoder",
    config_name="config_medium",
    version_base=None,
)
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())

    set_seed()

    ref_bm_path = "body_models/smplh/neutral/model.npz"
    ref_bm = np.load(ref_bm_path)

    """Data"""
    data_path = cfg.dataset
    train_data = DatasetMeshFromSmpl(
        folder=data_path.train_path, finetune_folder=data_path.finetune_path
    )
    val_data = DatasetMeshTest(dataset_file=data_path.val_path)

    """ Model """
    fully_conv_ae = FullyConvAE(cfg.modelconv, test_mode=False)
    mesh_vqvae = MeshVQVAE(fully_conv_ae, **cfg.model)
    mesh_vqvae.load(cfg.checkpoint)
    pytorch_total_params = sum(
        p.numel() for p in mesh_vqvae.parameters() if p.requires_grad
    )
    print(f"Mesh-VQ-VAE: {pytorch_total_params}")

    """ Training """
    pretrain_mesh_vqvae = MeshVQVAE_Train(
        mesh_vqvae,
        train_data,
        val_data,
        config_training=cfg.train,
        faces=torch.from_numpy(ref_bm["f"].astype(np.int32)),
    )
    pretrain_mesh_vqvae.fit()


if __name__ == "__main__":
    main()
