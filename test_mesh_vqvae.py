"""
Code for testing the Mesh-VQ-VAE in terms of reconstruction.
"""

from mesh_vq_vae import (
    MeshVQVAE,
    MeshVQVAE_Train,
    FullyConvAE,
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
    test_data = DatasetMeshTest(dataset_file=data_path.test_path)

    """ Model """
    fully_conv_ae = FullyConvAE(cfg.modelconv, test_mode=True)
    mesh_vqvae = MeshVQVAE(fully_conv_ae, **cfg.model)
    mesh_vqvae.load(cfg.checkpoint)
    fully_conv_ae.init_test_mode()

    """ Training """
    pretrain_mesh_vqvae = MeshVQVAE_Train(
        mesh_vqvae,
        test_data,
        test_data,
        config_training=cfg.train,
        faces=torch.from_numpy(ref_bm["f"].astype(np.int32)),
    )
    pretrain_mesh_vqvae.test()


if __name__ == "__main__":
    main()
