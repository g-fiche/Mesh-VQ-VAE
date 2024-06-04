"""
Code for training Mesh-VQ-VAE
"""

from torch.utils.data import DataLoader, Dataset
import torch
import torch.optim as optim
from tqdm import tqdm
from ..base import Train

from .follow_up import Follow
from ..model import MeshVQVAE
from ..utils.mesh_render import plot_meshes, get_colors_from_meshes
from ..utils.eval import *
from statistics import mean
from loguru import logger
import os


class MeshVQVAE_Train(Train):
    def __init__(
        self,
        model: MeshVQVAE,
        training_data: Dataset,
        validation_data: Dataset,
        config_training: dict = None,
        faces=None,
    ):
        """Initialize the training

        Args:
            model (MeshVQVAE): The Mesh-VQ-VAE to be trained.
            training_data (Dataset): Training dataset.
            validation_data (Dataset): Validation dataset.
            config_training (dict, optional): Config file for the training. Defaults to None.
            faces (_type_, optional): Faces of the mesh used for rendering results. Defaults to None.
        """
        # Model
        self.model = model

        # Dataloader
        self.training_loader = DataLoader(
            training_data,
            batch_size=config_training["batch"],
            shuffle=True,
            pin_memory=True,
            num_workers=config_training["workers"],
            drop_last=True,
        )
        self.validation_loader = DataLoader(
            validation_data,
            batch_size=config_training["batch"],
            shuffle=True,
            pin_memory=True,
            num_workers=config_training["workers"],
            drop_last=True,
        )

        # Optimizer
        self.device = torch.device("cuda")
        self.model.to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=config_training["lr"], amsgrad=False
        )
        self.train_steps = config_training["train_steps"]
        self.eval_steps = config_training["eval_steps"]

        # Config
        self.config_training = config_training

        # Follow
        self.train_loss = []
        self.train_perplexity = []
        self.train_v2v = []
        self.validation_loss = []
        self.validation_v2v = []
        self.load_epoch = 0
        self.parameters = dict()
        self.follow = Follow("mesh_vqvae", dir_save="checkpoint")
        self.epochs = config_training["epochs"]

        # Faces of the mesh for rendering
        self.f = faces

    def one_epoch(self):
        pass

    def fit(self):
        logger.add(
            os.path.join(self.follow.path, "train.log"),
            level="INFO",
            colorize=False,
        )
        logger.info("Start to train the Mesh-VQ-VAE")
        self.model.train()
        num_training_updates = len(self.training_loader) * self.epochs
        num_validation_batches = self.eval_steps
        i = self.load_epoch
        while i < num_training_updates:
            for data in self.training_loader:
                data = data.to(self.device)
                self.optimizer.zero_grad()
                vq_loss, data_recon, perplexity = self.model(data)
                recon_error = v2v(data, data_recon)
                loss = recon_error + vq_loss
                loss.backward()
                self.train_loss.append(loss.item())
                self.train_perplexity.append(perplexity.item())
                self.train_v2v.append(1000 * recon_error.item())
                self.optimizer.step()
                i += 1
                if i % self.train_steps == 0:
                    with torch.no_grad():
                        j = 0
                        for data in self.validation_loader:
                            while j < num_validation_batches:
                                data = data.to(self.device)
                                # data_recon, vq_loss_val = self.model(data)
                                vq_loss_val, data_recon, perplexity_val = self.model(
                                    data
                                )
                                recon_error_val = v2v(data, data_recon)
                                val_loss = recon_error_val + vq_loss_val
                                self.validation_loss.append(val_loss.item())
                                self.validation_v2v.append(
                                    1000 * recon_error_val.item()
                                )
                                j += 1
                            break
                        logger.info(
                            f"In iter. {i}, average training v2v is {mean(self.train_v2v[-self.train_steps:]):.2f}"
                            f", average perplexity is {mean(self.train_perplexity[-self.train_steps:]):.2f}"
                            f", and average validation v2v is {mean(self.validation_v2v[-self.eval_steps:]):.2f}"
                        )
                        plot_meshes(
                            data_recon[:4],
                            self.f,
                            self.device,
                            show=False,
                            save=f"{self.follow.path_samples}/{int((i)/self.train_steps)}-reconstruction.png",
                        )
                        plot_meshes(
                            data[:4].to(self.device),
                            self.f,
                            self.device,
                            show=False,
                            save=f"{self.follow.path_samples}/{int((i)/self.train_steps)}-real.png",
                        )
                        colored_mesh = get_colors_from_meshes(
                            data.to(self.device),
                            data_recon,
                            0,
                            3,
                        )
                        plot_meshes(
                            data_recon[:4],
                            self.f,
                            self.device,
                            show=False,
                            save=f"{self.follow.path_samples}/{int((i)/self.train_steps)}-reconstruction_colored.png",
                            colors=torch.from_numpy(colored_mesh)[:4].float(),
                        )
                        self.parameters = dict(
                            model=self.model.state_dict(),
                            optimizer=self.optimizer.state_dict(),
                            epoch=i,
                            loss=mean(self.validation_loss[-self.eval_steps :]),
                            v2v=mean(self.validation_v2v[-self.eval_steps :]),
                        )
                        self.follow(
                            epoch=i,
                            loss_train=mean(self.train_loss[-self.train_steps :]),
                            loss_validation=mean(
                                self.validation_loss[-self.eval_steps :]
                            ),
                            perplexity_train=mean(
                                self.train_perplexity[-self.train_steps :]
                            ),
                            v2v_train=mean(self.train_v2v[-self.train_steps :]),
                            v2v_validation=mean(
                                self.validation_v2v[-self.eval_steps :]
                            ),
                            parameters=self.parameters,
                        )

    def test(self):
        logger.add(
            os.path.join(self.follow.path, "train.log"),
            level="INFO",
            colorize=False,
        )
        self.model.eval()
        recon_list = []
        paerr_list = []
        for data in tqdm(self.validation_loader):
            data = data.to(self.device)
            # data_recon, _ = self.model(data)
            vq_loss, data_recon, perplexity = self.model(data)
            recon_error = v2v(data, data_recon)
            recon_list.append(1000 * recon_error.item())
            pa_error = pa_v2v(data, data_recon)
            paerr_list.append(1000 * pa_error.item())

        logger.info(
            f"Average V2V error: {mean(recon_list)}, PA-V2V: {mean(paerr_list)}"
        )
        plot_meshes(
            data_recon[:4],
            self.f,
            self.device,
            show=False,
            save=f"{self.follow.path_samples}/reconstruction.png",
        )
        plot_meshes(
            data[:4],
            self.f,
            self.device,
            show=False,
            save=f"{self.follow.path_samples}/real.png",
        )
        colored_mesh = get_colors_from_meshes(
            data,
            data_recon,
            0,
            3,
        )
        plot_meshes(
            data_recon[:4],
            self.f,
            self.device,
            show=False,
            save=f"{self.follow.path_samples}/reconstruction_colored.png",
            colors=torch.from_numpy(colored_mesh)[:4].float(),
        )

    def test_detailed(self):
        logger.add(
            os.path.join(self.follow.path, "train.log"),
            level="INFO",
            colorize=False,
        )
        self.model.eval()
        recon_list = []
        paerr_list = []

        for t in range(self.model.num_quantizer):
            recon_list.append([])
            paerr_list.append([])

        for data in tqdm(self.validation_loader):
            data = data.to(self.device)
            # data_recon, _ = self.model(data)
            _, data_recon, _ = self.model(data, detailed=True)

            for t, data_recon_t in enumerate(data_recon):
                recon_error = v2v(data, data_recon_t)
                recon_list[t].append(1000 * recon_error.item())
                pa_error = pa_v2v(data, data_recon_t)
                paerr_list[t].append(1000 * pa_error.item())

        for t in range(self.model.num_quantizer):
            logger.info(
                f"Average V2V error {t}: {mean(recon_list[t])}, PA-V2V: {mean(paerr_list[t])}"
            )
            plot_meshes(
                data_recon[t][:4],
                self.f,
                self.device,
                show=False,
                save=f"{self.follow.path_samples}/{t}_reconstruction.png",
            )
            plot_meshes(
                data[:4],
                self.f,
                self.device,
                show=False,
                save=f"{self.follow.path_samples}/real.png",
            )
            colored_mesh = get_colors_from_meshes(
                data,
                data_recon[t],
                0,
                3,
            )
            plot_meshes(
                data_recon[t][:4],
                self.f,
                self.device,
                show=False,
                save=f"{self.follow.path_samples}/{t}_reconstruction_colored.png",
                colors=torch.from_numpy(colored_mesh)[:4].float(),
            )

    def load(self, path: str = "", optimizer: bool = True):
        print("LOAD [", end="")
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model"])
        if optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.load_epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]
        print(
            f"model: ok  | optimizer:{optimizer}  |  loss: {loss}  |  epoch: {self.load_epoch}]"
        )
