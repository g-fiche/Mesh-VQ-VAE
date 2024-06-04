"""
Used to keep track of losses along training.
"""

from datetime import datetime
import pandas
import torch
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import shutil


class Follow:
    def __init__(self, name: str, dir_save: str = ""):
        self.name = name
        self.datatime_start = datetime.today()
        self.dir_save = dir_save
        self.create_directory()
        self.table = {
            "epoch": [],
            "loss_train": [],
            "loss_validation": [],
            "perplexity_train": [],
            "v2v_train": [],
            "v2v_validation": [],
        }
        self.best_loss = 1e8
        self.best_v2v = 1e8

    def create_directory(self):
        dir_sav = Path(self.dir_save) / self.name.upper()
        dir_sav.mkdir(exist_ok=True)
        to_day = (
            str(self.datatime_start.date().year)
            + "-"
            + str(self.datatime_start.date().month)
            + "-"
            + str(self.datatime_start.date().day)
        )
        time = (
            str(self.datatime_start.time().hour)
            + "-"
            + str(self.datatime_start.time().minute)
        )
        path_date = dir_sav / to_day
        path_date.mkdir(exist_ok=True)
        path_time = path_date / time
        path_time.mkdir(exist_ok=True)
        self.path = path_time
        shutil.copytree("config_autoencoder/", path_time / "configs")
        shutil.copytree("mesh_vq_vae", path_time / "lib")
        path_sample = path_time / "samples"
        self.path_samples = path_sample
        path_sample.mkdir(exist_ok=True)

    def find_best_model(self, loss_validation):
        if loss_validation <= self.best_loss:
            self.best_loss = loss_validation
            return True
        else:
            return False

    def find_best_v2v(self, v2v):
        if v2v <= self.best_v2v:
            self.best_v2v = v2v
            return True
        else:
            return False

    def save_model(
        self,
        best_model: bool,
        best_v2v: bool,
        parameters: dict,
        epoch: int,
        every_step: int = 10,
    ):
        if epoch % every_step == 0:
            torch.save(parameters, f"{self.path}/model_checkpoint")
            print(f"\t - Model saved: [loss:{parameters['loss']}]")
        if best_model:
            torch.save(parameters, f"{self.path}/model_best_loss")
            print(f"\t - Best Model saved: [loss:{parameters['loss']}]")
        if best_v2v:
            torch.save(parameters, f"{self.path}/model_best_v2v")
            print(f"\t - Best V2V saved: [loss:{parameters['v2v']}]")

    def push(
        self,
        epoch: int,
        loss_train: float,
        loss_validation: float,
        perplexity_train: float,
        v2v_train: float,
        v2v_validation: float,
    ):
        self.table["epoch"].append(epoch)
        self.table["loss_train"].append(loss_train)
        self.table["loss_validation"].append(loss_validation)
        self.table["perplexity_train"].append(perplexity_train)
        self.table["v2v_train"].append(v2v_train)
        self.table["v2v_validation"].append(v2v_validation)

    def save_csv(self):
        df = pandas.DataFrame(self.table)
        df.to_csv(path_or_buf=f"{self.path}/model_table.csv")

    def plot(self):
        plt.figure(figsize=(10, 10))
        plt.plot(self.table["epoch"], self.table["loss_train"], label="train")
        plt.plot(self.table["epoch"], self.table["loss_validation"], label="validation")
        plt.xlabel("Epochs")
        plt.ylabel("Loss (mean)")
        plt.savefig(f"{self.path}/loss.png")
        plt.legend()
        plt.close()

        plt.figure(figsize=(10, 10))
        plt.plot(self.table["epoch"], self.table["perplexity_train"], label="train")
        plt.xlabel("Epochs")
        plt.ylabel("Perplexity (mean)")
        plt.savefig(f"{self.path}/perplexity.png")
        plt.legend()
        plt.close()

        plt.figure(figsize=(10, 10))
        plt.plot(self.table["epoch"], self.table["v2v_train"], label="train")
        plt.plot(self.table["epoch"], self.table["v2v_validation"], label="validation")
        plt.xlabel("Epochs")
        plt.ylabel("Loss (mean)")
        plt.savefig(f"{self.path}/v2v.png")
        plt.legend()
        plt.close()

    def load_dict(self, path: str):
        a_file = open(f"{path}/table.pkl", "rb")
        self.table = pickle.load(a_file)

    def __call__(
        self,
        epoch: int,
        loss_train: float,
        loss_validation: float,
        perplexity_train: float,
        v2v_train: float,
        v2v_validation: float,
        parameters: dict,
    ):
        self.push(
            epoch,
            loss_train,
            loss_validation,
            perplexity_train,
            v2v_train,
            v2v_validation,
        )
        self.save_model(
            best_model=self.find_best_model(loss_validation),
            best_v2v=self.find_best_v2v(v2v_validation),
            parameters=parameters,
            epoch=epoch,
            every_step=2,
        )
        self.save_csv()
        self.plot()
