"""
This dataset is used for training Mesh-VQ-VAE. It selects a random mesh.
"""

from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import os
from ..utils.body_model import BodyModel, SMPLH_PATH, NUM_BETAS
import torch


class DatasetMeshFromSmpl(Dataset):
    def __init__(
        self,
        folder: str,
        structure: str = "**",
        normalize: bool = True,
        canonical: bool = True,
        finetune_folder: str = None,
        pose_hand: bool = False,
    ):
        """Initialize the dataset

        Args:
            folder (str): Folder with all the npz files containing motion capture data
            structure (str, optional): Used to get all the subfolders. Defaults to "**".
            normalize (bool, optional): Translates the meshes to 0. Defaults to True.
            canonical (bool, optional): Set the global orientation to 0 (facing the camera). Defaults to True.
            finetune_folder (str, optional): A folder with meshes for finetuning the Mesh-VQ-VAE. Defaults to None.
            pose_hand (bool, optional): If True, the SMPL hand pose parameter will be given in the npz files. Defaults to False.
        """
        super().__init__()
        paths = [
            p
            for p in Path(f"{folder}").glob(f"{structure}/*.npz")
            if "shape" not in str(p)
        ]
        finetune_paths = []
        self.finetune = False
        if finetune_folder is not None:
            self.finetune = True
            finetune_paths = [
                p for p in Path(f"{finetune_folder}").glob(f"{structure}/*.npz")
            ]
        self.normalize = normalize
        self.canonical = canonical
        self.pose_hand = pose_hand
        self.len_list, self.len_list_finetune = self.load(paths, finetune_paths)
        self.paths = list(self.len_list.keys())
        self.finetune_paths = list(self.len_list_finetune.keys())
        self.device = "cpu"
        male_bm_path = os.path.join(SMPLH_PATH, "m/model.npz")
        female_bm_path = os.path.join(SMPLH_PATH, "f/model.npz")
        self.male_bm = BodyModel(bm_path=male_bm_path, num_betas=10).to(self.device)
        self.female_bm = BodyModel(bm_path=female_bm_path, num_betas=10).to(self.device)

    def load(self, paths, finetune_paths):
        """_summary_

        Args:
            paths (_type_): List of all npz files for training.
            finetune_paths (_type_): List of all npz files for finetuning.

        Returns:
            dict, dict: Dictionaries with the length of each training and finetuning motion.
        """
        len_list = dict()
        len_list_finetune = dict()
        for seq in paths:
            bdata = np.load(seq)
            try:
                seq_len = bdata["poses"].shape[0]
                len_list[seq] = seq_len
            except:
                continue
        for seq in finetune_paths:
            bdata = np.load(seq)
            try:
                seq_len = bdata["root_orient"].shape[0]
                len_list_finetune[seq] = seq_len
            except:
                continue
        return len_list, len_list_finetune

    def get_mesh(
        self, gender, pose_body, betas, pose_hand=None, root_orient=None, trans=None
    ):
        """Get the mesh given the SMPL parameters.

        Args:
            gender (str): "m" if male, "f" if female.
            pose_body (np.array): The body pose in SMPL format.
            betas (np.array): The SMPL body shape parameter.
            pose_hand (np.array, optional): The SMPL hand pose parameter. Defaults to None.
            root_orient (np.array, optional): Global orientation. Defaults to None.
            trans (np.array, optional): Global translation. Defaults to None.

        Returns:
            np.array: The 3D coordinates for the 6890 vertices of the SMPL mesh.
        """
        gender = str(gender)
        if gender == "m":
            bm = self.male_bm
        else:
            bm = self.female_bm

        pose_body = torch.Tensor(pose_body).to(self.device)
        betas = torch.Tensor(betas[:NUM_BETAS][np.newaxis]).to(self.device)
        if pose_hand is not None:
            pose_hand = torch.Tensor(pose_hand).to(self.device)
        if root_orient is not None:
            root_orient = torch.Tensor(root_orient).to(self.device)
        if trans is not None:
            trans = torch.Tensor(trans).to(self.device)
        body = bm(
            pose_body=pose_body,
            pose_hand=pose_hand,
            betas=betas,
            root_orient=root_orient,
            trans=trans,
        )
        return body.v.squeeze(0).clone().detach().cpu().numpy()

    def __len__(self):
        return max(len(self.paths), len(self.finetune_paths))

    def __getitem__(self, index):
        """Get a human mesh.

        Args:
            index (_type_): The index of the npz path.

        Returns:
            np.array: A 3D mesh with 6890 vertices in 3 dimensions.
        """
        p = np.random.rand()
        if self.finetune and p > 0.5:
            path = self.finetune_paths[index % len(self.finetune_paths)]
            idx = np.random.randint(0, self.len_list_finetune[path])
        else:
            path = self.paths[index % len(self.paths)]
            idx = np.random.randint(0, self.len_list[path])
        bdata = np.load(path)
        gender = np.array(bdata["gender"], ndmin=1)[0]
        gender = str(gender, "utf-8") if isinstance(gender, bytes) else str(gender)
        if path not in self.finetune_paths:
            if self.canonical:
                root_orient = None
            else:
                root_orient = bdata["poses"][idx : idx + 1, :3]
            pose_body = bdata["poses"][idx : idx + 1, 3:66]
            if self.pose_hand:
                pose_hand = bdata["poses"][idx : idx + 1, 66:]
            else:
                pose_hand = None
        else:
            if self.canonical:
                root_orient = None
            else:
                root_orient = bdata["root_orient"][idx : idx + 1]
            pose_body = bdata["pose_body"][idx : idx + 1]
            pose_hand = None

        betas = bdata["betas"][:10]
        mesh = self.get_mesh(
            gender, pose_body, betas, root_orient=root_orient, pose_hand=pose_hand
        )
        if self.normalize:
            mesh = mesh - np.mean(mesh, axis=0, keepdims=True)
        return mesh
