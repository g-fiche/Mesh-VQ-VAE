"""
This dataset is only used for visualizations in the latent space of Mesh-VQ-VAE. 
It outputs 3 meshes M1, M2, and M3. 
M1 and M2 have the same body shape; M2 and M3 have the same pose.
"""

from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import os
from ..utils.body_model import BodyModel, SMPLH_PATH, NUM_BETAS
import torch
import random


class DatasetMeshDisentangled(Dataset):
    def __init__(
        self,
        folder: str,
        structure: str = "**",
        normalize: bool = True,
        canonical: bool = True,
        pose_hand: bool = False,
    ):
        """Initialize the dataset

        Args:
            folder (str): Folder with all the npz files containing motion capture data
            structure (str, optional): Used to get all the subfolders. Defaults to "**".
            normalize (bool, optional): Translates the meshes to 0. Defaults to True.
            canonical (bool, optional): Set the global orientation to 0 (facing the camera). Defaults to True.
            pose_hand (bool, optional): If True, the SMPL hand pose parameter will be given in the npz files. Defaults to False.
        """
        super().__init__()

        paths = [p for p in Path(f"{folder}").glob(f"{structure}/*.npz")]

        self.normalize = normalize
        self.canonical = canonical
        self.pose_hand = pose_hand

        self.len_list, self.subject_list = self.load(paths)
        self.paths = list(self.len_list.keys())

        self.device = "cpu"

        male_bm_path = os.path.join(SMPLH_PATH, "m/model.npz")
        female_bm_path = os.path.join(SMPLH_PATH, "f/model.npz")
        self.male_bm = BodyModel(bm_path=male_bm_path, num_betas=10).to(self.device)
        self.female_bm = BodyModel(bm_path=female_bm_path, num_betas=10).to(self.device)

    def load(self, paths):
        """For each npz file, computes the number of frames and the subject.

        Args:
            paths (list): List of all npz files' paths.

        Returns:
            dict, dict: The number of frames (len_list) and the subject (subject_list) for each file.
        """
        len_list = dict()
        subject_list = dict()
        for seq in paths:
            bdata = np.load(seq)
            subject = self.get_subject(seq)
            if subject not in subject_list:
                subject_list[subject] = []
            try:
                seq_len = bdata["poses"].shape[0]
                len_list[seq] = seq_len
                subject_list[subject].append(seq)
            except:
                continue
        return len_list, subject_list

    def get_subject(self, path):
        """Get the subject from the path name. This suppose that the data is organized as described in the README.

        Args:
            path (str): Path of the npz file containing the motion data.

        Returns:
            str: A string with the subject name.
        """
        return "_".join(str(path).split("/")[-3].split("_")[:2])

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
        return len(self.paths)

    def __getitem__(self, index):
        """We output 3 meshes: M1 and M2 have the same body shape; M2 and M3 have the same pose.

        Args:
            index (int): The index of the npz path.
        Returns:
            dict: A dictionary with the 3 meshes.
        """
        # Choose a random mesh for M1
        path1 = self.paths[index % len(self.paths)]
        idx1 = np.random.randint(0, self.len_list[path1])
        subject1 = self.get_subject(path1)

        # Choose a random mesh with the same subject as M1
        path2 = random.choice(self.subject_list[subject1])
        idx2 = np.random.randint(0, self.len_list[path2])

        # Choose a random subject for M3 (we will only use the body shape)
        subject3 = random.choice(list(self.subject_list.keys()))
        path3 = random.choice(self.subject_list[subject3])
        idx3 = np.random.randint(0, self.len_list[path3])

        item = dict()
        for i, (path, idx) in enumerate(zip([path1, path2, path3], [idx1, idx2, idx3])):
            bdata = np.load(path)
            gender = np.array(bdata["gender"], ndmin=1)[0]
            gender = str(gender, "utf-8") if isinstance(gender, bytes) else str(gender)
            if self.canonical:
                root_orient = None
            else:
                root_orient = bdata["poses"][idx : idx + 1, :3]
            if not i == 2:
                pose_body = bdata["poses"][idx : idx + 1, 3:66]
                if self.pose_hand:
                    pose_hand = bdata["poses"][idx : idx + 1, 66:]
                else:
                    pose_hand = None

            betas = bdata["betas"][:10]
            mesh = self.get_mesh(
                gender, pose_body, betas, root_orient=root_orient, pose_hand=pose_hand
            )
            if self.normalize:
                mesh = mesh - np.mean(mesh, axis=0, keepdims=True)

            item[f"x{i+1}"] = mesh

        return item
