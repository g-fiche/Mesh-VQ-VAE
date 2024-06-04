from torch.utils.data import Dataset
import numpy as np
import torch
import os
import random
from ..utils.body_model import BodyModel, SMPLH_PATH, NUM_BETAS


class DatasetMeshTest(Dataset):
    def __init__(
        self,
        dataset_file: str,
        normalize: bool = True,
        canonical: bool = True,
        proportion: float = 1,
    ):
        """Initialize the dataset

        Args:
            dataset_file (str): A npz file with the SMPL parameters for test meshes.
            normalize (bool, optional): Translates the meshes to 0. Defaults to True.
            canonical (bool, optional): Set the global orientation to 0 (facing the camera). Defaults to True.
            proportion (float, optional): Allows to reduce the dataset size for faster testing. Defaults to 1.
        """
        super().__init__()

        self.data = np.load(dataset_file, allow_pickle=True)

        full_len = len(self.data["root_orient"])
        new_len = full_len
        sampled_indices = [x for x in range(full_len)]
        if proportion != 1:
            new_len = int(proportion * full_len)
            sampled_indices = random.sample(range(full_len), new_len)

        self.root_orient = self.data["root_orient"][sampled_indices]
        self.gender = self.data["gender"][sampled_indices]
        self.betas = self.data["betas"][sampled_indices]
        self.pose_body = self.data["pose_body"][sampled_indices]

        self.normalize = normalize
        self.canonical = canonical

        self.device = "cpu"

        male_bm_path = os.path.join(SMPLH_PATH, "m/model.npz")
        female_bm_path = os.path.join(SMPLH_PATH, "f/model.npz")
        self.male_bm = BodyModel(bm_path=male_bm_path, num_betas=10).to(self.device)
        self.female_bm = BodyModel(bm_path=female_bm_path, num_betas=10).to(self.device)

    def __len__(self):
        return len(self.gender)

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

    def __getitem__(self, index):
        """Get a human mesh.

        Args:
            index (_type_): The index of the npz path.

        Returns:
            np.array: A 3D mesh with 6890 vertices in 3 dimensions.
        """
        gender = self.gender[index]
        betas = self.betas[index][:10]
        pose_body = self.pose_body[index : index + 1]

        if self.canonical:
            root_orient = None
        else:
            root_orient = self.root_orient[index : index + 1]

        mesh = self.get_mesh(
            gender,
            pose_body,
            betas,
            root_orient=root_orient,
        )
        if self.normalize:
            mesh = mesh - np.mean(mesh, axis=0, keepdims=True)
        return mesh
