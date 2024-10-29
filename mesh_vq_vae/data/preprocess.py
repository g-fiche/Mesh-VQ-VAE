import os
from os import path as osp
from tqdm import tqdm
import pickle as pkl
import numpy as np


def pkl_to_npz(folder, set, save_path):
    sequences = [
        x.split(".")[0] for x in os.listdir(osp.join(folder, "sequenceFiles", set))
    ]

    for seq in tqdm(sequences):
        data_file = osp.join(folder, "sequenceFiles", set, seq + ".pkl")

        data = pkl.load(open(data_file, "rb"), encoding="latin1")

        num_people = len(data["poses"])

        for p_id in range(num_people):
            root_orient = data["poses"][p_id][:, :3]
            pose_body = data["poses"][p_id][:, 3:66]
            betas = data["betas"][p_id][:10]
            gender = data["genders"][p_id]

            np.savez(
                f"{save_path}/{seq}_{p_id}.npz",
                root_orient=root_orient,
                pose_body=pose_body,
                betas=betas,
                gender=gender,
            )

def pkl_to_single_npz(folder, set, save_path):
    sequences = [
        x.split(".")[0] for x in os.listdir(osp.join(folder, "sequenceFiles", set))
    ]

    for seq in tqdm(sequences):
        data_file = osp.join(folder, "sequenceFiles", set, seq + ".pkl")

        data = pkl.load(open(data_file, "rb"), encoding="latin1")

        num_people = len(data["poses"])

        for p_id in range(num_people):
            root_orient = data["poses"][p_id][:, :3]
            pose_body = data["poses"][p_id][:, 3:66]
            betas = data["betas"][p_id][:10]
            gender = data["genders"][p_id]

    np.savez(
        save_path,
        root_orient=root_orient,
        pose_body=pose_body,
        betas=betas,
        gender=gender,
    )


if __name__ == "__main__":
    pkl_to_npz(
        "datasets/3DPW",
        set="train",
        save_path="datasets/3DPW/train",
    )
    pkl_to_single_npz(
        "datasets/3DPW",
        set="validation",
        save_path="datasets/3DPW/3DPW_validation.npz",
    )
    pkl_to_single_npz(
        "datasets/3DPW",
        set="test",
        save_path="datasets/3DPW/3DPW_test.npz",
    )
