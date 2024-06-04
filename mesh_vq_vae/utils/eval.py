import torch


def dist3d(data, data_recon, reduction=True):
    """
    Computes the Euclidian distance between 2 sets of 3D points.
    Inputs are in the shape (b,N,3) where b is the batch size and N is the
    number of vertices which is 6890 for the SMPL full mesh.
    If reduction is True, this function returns the mean distance,
    otherwise the shape of the output is (b,N).
    """
    if reduction:
        return torch.norm(data - data_recon, dim=-1).mean()
    else:
        return torch.norm(data - data_recon, dim=-1).mean(dim=-1)


def batch_compute_similarity_transform_torch(S1, S2):
    """
    Borrowed from HMR: https://github.com/akanazawa/hmr/blob/master/src/benchmark/eval_util.py

    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.permute(0, 2, 1)
        S2 = S2.permute(0, 2, 1)
        transposed = True
    assert S2.shape[1] == S1.shape[1]

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, _, V = torch.svd(K)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0], 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0, 2, 1))))

    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0, 2, 1)))

    # 5. Recover scale.
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

    # 6. Recover translation.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    # 7. Error:
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

    if transposed:
        S1_hat = S1_hat.permute(0, 2, 1)

    return S1_hat


def v2v(gt_v, pred_v, reduction=True):
    """
    Computes the vertex-to-vertex error (V2V or PVE) between 2 sets of vertices.
    """
    pred_v = pred_v - torch.mean(pred_v, axis=1, keepdims=True)
    return dist3d(gt_v, pred_v, reduction)


def pa_v2v(gt_v, pred_v, reduction=True):
    """
    Computes the procruste-aligned vertex-to-vertex error (PA-V2V) between 2 sets of vertices.
    """
    pred_sym = batch_compute_similarity_transform_torch(pred_v, gt_v)
    return dist3d(gt_v, pred_sym, reduction)
