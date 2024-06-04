"""
Inspired from https://github.com/samsad35/VQ-MAE-S-code/blob/main/vqmae/vqmae/model/speech/Vector_Quantizer_EMA.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizerEMA(nn.Module):
    def __init__(
        self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5
    ):
        """Initialize a vector quantizer

        Args:
            num_embeddings (int): Number of embeddings in the dictionary.
            embedding_dim (int): Dimension of each embedding in the dictionary.
            commitment_cost (float): Weight of the commitment loss for the VQ-VAE.
            decay (_type_): Decay for the moving averages used to train the codebook.
            epsilon (_type_, optional): Used for numerical stability. Defaults to 1e-5.
        """
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(
        self,
        inputs,
        detailed=False,
        return_indices=False,
    ):
        """Forward the quantization.

        Args:
            inputs (torch.Tensor): The continuous latent representation.
            detailed (bool, optional): Not used for a single quantizer but will be useful for RQ-VAEs. Defaults to False.
            return_indices (bool, optional): If True, returns the quantized latent representation and the corresponding indices. Defaults to False.

        Returns:
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor: loss is the loss associated with the encoding-decoding process, x_recon are the reconstructed meshes,
            and perplexity gives information about the codebook usage.
        """
        # convert inputs from BCL -> BLC
        inputs = inputs.contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1)
        encoding_indices_return = (
            encoding_indices.clone().detach().view(input_shape[0], -1)
        )
        encoding_indices = encoding_indices.unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (
                1 - self._decay
            ) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon)
                * n
            )

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(
                self._ema_w * self._decay + (1 - self._decay) * dw
            )

            self._embedding.weight = nn.Parameter(
                self._ema_w / self._ema_cluster_size.unsqueeze(1)
            )

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        if return_indices:
            return quantized.contiguous(), encoding_indices_return

        return loss, quantized.contiguous(), perplexity, encodings

    def get_codebook_indices(self, input):
        """Get indices from the continuous latent representation.

        Args:
            input (torch.Tensor): The continuous latent representation.

        Returns:
            torch.Tensor: The quantized latent representation.
        """
        inputs = input.contiguous()
        flat_input = inputs.view(-1, self._embedding_dim)
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight.to(flat_input) ** 2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.to(flat_input).t())
        )
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        return encoding_indices.view(inputs.shape[0], -1)

    def quantify(self, encoding_indices):
        """Get quantized latent representation from the indices.

        Args:
            encoding_indices (torch.Tensor): The indices of the embeddings to select in the dictionary.

        Returns:
            torch.Tensor: The quantized latent representation.
        """
        mesh_embeds = self._embedding(encoding_indices)
        return mesh_embeds


class RecursiveVectorQuantizerEMA(nn.Module):
    def __init__(
        self,
        num_quantizers,
        num_embeddings,
        embedding_dim,
        commitment_cost,
        decay,
        epsilon=1e-5,
        shared_codebook=False,
    ):
        """Initialize a RQ-VAE.

        Args:
            num_quantizers (int): Number of quantizations in the latent space.
            num_embeddings (int): Number of embeddings in each dictionary (is the same for all dictionaries).
            embedding_dim (int): Dimension of embeddings.
            commitment_cost (float): Weight of the commitment loss.
            decay (float): Decay for the moving averages.
            epsilon (float, optional): Used for numerical stability. Defaults to 1e-5.
            shared_codebook (bool, optional): If True, use the same codebook for all quantizers. Defaults to False.
        """
        super(RecursiveVectorQuantizerEMA, self).__init__()

        self.num_quantizers = num_quantizers
        self._embedding_dim = embedding_dim
        if shared_codebook:
            codebook = VectorQuantizerEMA(
                num_embeddings, embedding_dim, commitment_cost, decay, epsilon
            )
            self.layers = nn.ModuleList([codebook for _ in range(num_quantizers)])
        else:
            self.layers = nn.ModuleList(
                [
                    VectorQuantizerEMA(
                        num_embeddings, embedding_dim, commitment_cost, decay, epsilon
                    )
                    for _ in range(num_quantizers)
                ]
            )

    def forward(self, inputs, detailed=False):
        """Forward the succesive quantizations.

        Args:
            inputs (torch.Tensor): The continuous latent representation.
            detailed (bool, optional): If detailed, outputs all intermediate latent representations. Defaults to False.

        Returns:
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor: loss is the loss associated with the encoding-decoding process, x_recon are the reconstructed meshes,
            and perplexity gives information about the codebook usage.
        """
        quantized_out = torch.zeros_like(inputs)
        residuals = inputs

        all_losses = []
        all_perplexities = []
        all_encodings = []
        all_quantized = []

        for layer in self.layers:
            loss, quantized, perplexity, encodings = layer(residuals)
            residuals = residuals - quantized.detach()
            quantized_out = quantized_out + quantized

            all_losses.append(loss)
            all_perplexities.append(perplexity)
            all_encodings.append(encodings)
            if detailed:
                all_quantized.append(quantized_out)

        if detailed:
            return all_losses, all_quantized, all_perplexities, all_encodings
        return (
            sum(all_losses),
            quantized_out.contiguous(),
            sum(all_perplexities),
            all_encodings,
        )

    def quantify(self, encoding_indices):
        """Get quantized latent representation from the indices.

        Args:
            encoding_indices (torch.Tensor): The indices of the embeddings to select in the dictionary.

        Returns:
            torch.Tensor: The quantized latent representation.
        """
        all_mesh_embeds = []

        for i, layer in enumerate(self.layers):
            mesh_embeds = layer._embedding(encoding_indices[:, :, i])
            all_mesh_embeds.append(mesh_embeds.unsqueeze(2))
        return torch.cat(all_mesh_embeds, dim=2)

    def get_codebook_indices(self, inputs):
        """Get indices from the continuous latent representation.

        Args:
            input (torch.Tensor): The continuous latent representation.

        Returns:
            torch.Tensor: The quantized latent representation.
        """
        residuals = inputs

        all_indices = []

        for layer in self.layers:
            quantized, indices = layer(residuals, return_indices=True)
            residuals = residuals - quantized.detach()
            all_indices.append(indices.unsqueeze(-1))

        return torch.cat(all_indices, dim=-1)
