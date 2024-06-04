"""
Implements the Mesh-VQ-VAE model.
"""

import torch
import torch.nn as nn
from .Encoder_Decoder import Encoder, Decoder

from .Vector_Quantizer_EMA import VectorQuantizerEMA, RecursiveVectorQuantizerEMA


class MeshVQVAE(nn.Module):
    def __init__(
        self,
        fully_conv_ae,
        num_embeddings=512,
        embedding_dim=9,
        commitment_cost=0.25,
        decay=0,
        num_quantizer=1,
        shared_codebook=False,
    ):
        """Initialize a Mesh-VQ-VAE

        Args:
            fully_conv_ae (FullyConvAE): A fully convolutional mesh autoencoder.
            num_embeddings (int, optional): The number of embeddings in the dictionary. Defaults to 512.
            embedding_dim (int, optional): The dimension of each embedding. Defaults to 9.
            commitment_cost (float, optional): The weight for the commitment loss in training the VQ-VAE. Defaults to 0.25.
            decay (int, optional): Decay for the moving averages. Defaults to 0.
            num_quantizer (int, optional): Allows to implement a RQ-VAE with multiple quantizers. Defaults to 1.
            shared_codebook (bool, optional): In the case of RQ-VAE, shares the codebook among quantizations. Defaults to False.
        """
        super(MeshVQVAE, self).__init__()

        self.encoder = Encoder(fully_conv_ae)
        self.num_quantizer = num_quantizer
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        if num_quantizer == 0:
            self.quantize = False
        elif num_quantizer == 1:
            self.vq_vae = VectorQuantizerEMA(
                num_embeddings, embedding_dim, commitment_cost, decay
            )
            self.quantize = True
        else:
            self.vq_vae = RecursiveVectorQuantizerEMA(
                num_quantizer,
                num_embeddings,
                embedding_dim,
                commitment_cost,
                decay,
                shared_codebook=shared_codebook,
            )
            self.quantize = True

        self.decoder = Decoder(fully_conv_ae)

    def forward(self, x, detailed=False):
        """Encoding and decoding the input meshes x.

        Args:
            x (torch.Tensor): Batch of input meshes in a tensor of shape (B, 6890, 3)
            detailed (bool, optional): If detailed is True, we output all the intermediate meshes in the RQ-VAE. Defaults to False.

        Returns:
            torch.Tensor, torch.Tensor, torch.Tensor: loss is the loss associated with the encoding-decoding process, x_recon are the reconstructed meshes,
            and perplexity gives information about the codebook usage.
        """
        z = self.encoder(x)
        if self.quantize:
            loss, quantized, perplexity, _ = self.vq_vae(z, detailed=detailed)
            if not detailed:
                x_recon = self.decoder(quantized)
            else:
                x_recon = []
                for quantized_int in quantized:
                    x_recon.append(self.decoder(quantized_int))
        else:
            loss = torch.zeros(1).to(z)
            perplexity = torch.zeros(1).to(z)
            x_recon = self.decoder(z)
        return loss, x_recon, perplexity

    def load(self, path_model: str):
        """Load the Mesh-VQ-VAE model

        Args:
            path_model (str): Path of the checkpoint.
        """
        checkpoint = torch.load(path_model)
        self.load_state_dict(checkpoint["model"], strict=False)
        loss = checkpoint["loss"]
        print(f"\t [Mesh-VQ-VAE is loaded successfully with loss = {loss}]")

    def encode(self, meshes):
        """Encode meshes

        Args:
            meshes (torch.Tensor): A batch of human meshes.

        Returns:
            torch.Tensor: The continuous latent representation if there is no quantization, the sequence of indices otherwise.
        """
        z = self.encoder(meshes)
        if self.quantize:
            z = self.vq_vae.get_codebook_indices(z)
        return z

    def get_codebook_indices(self, meshes):
        """Get indices from the meshes.

        Args:
            meshes (torch.Tensor): A batch of human meshes of dimension (B, 6890, 3)

        Returns:
            torch.Tensor: A tensor of shape (B, N) with N the number of indices used to represent a mesh.
        """
        z = self.encoder(meshes)
        indices = self.vq_vae.get_codebook_indices(z)
        return indices

    def decode(self, z):
        """Get the full mesh in 3D given the latent representation.

        Args:
            z (torch.Tensor): The latent representation in the form of indices if this is a VQ-VAE, and continuous otherwise.

        Returns:
            torch.Tensor: Meshes in a tensor of shape (B, 6890, 3).
        """
        if self.quantize:
            all_mesh_embeds = self.vq_vae.quantify(z)
            if self.num_quantizer != 1:
                all_mesh_embeds = torch.sum(all_mesh_embeds, dim=-2)
        else:
            all_mesh_embeds = z
        meshes = self.decoder(all_mesh_embeds)
        return meshes
