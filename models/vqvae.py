# based on https://raw.githubusercontent.com/bshall/VectorQuantizedVAE/master/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, RelaxedOneHotCategorical
import math
from .nde import Logistic


class VQEmbeddingEMA(nn.Module):
    def __init__(self, latent_dim, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.999, epsilon=1e-5):
        super(VQEmbeddingEMA, self).__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        embedding = torch.zeros(latent_dim, num_embeddings, embedding_dim)
        embedding.uniform_(-1/num_embeddings, 1/num_embeddings)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(latent_dim, num_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())

    def forward(self, x, return_qy=False):
        B, C, H, W = x.size()
        N, M, D = self.embedding.size()
        assert C == N * D

        x = x.view(B, N, D, H, W).permute(1, 0, 3, 4, 2)
        x_flat = x.detach().reshape(N, -1, D)

        distances = torch.baddbmm(torch.sum(self.embedding ** 2, dim=2).unsqueeze(1) +
                                  torch.sum(x_flat ** 2, dim=2, keepdim=True),
                                  x_flat, self.embedding.transpose(1, 2),
                                  alpha=-2.0, beta=1.0)

        indices = torch.argmin(distances, dim=-1)
        encodings = F.one_hot(indices, M).float()
        quantized = torch.gather(self.embedding, 1, indices.unsqueeze(-1).expand(-1, -1, D))
        quantized = quantized.view_as(x)

        if self.training:
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=1)

            n = torch.sum(self.ema_count, dim=-1, keepdim=True)
            self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n

            dw = torch.bmm(encodings.transpose(1, 2), x_flat)
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw

            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = self.commitment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()

        avg_probs = torch.mean(encodings, dim=1)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10), dim=-1))
        if return_qy == False:
            return quantized.permute(1, 0, 4, 2, 3).reshape(B, C, H, W), loss, perplexity.sum()
        else:
            return quantized.permute(1, 0, 4, 2, 3).reshape(B, C, H, W), loss, perplexity.sum(), indices


class VQEmbeddingGSSoft(nn.Module):
    def __init__(self, latent_dim, num_embeddings, embedding_dim):
        super(VQEmbeddingGSSoft, self).__init__()

        self.embedding = nn.Parameter(torch.Tensor(latent_dim, num_embeddings, embedding_dim))
        nn.init.uniform_(self.embedding, -1/num_embeddings, 1/num_embeddings)

    def forward(self, x, return_qy=False):
        B, C, H, W = x.size()
        N, M, D = self.embedding.size()
        assert C == N * D

        x = x.view(B, N, D, H, W).permute(1, 0, 3, 4, 2)
        x_flat = x.reshape(N, -1, D)

        distances = torch.baddbmm(torch.sum(self.embedding ** 2, dim=2).unsqueeze(1) +
                                  torch.sum(x_flat ** 2, dim=2, keepdim=True),
                                  x_flat, self.embedding.transpose(1, 2),
                                  alpha=-2.0, beta=1.0)
        distances = distances.view(N, B, H, W, M)

        dist = RelaxedOneHotCategorical(0.5, logits=-distances)
        if self.training:
            samples = dist.rsample().view(N, -1, M)
        else:
            samples = torch.argmax(dist.probs, dim=-1)
            samples = F.one_hot(samples, M).float()
            samples = samples.view(N, -1, M)

        quantized = torch.bmm(samples, self.embedding)
        quantized = quantized.view_as(x)

        KL = dist.probs * (dist.logits + math.log(M))
        KL[(dist.probs == 0).expand_as(KL)] = 0
        KL = KL.sum(dim=(0, 2, 3, 4)).mean()

        avg_probs = torch.mean(samples, dim=1)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10), dim=-1))

        if return_qy == False:
            return quantized.permute(1, 0, 4, 2, 3).reshape(B, C, H, W), KL, perplexity.sum()
        else:
            return quantized.permute(1, 0, 4, 2, 3).reshape(B, C, H, W), KL, perplexity.sum(), dist.probs[0]


class Residual(nn.Module):
    def __init__(self, channels):
        super(Residual, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)


class Encoder(nn.Module):
    def __init__(self, channels, latent_dim, embedding_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels),
            Residual(channels),
            Residual(channels),
            nn.Conv2d(channels, latent_dim * embedding_dim, 1)
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, channels, latent_dim, embedding_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim * embedding_dim, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            Residual(channels),
            Residual(channels),
            nn.ConvTranspose2d(channels, channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(channels, channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, 3, 1)
        )

    def forward(self, x, return_mean=False):
        x = self.decoder(x)
        return x


class VQVAE(nn.Module):
    def __init__(self, config):
        super(VQVAE, self).__init__()
        print('making VQ VAE')
        device = config.device
        channels = config.data.ngf
        latent_dim = 1
        num_embeddings = config.model.num_components
        embedding_dim = config.model.feature_size
        self.embedding_dim = embedding_dim
        self.n_channels = channels
        self.dataset = config.data.dataset
        self.image_size = config.data.image_size
        self.input_size = config.data.image_size ** 2 * config.data.channels
        self.dataset = config.data.dataset
        self.logit_transform = config.data.logit_transform

        self.encoder = Encoder(channels, latent_dim, embedding_dim)
        self.codebook = VQEmbeddingEMA(latent_dim, num_embeddings, embedding_dim)
        self.decoder = Decoder(channels, latent_dim, embedding_dim)

        if config.data.dataset == 'MNIST':
            self.decoder_dist = Bernoulli(device=device)
        else:
            self.decoder_dist = Logistic(device=device)

        self.register_parameter('dec_log_stdv', torch.nn.Parameter(torch.Tensor([0.])))

    def encode(self, x):
        if not self.logit_transform:
            x = x - 0.5
        return self.encoder(x.view(-1, self.n_channels, self.image_size, self.image_size))

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, return_z=False):
        z = self.encode(x)
        z_quantised, KL, perplexity = self.codebook(z)
        f = self.decode(z)
        if return_z == False:
            return f, KL
        else:
            return f, KL, z, z_quantised

    def compute_q_y_x(self, x, u=None):
        z = self.encode(x)
        z_quantised, KL, perplexity, q_y_x = self.codebook(z, return_qy=True)
        return q_y_x

    def elbo(self, x, u=None, terms=3):
        f, KL, z_emb, z_quantised = self.forward(x, True)
        if self.dataset == 'MNIST':
            decoder_params = [torch.sigmoid(f).view(-1, self.input_size)]
        else:
            decoder_params = [f.view(-1, self.input_size).clamp(min=-0.5 + 1. / 512., max=0.5 - 1. / 512.), self.dec_log_stdv]
        x = x.view(-1, self.input_size)
        if not self.logit_transform:
            if self.dataset != 'MNIST':
                x = x - 0.5

        # run standard VAE
        # term 1: compute log p(x|z), the MC estimate of E_{q(z|x)}[log p(x|z)] where z~q(z|x)
        log_prob_px_z = self.decoder_dist.log_pdf(x, *decoder_params)

        ELBO = log_prob_px_z.mean() - KL
        raw_ELBO = ELBO

        return ELBO, [log_prob_px_z.mean(), KL], z_emb


class GSSOFT(nn.Module):
    def __init__(self, config):
        super(GSSOFT, self).__init__()
        print('making relaxed VQ VAE')
        device = config.device
        channels = config.data.channels
        latent_dim = 1
        num_embeddings = config.model.num_components
        embedding_dim = config.model.feature_size
        self.embedding_dim = embedding_dim
        self.n_channels = channels
        self.dataset = config.data.dataset
        self.image_size = config.data.image_size
        self.input_size = config.data.image_size ** 2 * config.data.channels
        self.dataset = config.data.dataset
        self.logit_transform = config.data.logit_transform
        self.pass_through_z = config.model.pass_through_z

        if config.data.dataset == 'MNIST':
            self.decoder_dist = Bernoulli(device=device)
        else:
            self.decoder_dist = Logistic(device=device)

        self.encoder = Encoder(channels, latent_dim, embedding_dim)
        self.codebook = VQEmbeddingGSSoft(latent_dim, num_embeddings, embedding_dim)
        self.decoder = Decoder(channels, latent_dim, embedding_dim)

        self.register_parameter('dec_log_stdv', torch.nn.Parameter(torch.Tensor([0.])))

    def encode(self, x):
        if not self.logit_transform:
            x = x - 0.5
        return self.encoder(x.view(-1, self.n_channels, self.image_size, self.image_size))

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, return_z=False):
        z = self.encode(x)
        if self.pass_through_z == False:
            z_quantised, KL, perplexity = self.codebook(z)
            f = self.decode(z_quantised)
        else:
            KL = torch.tensor(0)
            z_quantised = z
            f = self.decode(z)
        if return_z == False:
            return f, KL
        else:
            return f, KL, z, z_quantised

    def compute_q_y_x(self, x, u=None):
        z = self.encode(x)
        z_quantised, KL, perplexity, q_y_x = self.codebook(z, return_qy=True)
        return q_y_x

    def elbo(self, x, u=None, terms=3):
        f, KL, z_emb, z_quantised = self.forward(x, True)
        if self.dataset == 'MNIST':
            decoder_params = [torch.sigmoid(f).view(-1, self.input_size)]
        else:
            decoder_params = [f.reshape(-1, self.input_size).clamp(min=-0.5 + 1. / 512., max=0.5 - 1. / 512.), self.dec_log_stdv]
        x = x.view(-1, self.input_size)
        if not self.logit_transform:
            if self.dataset != 'MNIST':
                x = x - 0.5

        # run standard VAE
        # term 1: compute log p(x|z), the MC estimate of E_{q(z|x)}[log p(x|z)] where z~q(z|x)
        log_prob_px_z = self.decoder_dist.log_pdf(x, *decoder_params)

        ELBO = log_prob_px_z.mean() - KL
        raw_ELBO = ELBO
        return ELBO, [log_prob_px_z.mean(), KL], z_emb
