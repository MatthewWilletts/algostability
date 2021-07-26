from .nets import ConvMLP, FullMLP, SimpleLinear, ConvMLPencoder, DeconvMLPdecoder, FullMLPencoder, FullMLPdecoder, ResnetDecoder, ResnetEncoder
from numbers import Number
from typing import Sequence, Optional

import numpy as np
import torch
from torch import distributions as dist
from torch import nn
from torch.nn import functional as F


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)


class MixtureSameFamily(dist.Distribution):
    """ Mixture (same-family) distribution.

    The `MixtureSameFamily` distribution implements a (batch of) mixture
    distribution where all components are from different parameterizations of
    the same distribution type. It is parameterized by a `Categorical`
    "selecting distribution" (over `k` components) and a components
    distribution, i.e., a `Distribution` with a rightmost batch shape
    (equal to `[k]`) which indexes each (batch of) component.
    """

    def __init__(self,
                 mixture_distribution,
                 components_distribution,
                 validate_args=None):
        """ Construct a 'MixtureSameFamily' distribution

        Args::
            mixture_distribution: `torch.distributions.Categorical`-like
                instance. Manages the probability of selecting components.
                The number of categories must match the rightmost batch
                dimension of the `components_distribution`. Must have either
                scalar `batch_shape` or `batch_shape` matching
                `components_distribution.batch_shape[:-1]`
            components_distribution: `torch.distributions.Distribution`-like
                instance. Right-most batch dimension indexes components.

        Examples::
            # Construct Gaussian Mixture Model in 1D consisting of 5 equally
            # weighted normal distributions
            >>> mix = D.Categorical(torch.ones(5,))
            >>> comp = D.Normal(torch.randn(5,), torch.rand(5,))
            >>> gmm = MixtureSameFamily(mix, comp)

            # Construct Gaussian Mixture Modle in 2D consisting of 5 equally
            # weighted bivariate normal distributions
            >>> mix = D.Categorical(torch.ones(5,))
            >>> comp = D.Independent(D.Normal(
                    torch.randn(5,2), torch.rand(5,2)), 1)
            >>> gmm = MixtureSameFamily(mix, comp)

            # Construct a batch of 3 Gaussian Mixture Models in 2D each
            # consisting of 5 random weighted bivariate normal distributions
            >>> mix = D.Categorical(torch.rand(3,5))
            >>> comp = D.Independent(D.Normal(
                    torch.randn(3,5,2), torch.rand(3,5,2)), 1)
            >>> gmm = MixtureSameFamily(mix, comp)

        """
        self._mixture_distribution = mixture_distribution
        self._components_distribution = components_distribution

        if not isinstance(self._mixture_distribution, dist.Categorical):
            raise ValueError(" The Mixture distribution needs to be an "
                             " instance of torch.distribtutions.Categorical")

        if not isinstance(self._components_distribution, dist.Distribution):
            raise ValueError("The Component distribution need to be an "
                             "instance of torch.distributions.Distribution")

        # Check that batch size matches
        mdbs = self._mixture_distribution.batch_shape
        cdbs = self._components_distribution.batch_shape[:-1]
        if len(mdbs) != 0 and mdbs != cdbs:
            raise ValueError("`mixture_distribution.batch_shape` ({0}) is not "
                             "compatible with `components_distribution."
                             "batch_shape`({1})".format(mdbs, cdbs))

        # Check that the number of mixture components matches
        km = self._mixture_distribution.logits.shape[-1]
        kc = self._components_distribution.batch_shape[-1]
        if km is not None and kc is not None and km != kc:
            raise ValueError("`mixture_distribution components` ({0}) does not"
                             " equal `components_distribution.batch_shape[-1]`"
                             " ({1})".format(km, kc))
        self._num_components = km

        event_shape = self._components_distribution.event_shape
        self._event_ndims = len(event_shape)
        super(MixtureSameFamily, self).__init__(batch_shape=cdbs,
                                                event_shape=event_shape,
                                                validate_args=validate_args)

    @property
    def mixture_distribution(self):
        return self._mixture_distribution

    @property
    def components_distribution(self):
        return self._components_distribution

    @property
    def mean(self):
        probs = self._pad_mixture_dimensions(self.mixture_distribution.probs)
        return torch.sum(probs * self.components_distribution.mean,
                         dim=-1-self._event_ndims)  # [B, E]

    @property
    def variance(self):
        # Law of total variance: Var(Y) = E[Var(Y|X)] + Var(E[Y|X])
        probs = self._pad_mixture_dimensions(self.mixture_distribution.probs)
        mean_cond_var = torch.sum(probs*self.components_distribution.variance,
                                  dim=-1-self._event_ndims)
        var_cond_mean = torch.sum(probs * (self.components_distribution.mean -
                                           self._pad(self.mean)).pow(2.0),
                                  dim=-1-self._event_ndims)
        return mean_cond_var + var_cond_mean

    def log_prob(self, x):
        x = self._pad(x)
        log_prob_x = self.components_distribution.log_prob(x)  # [S, B, k]
        log_mix_prob = torch.log_softmax(self.mixture_distribution.logits,
                                         dim=-1)  # [B, k]
        return torch.logsumexp(log_prob_x + log_mix_prob, dim=-1)  # [S, B]

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            # [n, B]
            mix_sample = self.mixture_distribution.sample(sample_shape)
            # [n, B, k, E]
            comp_sample = self.components_distribution.sample(sample_shape)
            # [n, B, k]
            mask = F.one_hot(mix_sample, self._num_components)
            # [n, B, k, [1]*E]
            mask = self._pad_mixture_dimensions(mask)
            return torch.sum(comp_sample * mask.float(),
                             dim=-1-self._event_ndims)

    def _pad(self, x):
        d = len(x.shape) - self._event_ndims
        s = x.shape
        x = x.reshape(*s[:d], 1, *s[d:])
        return x

    def _pad_mixture_dimensions(self, x):
        dist_batch_ndims = self.batch_shape.numel()
        cat_batch_ndims = self.mixture_distribution.batch_shape.numel()
        pad_ndims = 0 if cat_batch_ndims == 1 else \
            dist_batch_ndims - cat_batch_ndims
        s = x.shape
        x = torch.reshape(x, shape=(*s[:-1], *(pad_ndims*[1]),
                                    *s[-1:], *(self._event_ndims*[1])))
        return x


class Dist:
    def __init__(self):
        pass

    def sample(self, *args):
        pass

    def log_pdf(self, *args, **kwargs):
        pass


class Bernoulli(Dist):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self._dist = dist.bernoulli.Bernoulli(0.5 * torch.ones(1).to(self.device))
        self.name = 'bernoulli'

    def sample(self, p):
        eps = self._dist.sample(p.size())
        return eps

    def log_pdf(self, x, f, reduce=True, param_shape=None):
        """compute the log-pdf of a laplace distribution with diagonal covariance"""
        if param_shape is not None:
            f = f.view(param_shape)
        lpdf = x * torch.log(f) + (1 - x) * torch.log(1 - f)
        if reduce:
            return lpdf.sum(dim=-1)
        else:
            return lpdf


class Normal(Dist):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.c = 2 * np.pi * torch.ones(1).to(self.device)
        self._dist = dist.normal.Normal(torch.zeros(1).to(self.device), torch.ones(1).to(self.device))
        self.name = 'gauss'

    def sample(self, mu, v):
        eps = self._dist.sample(mu.size()).squeeze()
        scaled = eps.mul(v.sqrt())
        return scaled.add(mu)

    def log_pdf(self, x, mu, v, reduce=True, param_shape=None):
        """compute the log-pdf of a normal distribution with diagonal covariance"""
        if param_shape is not None:
            mu, v = mu.view(param_shape), v.view(param_shape)
        lpdf = -0.5 * (torch.log(self.c) + v.log() + (x - mu).pow(2).div(v))
        if reduce:
            return lpdf.sum(dim=-1)
        else:
            return lpdf

    def log_pdf_full(self, x, mu, v):
        """
        compute the log-pdf of a normal distribution with full covariance
        v is a batch of "pseudo sqrt" of covariance matrices of shape (batch_size, d_latent, d_latent)
        mu is batch of means of shape (batch_size, d_latent)
        """
        batch_size, d = mu.size()
        cov = torch.einsum('bik,bjk->bij', v, v)  # compute batch cov from its "pseudo sqrt"
        assert cov.size() == (batch_size, d, d)
        inv_cov = torch.inverse(cov)  # works on batches
        c = d * torch.log(self.c)
        # matrix log det doesn't work on batches!
        _, logabsdets = self._batch_slogdet(cov)
        xmu = x - mu
        return -0.5 * (c + logabsdets + torch.einsum('bi,bij,bj->b', [xmu, inv_cov, xmu]))

    def _batch_slogdet(self, cov_batch: torch.Tensor):
        """
        compute the log of the absolute value of determinants for a batch of 2D matrices. Uses torch.slogdet
        this implementation is just a for loop, but that is what's suggested in torch forums
        gpu compatible
        """
        batch_size = cov_batch.size(0)
        signs = torch.empty(batch_size, requires_grad=False).to(self.device)
        logabsdets = torch.empty(batch_size, requires_grad=False).to(self.device)
        for i, cov in enumerate(cov_batch):
            signs[i], logabsdets[i] = torch.slogdet(cov)
        return signs, logabsdets


class Logistic(Dist):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.name = 'logistic'

    def log_pdf(self, x, mean, logscale, reduce=True, param_shape=None, binsize=1 / 256.0):
        # actually discretized logistic, but who cares
        scale = torch.exp(logscale)
        x = (torch.floor(x / binsize) * binsize - mean) / scale
        lpdf = torch.log(torch.sigmoid(x + binsize / scale) - torch.sigmoid(x) + 1e-7)
        if reduce:
            return lpdf.sum(dim=-1)
        else:
            return lpdf


# image MODELS

class VaDEConvMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        device = config.device
        self.latent_dim = config.model.feature_size
        self.image_size = config.data.image_size
        self.n_channels = config.data.channels
        self.ngf = ngf = config.model.ngf
        self.num_components = config.model.num_components
        self.learn_y_prior = config.model.learn_y_prior
        self.input_size = config.data.image_size ** 2 * config.data.channels
        self.dataset = config.data.dataset
        self.logit_transform = config.data.logit_transform

        if config.data.dataset == 'MNIST':
            self.decoder_dist = Bernoulli(device=device)
        else:
            self.decoder_dist = Logistic(device=device)

        self.make_networks(config)

        self.prior_dist = Normal(device=device)
        self.encoder_dist = Normal(device=device)

        self.pi_p_y = nn.Parameter(torch.ones(self.num_components) / self.num_components,
                                   requires_grad=self.learn_y_prior).to(device)
        self.mu_p_z = nn.Parameter(torch.zeros(self.latent_dim, self.num_components), requires_grad=True).to(device)
        nn.init.xavier_normal_(self.mu_p_z)

        self.log_sigma_square_p_z = nn.Parameter(torch.Tensor(self.latent_dim, self.num_components), requires_grad=True).to(device)
        nn.init.xavier_normal_(self.log_sigma_square_p_z)

        self.decoder_var = config.model.decoder_var * torch.ones(1).to(device)
        self.register_parameter('dec_log_stdv', torch.nn.Parameter(torch.Tensor([0])))

        self.good_dims = None

        # self.apply(weights_init)

        self.free_bits_z = None
        self.free_bits_y = None

    def make_networks(self, config):
        print('making ConvNets')
        self.encoder = ConvMLPencoder(config)
        self.decoder = DeconvMLPdecoder(config)
        print('encoder', self.encoder)
        print('decoder', self.decoder)

    def encode(self, x):
        if not self.logit_transform:
            x = x - 0.5
        return self.encoder(x.view(-1, self.n_channels, self.image_size, self.image_size))

    def decode(self, z):
        h = self.decoder(z)
        return h.view(-1, self.n_channels, self.image_size, self.image_size)

    def sample(self, n_sample=1, per_comp=False):
        if self.num_components == 1:
            z_samples = self.prior_dist._dist.sample((n_sample, self.latent_dim)).squeeze()
            f = self.decode(z_samples)
            if self.dataset == 'MNIST':
                return torch.sigmoid(f).view(-1, self.input_size)
            else:
                return f.view(-1, self.input_size)
        if per_comp == False:
            l_p_y = torch.log_softmax(self.pi_p_y, dim=-1)
            p_y = dist.OneHotCategorical(logits=l_p_y)
            y_samples = p_y.sample((n_sample,))

            loc = y_samples @ self.mu_p_z.permute(1,0)
            scale = y_samples @ torch.exp(0.5 * self.log_sigma_square_p_z).permute(1, 0)
            pz_y = dist.Normal(loc=loc, scale=scale)
            z_samples = pz_y.sample()
            f = self.decode(z_samples)
            if self.dataset == 'MNIST':
                return torch.sigmoid(f).view(-1, self.input_size)
            else:
                return f.view(-1, self.input_size)
        else:
            y_samples = torch.tensor(np.eye(self.num_components, dtype=np.float32))
            loc = y_samples @ self.mu_p_z.permute(1,0)
            scale = y_samples @ torch.exp(0.5 * self.log_sigma_square_p_z).permute(1, 0)
            pz_y = dist.Normal(loc=loc, scale=scale)
            z_samples = pz_y.sample((n_sample,))
            z_samples = z_samples.view(-1, self.latent_dim)
            f = self.decode(z_samples)
            if self.dataset == 'MNIST':
                return torch.sigmoid(f).view(n_sample, -1, self.input_size)
            else:
                return f.view(n_sample, -1, self.input_size)

    @staticmethod
    def reparameterize(mu, std):
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, return_z=False):
        mu, logv = self.encode(x)
        var = F.softplus(logv)
        std = var.sqrt()
        z = self.reparameterize(mu, std)
        f = self.decode(z)
        if return_z == False:
            return f, mu, var
        else:
            return f, mu, var, z

    def elbo(self, x, u=None):
        good_dims = self.good_dims
        f, mu, var, z = self.forward(x, True)
        if self.dataset == 'MNIST':
            if good_dims is None:
                decoder_params = [torch.sigmoid(f).view(-1, self.input_size)]
            else:
                decoder_params = [torch.sigmoid(f.view(-1, self.input_size)[:, good_dims])]
        else:
            decoder_params = [f.view(-1, self.input_size).clamp(min=-0.5 + 1. / 512., max=0.5 - 1. / 512.), self.dec_log_stdv]
        x = x.view(-1, self.input_size)
        g = mu
        v = var
        if not self.logit_transform:
            if self.dataset != 'MNIST':
                x = x - 0.5
        if self.dataset == 'MNIST' and good_dims is not None:
            x = x[:, good_dims]

        if self.num_components != 1:
            # calculate '3 term loss'
            # ie, no 'VaDE' trick where we marginalise out y in the gen. model
            pz_y = dist.Independent(dist.Normal(loc=self.mu_p_z.permute(1, 0),
                                                scale=torch.exp(0.5 * self.log_sigma_square_p_z).permute(1, 0)),
                                    1)

            p_y = dist.Categorical(logits=self.pi_p_y)

            p_z = MixtureSameFamily(p_y, pz_y)

            # term 1: compute log p(x|z), the MC estimate of E_{q(z,c|x)}[log p(x|z)] where z~q(z|x)
            log_prob_px_z = self.decoder_dist.log_pdf(x, *decoder_params)

            # term 2: compute the MC estimate of E_{q(z,c|x)}[log p(z)] where z~q(z|x)
            log_prob_pz = p_z.log_prob(z)

            # term 3: compute the MC estimate of E_{q(z,c|x)}[log q(z|x)] where z~q(z|x)
            log_prob_qz_x = self.encoder_dist.log_pdf(z, g, v)

            KL_z = log_prob_qz_x - log_prob_pz
            if self.free_bits_z is not None:
                KL_z = free_bits_kl(KL_z.unsqueeze(-1), self.free_bits_z)
            else:
                KL_z = KL_z.mean()

            ELBO = log_prob_px_z.mean() - KL_z

            raw_ELBO = (log_prob_px_z - log_prob_qz_x + log_prob_pz).mean()
        else:
            # run standard VAE
            # term 1: compute log p(x|z), the MC estimate of E_{q(z|x)}[log p(x|z)] where z~q(z|x)
            log_prob_px_z = self.decoder_dist.log_pdf(x, *decoder_params)

            # term 2: compute the MC estimate of E_{q(z|x)}[log q(z|x)] where z~q(z|x)
            log_prob_qz_x = self.encoder_dist.log_pdf(z, g, v)

            # term 2: compute the MC estimate of E_{q(z|x)}[log p(z)] where z~q(z|x)
            log_prob_pz = self.prior_dist.log_pdf(z, torch.zeros(1), torch.ones(1))

            KL_z = (log_prob_qz_x - log_prob_pz).mean()
            ELBO = (log_prob_px_z).mean() - KL_z

        return ELBO, [log_prob_px_z.mean(), KL_z], z

    def compute_q_y_x(self, x, u=None):
        f, mu, var, z = self.forward(x, True)
        if self.num_components != 1:
            # ie, no 'VaDE' trick where we marginalise out y in the gen. model
            pz_y = dist.Independent(dist.Normal(loc=self.mu_p_z.permute(1, 0),
                                                scale=torch.exp(0.5 * self.log_sigma_square_p_z).permute(1, 0)),
                                    1)

            z_pad = torch.unsqueeze(z, -2)
            l_pz_y = pz_y.log_prob(z_pad)
            l_p_y = torch.log_softmax(self.pi_p_y, dim=-1)

            q_y_x = torch.softmax(l_pz_y + l_p_y, dim=1)
        return q_y_x

    def lambda_values(self):
        means = self.mu_p_z
        inv_variances = torch.exp(-1.0 * self.log_sigma_square_p_z)
        return torch.cat((means * inv_variances, -0.5 * inv_variances))


class VaDEFullMLP(VaDEConvMLP):
    def __init__(self, config, good_dims):
        super().__init__(config)
        self.good_dims = good_dims

    def make_networks(self, config):
        print('making MLPs')
        self.encoder = FullMLPencoder(config)
        self.decoder = FullMLPdecoder(config)

    def encode(self, x):
        if not self.logit_transform:
            x = x - 0.5
        return self.encoder(x.view(-1, self.input_size))

    def sample(self, n_sample=1, per_comp=False):
        if self.num_components == 1:
            n_sample = n_sample ** 2
            z_samples = self.prior_dist._dist.sample((n_sample, self.latent_dim)).squeeze()
            f = self.decode(z_samples)
            if self.dataset == 'MNIST':
                f = torch.sigmoid(f).view(-1, self.input_size)
                output = torch.zeros(f.shape)
                output[:,self.good_dims] = f[:,self.good_dims]
                output -= 0.5
                return output
            else:
                return f.view(-1, self.input_size)
        if per_comp == False:
            l_p_y = torch.log_softmax(self.pi_p_y, dim=-1)
            p_y = dist.OneHotCategorical(logits=l_p_y)
            y_samples = p_y.sample((n_sample,))

            loc = y_samples @ self.mu_p_z.permute(1,0)
            scale = y_samples @ torch.exp(0.5 * self.log_sigma_square_p_z).permute(1, 0)
            pz_y = dist.Normal(loc=loc, scale=scale)
            z_samples = pz_y.sample()
            f = self.decode(z_samples)
            if self.dataset == 'MNIST':
                f = torch.sigmoid(f).view(-1, self.input_size)
                output = torch.zeros(f.shape)
                output[:,self.good_dims] = f[:,self.good_dims]
                output -= 0.5
                return output
            else:
                return f.view(-1, self.input_size)
        else:
            y_samples = torch.tensor(np.eye(self.num_components, dtype=np.float32))
            loc = y_samples @ self.mu_p_z.permute(1,0)
            scale = y_samples @ torch.exp(0.5 * self.log_sigma_square_p_z).permute(1, 0)
            pz_y = dist.Normal(loc=loc, scale=scale)
            z_samples = pz_y.sample((n_sample,))
            z_samples = z_samples.view(-1, self.latent_dim)
            f = self.decode(z_samples)
            if self.dataset == 'MNIST':
                f = torch.sigmoid(f).view(n_sample, -1, self.input_size)
                output = torch.zeros(f.shape)
                output[:,:,self.good_dims] = f[:,:,self.good_dims]
                output -= 0.5
                return output
            else:
                return f.view(n_sample, -1, self.input_size)


class VaDEResNetMLP(VaDEFullMLP):
    def __init__(self, config, good_dims):
        super().__init__(config, good_dims)

    def encode(self, x):
        if not self.logit_transform:
            x = x - 0.5
        return self.encoder(x.view(-1, self.n_channels, self.image_size, self.image_size))

    def make_networks(self, config):
        self.encoder = ResnetEncoder(config)
        self.decoder = ResnetDecoder(config)


class iVAEConvMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        device = config.device
        self.latent_dim = config.model.feature_size
        self.image_size = config.data.image_size
        self.n_channels = config.data.channels
        self.ngf = ngf = config.model.ngf
        self.num_components = 10
        self.learn_y_prior = config.model.learn_y_prior
        self.input_size = config.data.image_size ** 2 * config.data.channels
        self.dataset = config.data.dataset
        self.logit_transform = config.data.logit_transform

        if config.data.dataset == 'MNIST':
            self.decoder_dist = Bernoulli(device=device)
        else:
            self.decoder_dist = Logistic(device=device)

        self.make_networks(config)

        self.mu_p_z = nn.Parameter(torch.zeros(self.latent_dim, 10), requires_grad=True).to(device)
        nn.init.xavier_normal_(self.mu_p_z)

        self.log_sigma_square_p_z = nn.Parameter(torch.Tensor(self.latent_dim, 10), requires_grad=True).to(device)
        nn.init.xavier_normal_(self.log_sigma_square_p_z)

        self.prior_dist = Normal(device=device)
        self.encoder_dist = Normal(device=device)

        self.decoder_var = config.model.decoder_var * torch.ones(1).to(device)
        self.register_parameter('dec_log_stdv', torch.nn.Parameter(torch.Tensor([0.])))

        # self.apply(weights_init)

        self.free_bits_z = None
        self.free_bits_y = None

        self.good_dims = None

    def make_networks(self, config):
        print('making ConvNets')
        self.encoder = ConvMLPencoder(config)
        self.decoder = DeconvMLPdecoder(config)
        print('encoder', self.encoder)
        print('decoder', self.decoder)

    def encode(self, x):
        if not self.logit_transform:
            x = x - 0.5
        return self.encoder(x.view(-1, self.n_channels, self.image_size, self.image_size))

    def decode(self, z):
        h = self.decoder(z)
        return h.view(-1, self.n_channels, self.image_size, self.image_size)

    def prior(self, y):
        # h2 = F.relu(self.l1(y))
        # h2 = self.l1(y)
        return (self.mu_p_z @ y.T).T, (self.log_sigma_square_p_z @ y.T).exp().T

    # def prior(self, y):
    #     h2 = F.relu(self.l1(y))
    #     # h2 = self.l1(y)
    #     return self.l21(h2), self.l22(h2).exp()

    @staticmethod
    def reparameterize(mu, std):
        eps = torch.randn_like(std)
        return mu + eps * std

    def sample(self, n_sample=1, per_comp=False):
        y_samples = torch.tensor(np.eye(10, dtype=np.float32))
        loc = y_samples @ self.mu_p_z.permute(1,0)
        scale = y_samples @ torch.exp(self.log_sigma_square_p_z).permute(1, 0)
        pz_y = dist.Normal(loc=loc, scale=scale)
        z_samples = pz_y.sample((n_sample,))
        z_samples = z_samples.view(-1, self.latent_dim)
        f = self.decode(z_samples)
        if self.dataset == 'MNIST':
            return torch.sigmoid(f).view(n_sample, -1, self.input_size) - 0.5
        else:
            return f.view(n_sample, -1, self.input_size)

    def forward(self, x, return_z=False):
        mu, logv = self.encode(x)
        var = F.softplus(logv)
        std = var.sqrt()
        z = self.reparameterize(mu, std)
        f = self.decode(z)
        if return_z == False:
            return f, mu, var
        else:
            return f, mu, var, z

    def elbo(self, x, u, terms=3):
        good_dims = self.good_dims
        f, mu, var, z = self.forward(x, True)
        if self.dataset == 'MNIST':
            if good_dims is None:
                decoder_params = [torch.sigmoid(f).view(-1, self.input_size)]
            else:
                decoder_params = [torch.sigmoid(f.view(-1, self.input_size)[:, good_dims])]
        else:
            decoder_params = [f.view(-1, self.input_size).clamp(min=-0.5 + 1. / 512., max=0.5 - 1. / 512.), self.dec_log_stdv]
        x = x.view(-1, self.input_size)
        g = mu
        v = var
        if not self.logit_transform:
            if self.dataset != 'MNIST':
                x = x - 0.5

        if self.dataset == 'MNIST' and good_dims is not None:
            x = x[:, good_dims]

        u_oh = torch.zeros(u.shape[0], 10)
        u_oh[range(u_oh.shape[0]), u] = 1
        prior_params = self.prior(u_oh)

        # run standard VAE
        # term 1: compute log p(x|z), the MC estimate of E_{q(z|x)}[log p(x|z)] where z~q(z|x)
        log_prob_px_z = self.decoder_dist.log_pdf(x, *decoder_params)

        # term 2: compute the MC estimate of E_{q(z|x)}[log q(z|x)] where z~q(z|x)
        log_prob_qz_x = self.encoder_dist.log_pdf(z, g, v)

        # term 2: compute the MC estimate of E_{q(z|x)}[log p(z)] where z~q(z|x)
        log_prob_pz = self.prior_dist.log_pdf(z, *prior_params)

        KL_z = (log_prob_qz_x - log_prob_pz).mean()
        ELBO = (log_prob_px_z).mean() - KL_z

        return ELBO, [log_prob_px_z.mean(), KL_z], z

    def compute_q_y_x(self, x, u=None):
        f, mu, var, z = self.forward(x, True)
        # ie, no 'VaDE' trick where we marginalise out y in the gen. model
        pz_y = dist.Independent(dist.Normal(loc=self.mu_p_z.permute(1, 0),
                                            scale=torch.exp(0.5 * self.log_sigma_square_p_z).permute(1, 0)),
                                1)

        z_pad = torch.unsqueeze(z, -2)
        l_pz_y = pz_y.log_prob(z_pad)
        q_y_x = torch.softmax(l_pz_y, dim=1)
        return q_y_x


class iVAEFullMLP(iVAEConvMLP):
    def __init__(self, config, good_dims):
        super().__init__(config)
        self.good_dims = good_dims

    def make_networks(self, config):
        self.encoder = FullMLPencoder(config)
        self.decoder = FullMLPdecoder(config)

    # def prior(self, y):
    #     # h2 = F.relu(self.l1(y))
    #     # h2 = self.l1(y)
    #     return (self.mu_p_z @ y.T).T, F.softplus(self.log_sigma_square_p_z @ y.T).T

    def encode(self, x):
        if not self.logit_transform:
            x = x - 0.5
        return self.encoder(x.view(-1, self.input_size))

    def sample(self, n_sample=1, per_comp=False):
        y_samples = torch.tensor(np.eye(10, dtype=np.float32))
        loc = y_samples @ self.mu_p_z.permute(1,0)
        scale = y_samples @ torch.exp(self.log_sigma_square_p_z).permute(1, 0)
        pz_y = dist.Normal(loc=loc, scale=scale)
        z_samples = pz_y.sample((n_sample,))
        z_samples = z_samples.view(-1, self.latent_dim)
        f = self.decode(z_samples)
        if self.dataset == 'MNIST':
            f = torch.sigmoid(f).view(n_sample, -1, self.input_size)
            output = torch.zeros(f.shape)
            output[:,:,self.good_dims] = f[:,:,self.good_dims]
            output -= 0.5
            return output
        else:
            return f.view(n_sample, -1, self.input_size)


class iVAEResNetMLP(iVAEFullMLP):
    def __init__(self, config, good_dims):
        super().__init__(config, good_dims)

    def encode(self, x):
        if not self.logit_transform:
            x = x - 0.5
        return self.encoder(x.view(-1, self.n_channels, self.image_size, self.image_size))

    def make_networks(self, config):
        self.encoder = ResnetEncoder(config)
        self.decoder = ResnetDecoder(config)