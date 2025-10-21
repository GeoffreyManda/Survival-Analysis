"""
utils/models.py

Neural network architectures for causal representation learning.
Includes CEVAE, Deep CFR, and neural survival models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli
import numpy as np


class CEVAE(nn.Module):
    """
    Causal Effect Variational Autoencoder (Louizos et al., 2017).

    Jointly learns latent confounders and causal effects from observational data.
    """

    def __init__(self, input_dim, latent_dim=10, hidden_dims=[64, 32],
                 treatment_dim=1, outcome_type='binary'):
        """
        Args:
            input_dim: Dimension of input features X
            latent_dim: Dimension of latent confounders Z
            hidden_dims: Hidden layer dimensions
            treatment_dim: Dimension of treatment A (1 for binary)
            outcome_type: 'binary' or 'continuous' or 'survival'
        """
        super(CEVAE, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.treatment_dim = treatment_dim
        self.outcome_type = outcome_type

        # Inference network q(z | x, a, y)
        encoder_input_dim = input_dim + treatment_dim + 1  # x, a, y
        self.encoder = self._build_encoder(encoder_input_dim, latent_dim, hidden_dims)

        # Generative network p(x | z)
        self.decoder_x = self._build_decoder(latent_dim, input_dim, hidden_dims)

        # Treatment model p(a | z)
        self.decoder_a = self._build_treatment_model(latent_dim, treatment_dim, hidden_dims[:1])

        # Outcome model p(y | a, z)
        outcome_input_dim = latent_dim + treatment_dim
        if outcome_type == 'binary':
            self.decoder_y = self._build_binary_outcome_model(outcome_input_dim, hidden_dims[:1])
        elif outcome_type == 'survival':
            # For survival, predict hazard parameters
            self.decoder_y = self._build_survival_outcome_model(outcome_input_dim, hidden_dims[:1])
        else:
            self.decoder_y = self._build_continuous_outcome_model(outcome_input_dim, hidden_dims[:1])

    def _build_encoder(self, input_dim, latent_dim, hidden_dims):
        """Build encoder network q(z | x, a, y)."""
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim

        # Output mean and log variance
        self.encoder_mu = nn.Linear(prev_dim, latent_dim)
        self.encoder_logvar = nn.Linear(prev_dim, latent_dim)

        return nn.Sequential(*layers)

    def _build_decoder(self, latent_dim, output_dim, hidden_dims):
        """Build decoder for p(x | z)."""
        layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    def _build_treatment_model(self, latent_dim, treatment_dim, hidden_dims):
        """Build treatment propensity model p(a | z)."""
        layers = []
        prev_dim = latent_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, treatment_dim))
        layers.append(nn.Sigmoid())  # Probability
        return nn.Sequential(*layers)

    def _build_binary_outcome_model(self, input_dim, hidden_dims):
        """Build binary outcome model p(y | a, z)."""
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    def _build_continuous_outcome_model(self, input_dim, hidden_dims):
        """Build continuous outcome model p(y | a, z)."""
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 2))  # mean and log variance
        return nn.Sequential(*layers)

    def _build_survival_outcome_model(self, input_dim, hidden_dims):
        """Build survival outcome model (Weibull AFT)."""
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim

        # Output: log(scale) and log(shape) for Weibull distribution
        layers.append(nn.Linear(prev_dim, 2))
        return nn.Sequential(*layers)

    def encode(self, x, a, y):
        """Encode to latent space: q(z | x, a, y)."""
        # Concatenate inputs
        if len(a.shape) == 1:
            a = a.unsqueeze(1)
        if len(y.shape) == 1:
            y = y.unsqueeze(1)

        encoder_input = torch.cat([x, a, y], dim=1)

        # Pass through encoder
        h = self.encoder(encoder_input)
        mu = self.encoder_mu(h)
        logvar = self.encoder_logvar(h)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = μ + σ * ε."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, a):
        """Decode from latent space."""
        # Reconstruct x
        x_recon = self.decoder_x(z)

        # Predict treatment propensity
        a_prob = self.decoder_a(z)

        # Predict outcome
        if len(a.shape) == 1:
            a = a.unsqueeze(1)
        za = torch.cat([z, a], dim=1)
        y_pred = self.decoder_y(za)

        return x_recon, a_prob, y_pred

    def forward(self, x, a, y):
        """Full forward pass through CEVAE."""
        # Encode
        mu, logvar = self.encode(x, a, y)
        z = self.reparameterize(mu, logvar)

        # Decode
        x_recon, a_prob, y_pred = self.decode(z, a)

        return x_recon, a_prob, y_pred, mu, logvar

    def predict_counterfactuals(self, x, a_obs, y_obs):
        """
        Predict counterfactual outcomes Y(0) and Y(1).

        Args:
            x: Covariates
            a_obs: Observed treatment
            y_obs: Observed outcome

        Returns:
            y0_pred: Predicted outcome under A=0
            y1_pred: Predicted outcome under A=1
        """
        self.eval()
        with torch.no_grad():
            # Infer latent confounders from observed data
            mu, logvar = self.encode(x, a_obs, y_obs)
            z = mu  # Use mean (no sampling) for prediction

            # Predict outcomes under A=0
            a0 = torch.zeros(len(x), 1)
            za0 = torch.cat([z, a0], dim=1)
            y0_pred = self.decoder_y(za0)

            # Predict outcomes under A=1
            a1 = torch.ones(len(x), 1)
            za1 = torch.cat([z, a1], dim=1)
            y1_pred = self.decoder_y(za1)

            if self.outcome_type == 'binary':
                return y0_pred.squeeze(), y1_pred.squeeze()
            elif self.outcome_type == 'survival':
                # Return survival time predictions (scale parameter)
                return torch.exp(y0_pred[:, 0]), torch.exp(y1_pred[:, 0])
            else:
                # Continuous outcome (return mean)
                return y0_pred[:, 0], y1_pred[:, 0]


class DeepCFR(nn.Module):
    """
    Deep Counterfactual Representation Network (Shalit et al., 2017).

    Learns balanced representations across treatment groups.
    """

    def __init__(self, input_dim, repr_dim=50, hidden_dims=[128, 64, 32],
                 outcome_type='binary', alpha=1.0):
        """
        Args:
            input_dim: Dimension of input features
            repr_dim: Dimension of learned representation
            hidden_dims: Hidden layer dimensions
            outcome_type: 'binary', 'continuous', or 'survival'
            alpha: Weight for representation balancing loss
        """
        super(DeepCFR, self).__init__()

        self.input_dim = input_dim
        self.repr_dim = repr_dim
        self.outcome_type = outcome_type
        self.alpha = alpha

        # Representation network Φ(X)
        self.repr_net = self._build_repr_network(input_dim, repr_dim, hidden_dims)

        # Separate outcome networks for control and treatment
        if outcome_type == 'binary':
            self.outcome_net_0 = self._build_binary_outcome_net(repr_dim)
            self.outcome_net_1 = self._build_binary_outcome_net(repr_dim)
        elif outcome_type == 'survival':
            self.outcome_net_0 = self._build_survival_outcome_net(repr_dim)
            self.outcome_net_1 = self._build_survival_outcome_net(repr_dim)
        else:
            self.outcome_net_0 = self._build_continuous_outcome_net(repr_dim)
            self.outcome_net_1 = self._build_continuous_outcome_net(repr_dim)

    def _build_repr_network(self, input_dim, repr_dim, hidden_dims):
        """Build shared representation network."""
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, repr_dim))
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def _build_binary_outcome_net(self, repr_dim):
        """Build binary outcome prediction network."""
        return nn.Sequential(
            nn.Linear(repr_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def _build_continuous_outcome_net(self, repr_dim):
        """Build continuous outcome prediction network."""
        return nn.Sequential(
            nn.Linear(repr_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def _build_survival_outcome_net(self, repr_dim):
        """Build survival outcome prediction network."""
        return nn.Sequential(
            nn.Linear(repr_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)  # scale and shape parameters
        )

    def forward(self, x, a):
        """
        Forward pass.

        Args:
            x: Input features
            a: Treatment assignment

        Returns:
            y_pred: Predicted outcome under observed treatment
            repr: Learned representation
        """
        # Get representation
        repr = self.repr_net(x)

        # Predict outcomes for both treatment levels
        y0 = self.outcome_net_0(repr)
        y1 = self.outcome_net_1(repr)

        # Select predicted outcome based on observed treatment
        if len(a.shape) == 1:
            a = a.unsqueeze(1)

        y_pred = a * y1 + (1 - a) * y0

        return y_pred, repr, y0, y1

    def predict_counterfactuals(self, x):
        """
        Predict counterfactual outcomes for all observations.

        Args:
            x: Input features

        Returns:
            y0: Potential outcomes under control
            y1: Potential outcomes under treatment
        """
        self.eval()
        with torch.no_grad():
            repr = self.repr_net(x)
            y0 = self.outcome_net_0(repr)
            y1 = self.outcome_net_1(repr)

            if self.outcome_type == 'binary':
                return y0.squeeze(), y1.squeeze()
            elif self.outcome_type == 'survival':
                # Return survival time (scale parameter)
                return torch.exp(y0[:, 0]), torch.exp(y1[:, 0])
            else:
                return y0.squeeze(), y1.squeeze()


class NeuralCoxPH(nn.Module):
    """
    Neural Cox Proportional Hazards model for survival analysis.
    """

    def __init__(self, input_dim, hidden_dims=[64, 32], treatment_specific=False):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dims: Hidden layer sizes
            treatment_specific: If True, use separate networks for A=0 and A=1
        """
        super(NeuralCoxPH, self).__init__()

        self.treatment_specific = treatment_specific

        if treatment_specific:
            # Separate networks for each treatment level
            self.net_0 = self._build_network(input_dim, hidden_dims)
            self.net_1 = self._build_network(input_dim, hidden_dims)
        else:
            # Single network (treatment as input)
            self.net = self._build_network(input_dim + 1, hidden_dims)

    def _build_network(self, input_dim, hidden_dims):
        """Build neural network for log-hazard."""
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))  # Log hazard
        return nn.Sequential(*layers)

    def forward(self, x, a=None):
        """
        Compute log-hazard.

        Args:
            x: Features
            a: Treatment (if not treatment_specific)

        Returns:
            log_hazard: Log of the hazard function
        """
        if self.treatment_specific:
            log_h0 = self.net_0(x)
            log_h1 = self.net_1(x)

            if a is not None:
                if len(a.shape) == 1:
                    a = a.unsqueeze(1)
                log_h = a * log_h1 + (1 - a) * log_h0
                return log_h, log_h0, log_h1
            else:
                return log_h0, log_h1
        else:
            if len(a.shape) == 1:
                a = a.unsqueeze(1)
            xa = torch.cat([x, a], dim=1)
            return self.net(xa)


def wasserstein_distance(X, Y):
    """
    Approximate Wasserstein distance between two sets of representations.

    Uses sliced Wasserstein (efficient approximation).

    Args:
        X: Tensor of shape (n1, d)
        Y: Tensor of shape (n2, d)

    Returns:
        Approximate Wasserstein distance
    """
    n_projections = 50
    d = X.shape[1]

    total_distance = 0.0

    for _ in range(n_projections):
        # Random projection direction
        theta = torch.randn(d, 1, device=X.device)
        theta = theta / torch.norm(theta)

        # Project both sets
        X_proj = torch.mm(X, theta).squeeze()
        Y_proj = torch.mm(Y, theta).squeeze()

        # Sort projections
        X_sorted, _ = torch.sort(X_proj)
        Y_sorted, _ = torch.sort(Y_proj)

        # Ensure same length (for distance calculation)
        min_len = min(len(X_sorted), len(Y_sorted))

        # 1D Wasserstein is just mean absolute difference of sorted values
        distance = torch.mean(torch.abs(X_sorted[:min_len] - Y_sorted[:min_len]))
        total_distance += distance

    return total_distance / n_projections


def mmd_loss(X, Y, kernel='rbf', bandwidth=1.0):
    """
    Maximum Mean Discrepancy (MMD) between two distributions.

    Args:
        X: Samples from first distribution (n1, d)
        Y: Samples from second distribution (n2, d)
        kernel: Kernel type ('rbf', 'linear')
        bandwidth: Kernel bandwidth

    Returns:
        MMD distance
    """
    if kernel == 'rbf':
        XX = rbf_kernel(X, X, bandwidth)
        YY = rbf_kernel(Y, Y, bandwidth)
        XY = rbf_kernel(X, Y, bandwidth)
    else:  # linear
        XX = torch.mm(X, X.t())
        YY = torch.mm(Y, Y.t())
        XY = torch.mm(X, Y.t())

    return XX.mean() + YY.mean() - 2 * XY.mean()


def rbf_kernel(X, Y, bandwidth):
    """RBF (Gaussian) kernel."""
    pairwise_sq_dists = torch.cdist(X, Y, p=2) ** 2
    return torch.exp(-pairwise_sq_dists / (2 * bandwidth ** 2))
