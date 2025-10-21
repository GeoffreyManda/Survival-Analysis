"""
utils/losses.py

Loss functions for causal representation learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def cevae_loss(x_recon, x, a_prob, a, y_pred, y, mu, logvar,
               outcome_type='binary', beta=1.0):
    """
    CEVAE loss function (ELBO).

    Args:
        x_recon: Reconstructed features
        x: Original features
        a_prob: Predicted treatment probability
        a: Actual treatment
        y_pred: Predicted outcome
        y: Actual outcome
        mu: Latent mean
        logvar: Latent log variance
        outcome_type: 'binary', 'continuous', or 'survival'
        beta: Weight for KL divergence (beta-VAE)

    Returns:
        Total loss, reconstruction loss, treatment loss, outcome loss, KL loss
    """
    batch_size = x.size(0)

    # Reconstruction loss: p(x | z)
    recon_loss = F.mse_loss(x_recon, x, reduction='sum') / batch_size

    # Treatment likelihood: p(a | z)
    if len(a.shape) == 1:
        a = a.unsqueeze(1)

    treatment_loss = F.binary_cross_entropy(a_prob, a, reduction='sum') / batch_size

    # Outcome likelihood: p(y | a, z)
    if outcome_type == 'binary':
        if len(y.shape) == 1:
            y = y.unsqueeze(1)
        outcome_loss = F.binary_cross_entropy(y_pred, y, reduction='sum') / batch_size
    elif outcome_type == 'continuous':
        # Assuming y_pred has mean and log variance
        y_mean = y_pred[:, 0]
        y_logvar = y_pred[:, 1]
        outcome_loss = gaussian_nll(y, y_mean, y_logvar).sum() / batch_size
    else:  # survival
        # Placeholder - would need time-to-event data
        outcome_loss = F.mse_loss(y_pred[:, 0], y, reduction='sum') / batch_size

    # KL divergence: KL(q(z | x, a, y) || p(z))
    # p(z) = N(0, I)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

    # Total loss
    total_loss = recon_loss + treatment_loss + outcome_loss + beta * kl_loss

    return total_loss, recon_loss, treatment_loss, outcome_loss, kl_loss


def deep_cfr_loss(y_pred, y, repr_treated, repr_control, a,
                  outcome_type='binary', alpha=1.0, distance='wasserstein'):
    """
    Deep Counterfactual Representation loss.

    Args:
        y_pred: Predicted outcomes under observed treatment
        y: Actual outcomes
        repr_treated: Representations for treated units
        repr_control: Representations for control units
        a: Treatment assignment
        outcome_type: 'binary', 'continuous', or 'survival'
        alpha: Weight for balancing loss
        distance: 'wasserstein', 'mmd', or 'kl'

    Returns:
        Total loss, prediction loss, balancing loss
    """
    # Prediction loss
    if outcome_type == 'binary':
        if len(y.shape) == 1:
            y = y.unsqueeze(1)
        pred_loss = F.binary_cross_entropy(y_pred, y, reduction='mean')
    elif outcome_type == 'survival':
        # Cox partial likelihood or other survival loss
        pred_loss = F.mse_loss(y_pred[:, 0], y, reduction='mean')
    else:  # continuous
        pred_loss = F.mse_loss(y_pred.squeeze(), y, reduction='mean')

    # Balancing loss - minimize distributional distance between treatment groups
    if len(repr_treated) > 0 and len(repr_control) > 0:
        if distance == 'wasserstein':
            from utils.models import wasserstein_distance
            balance_loss = wasserstein_distance(repr_treated, repr_control)
        elif distance == 'mmd':
            from utils.models import mmd_loss
            balance_loss = mmd_loss(repr_treated, repr_control)
        else:  # mean difference
            balance_loss = torch.norm(repr_treated.mean(0) - repr_control.mean(0), p=2)
    else:
        balance_loss = torch.tensor(0.0)

    # Total loss
    total_loss = pred_loss + alpha * balance_loss

    return total_loss, pred_loss, balance_loss


def cox_partial_likelihood_loss(log_hazard, event, duration):
    """
    Cox proportional hazards partial likelihood loss.

    Args:
        log_hazard: Log hazard ratios (relative risks)
        event: Binary event indicator (1 if event occurred)
        duration: Time to event or censoring

    Returns:
        Negative partial log-likelihood
    """
    # Sort by duration (descending)
    sorted_indices = torch.argsort(duration, descending=True)
    log_hazard = log_hazard[sorted_indices].squeeze()
    event = event[sorted_indices]

    # Compute risk set for each event time
    # Risk set: all individuals still at risk at time t_i
    hazard = torch.exp(log_hazard)
    cumsum_hazard = torch.cumsum(hazard, dim=0)

    # Log partial likelihood
    log_pl = log_hazard - torch.log(cumsum_hazard)

    # Only include events (not censored)
    loss = -torch.sum(log_pl * event) / torch.sum(event)

    return loss


def survival_ranking_loss(log_hazard, duration, event, margin=0.0):
    """
    Ranking loss for survival analysis (similar to concordance index).

    Penalizes when predicted hazard order doesn't match observed survival order.

    Args:
        log_hazard: Predicted log hazards
        duration: Observed times
        event: Event indicators
        margin: Margin for ranking loss

    Returns:
        Ranking loss
    """
    n = len(duration)
    log_hazard = log_hazard.squeeze()

    # Create pairwise comparisons
    # For all pairs (i, j) where:
    # - i had event at time t_i
    # - j was still at risk at time t_i (duration_j >= t_i)
    # We want: hazard_i > hazard_j

    loss = 0.0
    count = 0

    for i in range(n):
        if event[i] == 1:  # i had an event
            for j in range(n):
                if duration[j] > duration[i]:  # j was still at risk
                    # Want log_hazard[i] > log_hazard[j]
                    diff = log_hazard[j] - log_hazard[i] + margin
                    loss += F.relu(diff)  # Hinge loss
                    count += 1

    if count > 0:
        loss = loss / count

    return loss


def gaussian_nll(y_true, y_mean, y_logvar):
    """
    Gaussian negative log-likelihood.

    Args:
        y_true: True values
        y_mean: Predicted mean
        y_logvar: Predicted log variance

    Returns:
        Negative log-likelihood
    """
    return 0.5 * (y_logvar + (y_true - y_mean) ** 2 / torch.exp(y_logvar))


def weibull_aft_loss(scale_pred, shape_pred, duration, event):
    """
    Weibull AFT (Accelerated Failure Time) loss.

    Args:
        scale_pred: Predicted log(scale) parameter
        shape_pred: Predicted log(shape) parameter
        duration: Observed times
        event: Event indicators (1 if event occurred, 0 if censored)

    Returns:
        Negative log-likelihood
    """
    scale = torch.exp(scale_pred)
    shape = torch.exp(shape_pred)

    # Log-likelihood for Weibull distribution
    # f(t) = (k/λ) * (t/λ)^(k-1) * exp(-(t/λ)^k)
    # S(t) = exp(-(t/λ)^k)

    # For observed events
    log_lik_event = (
        torch.log(shape) - torch.log(scale) +
        (shape - 1) * (torch.log(duration) - torch.log(scale)) -
        (duration / scale) ** shape
    )

    # For censored observations (survival function)
    log_lik_censored = -(duration / scale) ** shape

    # Combine
    log_lik = event * log_lik_event + (1 - event) * log_lik_censored

    return -log_lik.mean()


def balanced_mse_loss(y_pred, y_true, a, weight_treated=1.0, weight_control=1.0):
    """
    MSE loss with balancing weights for treatment groups.

    Ensures equal contribution from treated and control groups.

    Args:
        y_pred: Predictions
        y_true: True values
        a: Treatment assignment
        weight_treated: Weight for treated group
        weight_control: Weight for control group

    Returns:
        Weighted MSE loss
    """
    if len(a.shape) == 1:
        a = a.unsqueeze(1)

    # Separate treated and control
    treated_mask = (a == 1).squeeze()
    control_mask = (a == 0).squeeze()

    if treated_mask.sum() > 0:
        loss_treated = F.mse_loss(
            y_pred[treated_mask],
            y_true[treated_mask],
            reduction='mean'
        )
    else:
        loss_treated = torch.tensor(0.0)

    if control_mask.sum() > 0:
        loss_control = F.mse_loss(
            y_pred[control_mask],
            y_true[control_mask],
            reduction='mean'
        )
    else:
        loss_control = torch.tensor(0.0)

    # Weighted combination
    total_loss = (
        weight_treated * loss_treated +
        weight_control * loss_control
    ) / (weight_treated + weight_control)

    return total_loss


def factual_loss(y0_pred, y1_pred, y_true, a):
    """
    Factual loss: only evaluate predictions for observed treatment.

    Args:
        y0_pred: Predicted outcomes under control
        y1_pred: Predicted outcomes under treatment
        y_true: True outcomes
        a: Treatment assignment

    Returns:
        MSE loss for factual predictions only
    """
    if len(a.shape) == 1:
        a = a.unsqueeze(1)

    # Select factual predictions
    y_factual = a * y1_pred + (1 - a) * y0_pred

    return F.mse_loss(y_factual, y_true, reduction='mean')


def ate_regularization(y0_pred, y1_pred, target_ate=None):
    """
    Regularization to constrain average treatment effect.

    Useful when we have prior knowledge about plausible ATE values.

    Args:
        y0_pred: Predicted outcomes under control
        y1_pred: Predicted outcomes under treatment
        target_ate: Target ATE value (if known)

    Returns:
        Regularization loss
    """
    estimated_ate = (y1_pred - y0_pred).mean()

    if target_ate is not None:
        # Penalize deviation from target
        return (estimated_ate - target_ate) ** 2
    else:
        # Regularize toward zero (conservative)
        return estimated_ate ** 2
