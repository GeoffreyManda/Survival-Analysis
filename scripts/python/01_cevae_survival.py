"""
01_cevae_survival.py

Train Causal Effect Variational Autoencoder (CEVAE) on COVID-19 data.
Estimates causal effects of viral load on mortality with unmeasured confounding.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
import sys
sys.path.append('.')
from utils.models import CEVAE
from utils.losses import cevae_loss

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)


class CEVAETrainer:
    """Trainer for CEVAE model."""

    def __init__(self, model, device='cpu', learning_rate=1e-3):
        """
        Args:
            model: CEVAE model
            device: 'cpu' or 'cuda'
            learning_rate: Learning rate
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, train_loader, beta=1.0):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_treatment = 0.0
        epoch_outcome = 0.0
        epoch_kl = 0.0

        for batch_idx, (x, a, y, t) in enumerate(train_loader):
            x = x.to(self.device)
            a = a.to(self.device)
            y = y.to(self.device)

            # Forward pass
            x_recon, a_prob, y_pred, mu, logvar = self.model(x, a, y)

            # Compute loss
            loss, recon, treatment, outcome, kl = cevae_loss(
                x_recon, x, a_prob, a, y_pred, y, mu, logvar,
                outcome_type=self.model.outcome_type,
                beta=beta
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Accumulate losses
            epoch_loss += loss.item()
            epoch_recon += recon.item()
            epoch_treatment += treatment.item()
            epoch_outcome += outcome.item()
            epoch_kl += kl.item()

        n_batches = len(train_loader)
        return {
            'total': epoch_loss / n_batches,
            'recon': epoch_recon / n_batches,
            'treatment': epoch_treatment / n_batches,
            'outcome': epoch_outcome / n_batches,
            'kl': epoch_kl / n_batches
        }

    def validate(self, val_loader, beta=1.0):
        """Validate model."""
        self.model.eval()
        val_loss = 0.0
        val_recon = 0.0
        val_treatment = 0.0
        val_outcome = 0.0
        val_kl = 0.0

        with torch.no_grad():
            for x, a, y, t in val_loader:
                x = x.to(self.device)
                a = a.to(self.device)
                y = y.to(self.device)

                # Forward pass
                x_recon, a_prob, y_pred, mu, logvar = self.model(x, a, y)

                # Compute loss
                loss, recon, treatment, outcome, kl = cevae_loss(
                    x_recon, x, a_prob, a, y_pred, y, mu, logvar,
                    outcome_type=self.model.outcome_type,
                    beta=beta
                )

                val_loss += loss.item()
                val_recon += recon.item()
                val_treatment += treatment.item()
                val_outcome += outcome.item()
                val_kl += kl.item()

        n_batches = len(val_loader)
        return {
            'total': val_loss / n_batches,
            'recon': val_recon / n_batches,
            'treatment': val_treatment / n_batches,
            'outcome': val_outcome / n_batches,
            'kl': val_kl / n_batches
        }

    def fit(self, train_loader, val_loader, epochs=100, beta=1.0,
            early_stopping_patience=10, verbose=True):
        """
        Train CEVAE model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            beta: KL divergence weight
            early_stopping_patience: Patience for early stopping
            verbose: Print progress
        """
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, beta=beta)
            val_metrics = self.validate(val_loader, beta=beta)

            # Store losses
            self.train_losses.append(train_metrics['total'])
            self.val_losses.append(val_metrics['total'])

            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train Loss: {train_metrics['total']:.4f} "
                      f"(Recon: {train_metrics['recon']:.4f}, "
                      f"Treatment: {train_metrics['treatment']:.4f}, "
                      f"Outcome: {train_metrics['outcome']:.4f}, "
                      f"KL: {train_metrics['kl']:.4f})")
                print(f"  Val Loss:   {val_metrics['total']:.4f}")

            # Early stopping
            if val_metrics['total'] < best_val_loss:
                best_val_loss = val_metrics['total']
                patience_counter = 0
                # Save best model
                self.best_model_state = self.model.state_dict()
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Load best model
        self.model.load_state_dict(self.best_model_state)
        print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f}")

    def plot_losses(self, output_path):
        """Plot training and validation losses."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train')
        plt.plot(self.val_losses, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('CEVAE Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


def estimate_effects(model, X_test, A_test, Y_test, device='cpu'):
    """
    Estimate treatment effects using trained CEVAE.

    Args:
        model: Trained CEVAE model
        X_test: Test features
        A_test: Test treatments
        Y_test: Test outcomes
        device: Device

    Returns:
        Dictionary of estimated effects
    """
    model.eval()
    with torch.no_grad():
        X_test = torch.FloatTensor(X_test).to(device)
        A_test = torch.FloatTensor(A_test).to(device)
        Y_test = torch.FloatTensor(Y_test).to(device)

        # Predict counterfactuals
        Y0_pred, Y1_pred = model.predict_counterfactuals(X_test, A_test, Y_test)

        Y0_pred = Y0_pred.cpu().numpy()
        Y1_pred = Y1_pred.cpu().numpy()

        # Individual treatment effects
        ITE = Y1_pred - Y0_pred

        # Average treatment effect
        ATE = ITE.mean()

        # ATT (Average Treatment effect on Treated)
        treated_mask = (A_test.cpu().numpy() == 1)
        ATT = ITE[treated_mask].mean() if treated_mask.sum() > 0 else np.nan

        # ATC (Average Treatment effect on Controls)
        control_mask = (A_test.cpu().numpy() == 0)
        ATC = ITE[control_mask].mean() if control_mask.sum() > 0 else np.nan

    results = {
        'ATE': ATE,
        'ATT': ATT,
        'ATC': ATC,
        'ITE_mean': ITE.mean(),
        'ITE_std': ITE.std(),
        'ITE_min': ITE.min(),
        'ITE_max': ITE.max(),
        'Y0_mean': Y0_pred.mean(),
        'Y1_mean': Y1_pred.mean()
    }

    return results, ITE


def main():
    """Main training pipeline."""
    print("="*70)
    print("CEVAE TRAINING - COVID-19 SURVIVAL ANALYSIS")
    print("="*70)

    # Configuration
    config = {
        'latent_dim': 10,
        'hidden_dims': [64, 32],
        'learning_rate': 1e-3,
        'batch_size': 128,
        'epochs': 100,
        'beta': 1.0,
        'early_stopping_patience': 15
    }

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load data
    print("\nLoading preprocessed data...")
    data_dir = Path("../../data/processed")

    X_train = np.load(data_dir / "X_train.npy")
    X_val = np.load(data_dir / "X_val.npy")
    X_test = np.load(data_dir / "X_test.npy")

    A_train = np.load(data_dir / "A_train.npy")
    A_val = np.load(data_dir / "A_val.npy")
    A_test = np.load(data_dir / "A_test.npy")

    Y_train = np.load(data_dir / "Y_train.npy")
    Y_val = np.load(data_dir / "Y_val.npy")
    Y_test = np.load(data_dir / "Y_test.npy")

    T_train = np.load(data_dir / "T_train.npy")
    T_val = np.load(data_dir / "T_val.npy")
    T_test = np.load(data_dir / "T_test.npy")

    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Val:   {X_val.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")

    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(A_train),
        torch.FloatTensor(Y_train),
        torch.FloatTensor(T_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(A_val),
        torch.FloatTensor(Y_val),
        torch.FloatTensor(T_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                               shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                             shuffle=False)

    # Initialize model
    print("\nInitializing CEVAE model...")
    model = CEVAE(
        input_dim=X_train.shape[1],
        latent_dim=config['latent_dim'],
        hidden_dims=config['hidden_dims'],
        treatment_dim=1,
        outcome_type='binary'  # Binary mortality outcome
    )

    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Initialize trainer
    trainer = CEVAETrainer(model, device=device,
                           learning_rate=config['learning_rate'])

    # Train
    print("\nTraining CEVAE...")
    trainer.fit(
        train_loader, val_loader,
        epochs=config['epochs'],
        beta=config['beta'],
        early_stopping_patience=config['early_stopping_patience'],
        verbose=True
    )

    # Save model
    output_dir = Path("../../models/causal_rep")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "cevae_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, model_path)
    print(f"\nModel saved to: {model_path}")

    # Plot training curves
    fig_dir = Path("../../figures/causal_rep")
    fig_dir.mkdir(parents=True, exist_ok=True)

    trainer.plot_losses(fig_dir / "cevae_training_curves.png")
    print(f"Training curves saved to: {fig_dir / 'cevae_training_curves.png'}")

    # Estimate treatment effects on test set
    print("\nEstimating treatment effects on test set...")
    effects, ITE = estimate_effects(model, X_test, A_test, Y_test, device=device)

    print("\nCausal Effect Estimates:")
    print(f"  ATE (Average Treatment Effect): {effects['ATE']:.4f}")
    print(f"  ATT (Effect on Treated):        {effects['ATT']:.4f}")
    print(f"  ATC (Effect on Controls):       {effects['ATC']:.4f}")
    print(f"  ITE std:                        {effects['ITE_std']:.4f}")
    print(f"  ITE range:                      [{effects['ITE_min']:.4f}, "
          f"{effects['ITE_max']:.4f}]")
    print(f"\nPotential Outcomes:")
    print(f"  E[Y(0)] (under control):        {effects['Y0_mean']:.4f}")
    print(f"  E[Y(1)] (under treatment):      {effects['Y1_mean']:.4f}")

    # Save results
    results_dir = Path("../../output/causal_rep")
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "cevae_effects.json", 'w') as f:
        json.dump(effects, f, indent=2)

    # Plot ITE distribution
    plt.figure(figsize=(10, 6))
    plt.hist(ITE, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(effects['ATE'], color='red', linestyle='--', linewidth=2,
                label=f'ATE = {effects["ATE"]:.4f}')
    plt.xlabel('Individual Treatment Effect (ITE)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Individual Treatment Effects (CEVAE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(fig_dir / "cevae_ite_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nResults saved to: {results_dir}")
    print("\n" + "="*70)
    print("CEVAE TRAINING COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
