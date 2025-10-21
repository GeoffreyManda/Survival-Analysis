# Causal Representation Learning for COVID-19 Survival Analysis

This directory contains Python implementation of causal representation learning methods for analyzing COVID-19 patient outcomes.

## Overview

Unlike traditional causal inference methods (IPW, g-formula) that require all confounders to be measured, **causal representation learning** uses deep learning to:
- Learn latent confounders from observed data
- Capture complex non-linear relationships
- Generate individual-level counterfactual predictions
- Handle unmeasured confounding (under certain assumptions)

## Quick Start

### 1. Setup Environment

```bash
# Create conda environment
conda create -n causal_rep python=3.9
conda activate causal_rep

# Install dependencies
pip install -r requirements.txt
```

### 2. Preprocess Data

```bash
cd scripts/python
python 00_data_preprocessing.py
```

This creates train/val/test splits and saves to `data/processed/`.

### 3. Train CEVAE

```bash
python 01_cevae_survival.py
```

Trains the Causal Effect VAE and estimates:
- ATE (Average Treatment Effect)
- ITE (Individual Treatment Effects)
- Counterfactual outcomes Y(0) and Y(1)

### 4. View Results

Results are saved to:
- `models/causal_rep/` - Trained models
- `output/causal_rep/` - Effect estimates (JSON)
- `figures/causal_rep/` - Visualizations

## Files

### Core Scripts

| File | Description |
|------|-------------|
| `00_data_preprocessing.py` | Load and preprocess COVID-19 data |
| `01_cevae_survival.py` | Train CEVAE model |
| `02_deep_cfr_survival.py` | Train Deep CFR model |
| `03_neural_causal_survival.py` | Neural Cox/AFT models |
| `04_causal_forest_embeddings.py` | Causal forests on learned representations |
| `05_variational_ci.py` | Variational causal inference |
| `06_evaluation.py` | Model evaluation and comparison |
| `07_visualization.py` | Generate plots and interpretations |

### Utilities

| File | Description |
|------|-------------|
| `utils/models.py` | Neural network architectures (CEVAE, Deep CFR, etc.) |
| `utils/losses.py` | Loss functions for causal learning |
| `utils/metrics.py` | Evaluation metrics (PEHE, policy value) |

### R Interface

| File | Description |
|------|-------------|
| `../r/09_run_python_causal.r` | R wrapper to call Python scripts |

## Methods Implemented

### 1. CEVAE (Causal Effect VAE)

**Paper**: Louizos et al. (2017)

**How it works**:
- Variational autoencoder that learns latent confounders Z
- Jointly models: P(X|Z), P(A|Z), P(Y|A,Z)
- Generates counterfactuals by intervening on A in latent space

**When to use**:
- Suspect important unmeasured confounders (e.g., disease severity, comorbidities)
- Want uncertainty quantification (Bayesian)
- Need individual treatment effects

**Example**:
```python
from utils.models import CEVAE

model = CEVAE(input_dim=4, latent_dim=10, outcome_type='binary')
model.fit(X_train, A_train, Y_train)

# Estimate counterfactuals
Y0, Y1 = model.predict_counterfactuals(X_test, A_test, Y_test)
ITE = Y1 - Y0  # Individual treatment effects
ATE = ITE.mean()  # Average treatment effect
```

### 2. Deep CFR (Counterfactual Representation)

**Paper**: Shalit et al. (2017)

**How it works**:
- Learns representations Φ(X) that are balanced across treatment groups
- Minimizes: Prediction Loss + λ * Wasserstein(Φ_treated, Φ_control)
- Separate outcome heads for A=0 and A=1

**When to use**:
- Strong covariate shift between treated and control groups
- Want representations that respect causal structure
- Have sufficient sample size for deep learning

**Advantages**:
- Theoretical guarantees on generalization
- Reduces selection bias through balancing
- Interpretable via learned representations

### 3. Neural Survival Models

**How it works**:
- Deep learning for survival analysis
- Options: Deep Cox, DeepHit, Deep AFT
- Treatment-specific hazard functions

**When to use**:
- Time-to-event outcomes (hospital length of stay)
- Right-censoring present
- Complex non-linear hazard functions

### 4. Causal Forests + Deep Learning (Hybrid)

**How it works**:
- Stage 1: Learn representations with neural network
- Stage 2: Apply causal forest on learned features

**When to use**:
- Want heterogeneous treatment effects (CATE)
- Need valid confidence intervals
- Have moderate sample size

**Advantages**:
- Combines flexibility of neural nets with forest interpretability
- Honest confidence intervals (unlike pure neural networks)
- Discovers subgroups with differential effects

## Configuration

### Model Hyperparameters

Edit configuration in training scripts or pass as arguments:

```python
config = {
    'latent_dim': 10,           # Latent confounder dimension
    'hidden_dims': [64, 32],    # Hidden layer sizes
    'learning_rate': 1e-3,      # Learning rate
    'batch_size': 128,          # Batch size
    'epochs': 100,              # Training epochs
    'beta': 1.0,                # KL divergence weight (CEVAE)
    'alpha': 1.0                # Balancing weight (Deep CFR)
}
```

### Hyperparameter Tuning

Use validation set for tuning:
- `latent_dim`: 5-20 (higher for complex confounding)
- `hidden_dims`: [64,32] or [128,64,32] (deeper for larger datasets)
- `beta`: 0.1-2.0 (controls KL regularization in VAE)
- `alpha`: 0.1-10.0 (controls representation balancing)

## Evaluation

### Metrics

1. **ATE (Average Treatment Effect)**
   ```
   ATE = E[Y(1) - Y(0)]
   ```
   Population-level causal effect

2. **ATT (Average Treatment effect on Treated)**
   ```
   ATT = E[Y(1) - Y(0) | A=1]
   ```
   Effect for those who received treatment

3. **CATE (Conditional ATE)**
   ```
   CATE(x) = E[Y(1) - Y(0) | X=x]
   ```
   Treatment effect given covariates

4. **PEHE (Precision in Estimation of Heterogeneous Effects)**
   ```
   PEHE = sqrt(E[(τ(x) - τ̂(x))²])
   ```
   Requires ground truth (semi-synthetic experiments)

5. **Policy Value**
   ```
   V(π) = E[Y under policy π]
   ```
   Expected outcome under learned treatment rule

### Validation Strategies

1. **Cross-validation**: 5-fold CV on training data
2. **Held-out test set**: 20% of data
3. **Calibration**: Are predicted effects well-calibrated?
4. **Comparison with benchmarks**: IPW, g-formula, doubly robust
5. **Sensitivity analysis**: Robustness to hyperparameters

## Interpretation

### SHAP Values

Explain latent representations:

```python
import shap

# Train model
model.fit(X_train, A_train, Y_train)

# Get representations
representations = model.encoder(X_test)

# SHAP explainer
explainer = shap.DeepExplainer(model.encoder, X_train)
shap_values = explainer.shap_values(X_test)

# Plot
shap.summary_plot(shap_values, X_test, feature_names=['age', 'gender', 'wave', 'hospital'])
```

### Subgroup Discovery

Find heterogeneous treatment effects:

```python
# Estimate ITEs
ITE = model.predict_ite(X_test)

# Find high-benefit subgroup
high_benefit = X_test[ITE > np.percentile(ITE, 75)]

# Characterize subgroup
print("High-benefit subgroup characteristics:")
print(pd.DataFrame(high_benefit, columns=['age', 'gender', 'wave', 'hospital']).describe())
```

## Comparison with Traditional Methods

| Aspect | IPW/G-formula | CEVAE | Deep CFR | Causal Forest+DL |
|--------|---------------|-------|----------|------------------|
| **Unmeasured confounding** | ❌ No | ✅ Yes (proxies) | ⚠️ Partial | ❌ No |
| **Non-linearity** | ⚠️ Manual | ✅ Automatic | ✅ Automatic | ✅ Automatic |
| **Heterogeneity (CATE)** | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |
| **Uncertainty quantification** | ⚠️ Bootstrap | ✅ Bayesian | ⚠️ Limited | ✅ CI from forest |
| **Interpretability** | ✅ High | ⚠️ Moderate | ⚠️ Moderate | ✅ High |
| **Sample size required** | Small (100s) | Medium (1000s) | Medium (1000s) | Large (10000s) |
| **Computational cost** | Low | High | High | Medium |

**Recommendation for COVID-19 data (N ≈ 10,000)**:
- **Primary**: CEVAE (handles unmeasured confounding, sufficient sample size)
- **Secondary**: Deep CFR (balancing, good for treatment selection bias)
- **Benchmark**: IPW and g-formula (from R scripts) for comparison

## Common Issues

### 1. Model Not Converging

**Symptoms**: Loss plateaus at high value, poor predictions

**Solutions**:
- Reduce learning rate: `1e-3 → 1e-4`
- Increase model capacity: `hidden_dims=[64,32] → [128,64,32]`
- Adjust batch size: smaller batches (64) for noisy gradients
- Check data normalization: ensure features are scaled

### 2. Unrealistic Treatment Effects

**Symptoms**: ATE >> 1 or << -1 (for binary outcome)

**Solutions**:
- Check outcome encoding: should be 0/1 for binary
- Reduce latent dimension: `latent_dim=10 → 5`
- Increase regularization: `beta=1.0 → 2.0`
- Add outcome constraints (e.g., sigmoid activation)

### 3. Poor Counterfactual Quality

**Symptoms**: Y(0) and Y(1) very different from Y_obs

**Solutions**:
- Check factual loss: predictions should match observed outcomes
- Balance treatment/outcome losses in ELBO
- Use more informative priors on Z
- Add consistency regularization

### 4. Overfitting

**Symptoms**: Low train loss, high validation loss

**Solutions**:
- Increase dropout: `0.2 → 0.5`
- Reduce model capacity
- Early stopping with patience
- Add L2 regularization

### 5. Class Imbalance

**Symptoms**: Model predicts majority class only

**Solutions**:
- Weighted loss functions
- Oversample minority class (SMOTE)
- Stratified sampling in data loader
- Adjust class weights in loss

## Advanced Usage

### Custom Architectures

Modify `utils/models.py` to add custom layers:

```python
class CustomCEVAE(CEVAE):
    def _build_encoder(self, input_dim, latent_dim, hidden_dims):
        # Add batch normalization, different activations, etc.
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.LeakyReLU(0.2),
            # ... more layers
        )
```

### Multi-valued Treatments

For >2 treatment levels (e.g., viral load categories):

```python
model = CEVAE(
    input_dim=4,
    latent_dim=10,
    treatment_dim=3,  # 3 treatment levels
    outcome_type='binary'
)

# Use categorical treatment encoding
A_train_onehot = F.one_hot(torch.LongTensor(A_train), num_classes=3)
```

### Continuous Treatments

For dose-response (e.g., continuous Ct value):

```python
# Modify outcome model to accept continuous treatment
def predict_dose_response(model, X, doses):
    responses = []
    for dose in doses:
        A_dose = torch.full((len(X), 1), dose)
        Y_dose = model.predict_outcome(X, A_dose)
        responses.append(Y_dose)
    return torch.stack(responses)

# Plot dose-response curve
doses = np.linspace(10, 40, 100)  # Ct values
responses = predict_dose_response(model, X_test, doses)
plt.plot(doses, responses.mean(axis=1))
```

## References

### Papers

1. **Louizos et al. (2017)** - "Causal Effect Inference with Deep Latent-Variable Models"
   - CEVAE framework
   - NIPS 2017

2. **Shalit et al. (2017)** - "Estimating individual treatment effect"
   - Deep counterfactual representations
   - ICML 2017

3. **Kaddour et al. (2022)** - "Causal Machine Learning: A Survey and Open Problems"
   - Comprehensive review
   - arXiv:2206.15475

4. **Curth & van der Schaar (2021)** - "Nonparametric Estimation of Heterogeneous Treatment Effects"
   - Neural causal models
   - AISTATS 2021

### Software

- **PyTorch**: https://pytorch.org/
- **EconML**: https://econml.azurewebsites.net/
- **DoWhy**: https://microsoft.github.io/dowhy/
- **Pyro**: https://pyro.ai/

## Support

For questions or issues:
1. Check this README
2. Review `CAUSAL_REPRESENTATION_LEARNING.md` in project root
3. Consult papers listed in References
4. Open GitHub issue

## License

CC BY 4.0 - see LICENSE.md

## Citation

If you use this code, please cite:

```bibtex
@software{manda2025causal,
  author = {Manda, Geoffrey},
  title = {Causal Representation Learning for COVID-19 Survival Analysis},
  year = {2025},
  url = {https://github.com/GeoffreyManda/Survival-Analysis}
}
```
