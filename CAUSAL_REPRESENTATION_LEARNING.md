# Causal Representation Learning for COVID-19 Survival Analysis

## Overview

This document outlines a **causal representation learning** framework that combines deep learning with causal inference to analyze COVID-19 patient outcomes. Unlike traditional causal inference methods that rely on measured confounders, this approach **learns latent representations** that capture unmeasured confounding and complex non-linear relationships.

## Table of Contents

1. [Why Causal Representation Learning?](#why-causal-representation-learning)
2. [Theoretical Framework](#theoretical-framework)
3. [Methods Implemented](#methods-implemented)
4. [Architecture Overview](#architecture-overview)
5. [Implementation Guide](#implementation-guide)
6. [Comparison with Traditional Methods](#comparison-with-traditional-methods)
7. [Evaluation and Validation](#evaluation-and-validation)

---

## Why Causal Representation Learning?

### Limitations of Traditional Causal Inference

**Traditional methods** (IPW, g-formula) assume:
- All confounders are measured
- Correct functional form specification
- Linear or simple non-linear relationships

**Reality in COVID-19 data**:
- ❌ **Unmeasured confounders**: Comorbidities, disease severity, immune status
- ❌ **Complex interactions**: Age × viral load × hospital capacity
- ❌ **High-dimensional patterns**: Combinations of factors matter
- ❌ **Latent patient states**: "Frailty" not directly observable

### Advantages of Causal Representation Learning

**Causal representation learning** addresses these by:
- ✅ **Learning latent confounders** from observed covariates
- ✅ **Flexible non-linear modeling** via neural networks
- ✅ **Automatic feature learning** (no manual feature engineering)
- ✅ **Counterfactual generation** in learned representation space
- ✅ **Handling high-dimensional covariates** efficiently

---

## Theoretical Framework

### Structural Causal Model (SCM)

We assume the following generative process:

```
Z (latent confounders) ~ P(Z)
    |
    v
X (covariates) ~ P(X | Z)
    |
    v
A (treatment) ~ P(A | X, Z)
    |
    v
Y (outcome) ~ P(Y | A, X, Z, T)
    |
    v
T (survival time)
```

**Key Variables:**
- **Z**: Latent confounders (e.g., unmeasured frailty, disease severity, immune competence)
- **X**: Observed covariates (age, gender, hospital, wave)
- **A**: Treatment (viral load, wave assignment)
- **Y**: Outcome (mortality)
- **T**: Time-to-event (length of stay)

### Identifiability

**Challenge**: Z is unobserved, so standard causal identification fails.

**Solution**: Under certain assumptions (proxy variables, sufficient richness of X), we can:
1. **Learn a representation** φ(X) that captures Z
2. **Estimate causal effects** using learned representations
3. **Generate counterfactuals** in latent space

**Assumptions Required:**
1. **Latent unconfoundedness**: Y(a) ⊥ A | X, Z
2. **Proxy variable assumption**: X contains information about Z
3. **Positivity in representation space**: Overlap after learning φ(X)
4. **Consistency**: Well-defined interventions

---

## Methods Implemented

### 1. CEVAE (Causal Effect Variational Autoencoder)

**Paper**: Louizos et al. (2017) - "Causal Effect Inference with Deep Latent-Variable Models"

**Approach**:
- Variational autoencoder that jointly learns:
  - Latent representations Z
  - Treatment propensity P(A|Z)
  - Outcome model P(Y|A,Z)
- Uses proxy variables to infer latent confounders
- Generates counterfactual outcomes by intervening on A

**Architecture**:
```python
# Inference network (encoder)
q(z | x, a, y) = Encoder(x, a, y)

# Generative model (decoder)
p(x | z) = Decoder_X(z)
p(a | z) = Decoder_A(z)  # Treatment model
p(y | a, z) = Decoder_Y(a, z)  # Outcome model

# Objective
ELBO = E_q[log p(x,a,y|z)] - KL(q(z|x,a,y) || p(z))
```

**For COVID-19**:
- **Latent Z**: Unmeasured frailty, disease severity
- **Observed X**: Age, gender, hospital, wave
- **Treatment A**: Viral load (Ct value)
- **Outcome Y**: Mortality, time-to-event

**Advantages**:
- Handles unmeasured confounding
- Provides uncertainty quantification
- Generates individual-level counterfactuals

**Implementation**:
- `scripts/python/cevae_survival.py`
- PyTorch-based with survival extensions

---

### 2. Deep Counterfactual Networks (CFR)

**Paper**: Shalit et al. (2017) - "Estimating individual treatment effect: generalization bounds and algorithms"

**Approach**:
- Learns representations that are:
  - **Predictive** of outcomes
  - **Balanced** across treatment groups (minimizes distributional distance)
- Uses representation balancing regularization

**Architecture**:
```python
# Representation network
Φ = RepresentationNet(X)

# Outcome heads (separate for each treatment level)
μ₀ = OutcomeNet₀(Φ)  # Potential outcome under control
μ₁ = OutcomeNet₁(Φ)  # Potential outcome under treatment

# Loss
Loss = Prediction_Loss + λ * IPM(Φ_treated, Φ_control)
```
where IPM = Integral Probability Metric (e.g., Wasserstein distance)

**For COVID-19**:
- Learns balanced representations across viral load groups
- Estimates individual treatment effects: τ(x) = μ₁(Φ(x)) - μ₀(Φ(x))
- Extends to survival outcomes with Cox/AFT neural networks

**Advantages**:
- Strong theoretical guarantees
- Reduces covariate shift between treatment groups
- Scalable to large datasets

**Implementation**:
- `scripts/python/deep_cfr_survival.py`
- TensorFlow/PyTorch with survival loss functions

---

### 3. Neural Causal Survival Models

**Approach**:
- Combines neural networks with survival analysis
- Causal structure encoded in network architecture
- Methods include:
  - **DeepSurv**: Deep Cox proportional hazards
  - **DeepHit**: Competing risks with deep learning
  - **DRSA**: Deep recurrent survival analysis

**Architecture** (Causal DeepSurv):
```python
# Shared representation
Z = SharedNet(X)

# Treatment-specific survival networks
h₀(t | x) = DeepCoxNet₀(Z, t)  # Baseline hazard under A=0
h₁(t | x) = DeepCoxNet₁(Z, t)  # Hazard under A=1

# Counterfactual survival
S₀(t | x) = exp(-∫h₀(s|x)ds)
S₁(t | x) = exp(-∫h₁(s|x)ds)

# Individual treatment effect
ITE(x) = S₁(t | x) - S₀(t | x)
```

**For COVID-19**:
- Estimates individualized survival curves under different treatments
- Accounts for time-to-event censoring
- Learns non-linear hazard functions

**Advantages**:
- Handles right-censoring natively
- Captures time-varying effects
- Individual-level predictions

**Implementation**:
- `scripts/python/neural_causal_survival.py`
- Uses `pycox` and `lifelines` with custom causal layers

---

### 4. Causal Forests with Learned Representations

**Approach**:
- Two-stage method:
  1. Learn representations with deep neural network
  2. Apply causal forest on learned representations

**Why this hybrid?**
- **Neural nets**: Capture complex non-linear patterns
- **Causal forests**: Provide honest treatment effect estimates with valid inference

**Architecture**:
```python
# Stage 1: Representation learning
Φ = AutoEncoder(X) or SupervisedNet(X → Y)

# Stage 2: Causal forest on representations
CF = CausalForest(treatment=A, outcome=Y, features=Φ)

# Estimate CATE
τ(x) = CF.predict(Φ(x))
```

**For COVID-19**:
- Learn embeddings of patient states
- Estimate heterogeneous treatment effects
- Identify subgroups with differential effects

**Advantages**:
- Combines deep learning flexibility with forest interpretability
- Honest confidence intervals (via causal forest theory)
- Discovers heterogeneous effects

**Implementation**:
- `scripts/python/causal_forest_embeddings.py`
- Uses `econml` and `sklearn` with custom feature extractors

---

### 5. Variational Causal Inference (VCI)

**Approach**:
- Bayesian approach to causal effect estimation
- Uses variational inference to:
  - Infer latent confounders
  - Quantify uncertainty in causal estimates
  - Generate posterior distributions over counterfactuals

**Architecture**:
```python
# Posterior over latent confounders
q(Z | X, A, Y) = InferenceNet(X, A, Y)

# Generative model with causal structure
p(X, A, Y | Z) = p(X|Z) * p(A|X,Z) * p(Y|A,X,Z)

# Counterfactual inference
p(Y^a | X) = ∫ p(Y|A=a,X,Z) q(Z|X,A_obs,Y_obs) dZ
```

**For COVID-19**:
- Posterior distributions over treatment effects
- Credible intervals for causal estimates
- Sensitivity to prior specifications

**Advantages**:
- Full uncertainty quantification
- Incorporates prior knowledge
- Robust to model misspecification (via Bayesian model averaging)

**Implementation**:
- `scripts/python/variational_causal_inference.py`
- PyTorch with Pyro for probabilistic programming

---

## Architecture Overview

### Overall Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PREPROCESSING                            │
│  - Load COVID-19 data                                           │
│  - Handle missing values                                        │
│  - Normalize/standardize features                               │
│  - Create train/validation/test splits                          │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     v
┌─────────────────────────────────────────────────────────────────┐
│             CAUSAL REPRESENTATION LEARNING                       │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    │
│  │    CEVAE     │    │  Deep CFR    │    │ Neural Surv  │    │
│  │              │    │              │    │   Models     │    │
│  │ Learn Z,     │    │ Balanced     │    │ Deep Cox,    │    │
│  │ P(Y|A,Z)     │    │ Repr. Φ(X)   │    │ DeepHit      │    │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘    │
│         │                    │                    │             │
│         └────────────────────┴────────────────────┘             │
│                              │                                  │
└──────────────────────────────┼──────────────────────────────────┘
                               │
                               v
┌─────────────────────────────────────────────────────────────────┐
│              CAUSAL EFFECT ESTIMATION                            │
│  - Individual Treatment Effects (ITE)                           │
│  - Conditional Average Treatment Effects (CATE)                 │
│  - Average Treatment Effects (ATE)                              │
│  - Counterfactual survival curves                               │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     v
┌─────────────────────────────────────────────────────────────────┐
│           EVALUATION & VALIDATION                                │
│  - PEHE (Precision in Estimation of Heterogeneous Effects)     │
│  - Policy value estimation                                      │
│  - Calibration assessment                                       │
│  - Comparison with traditional methods (IPW, g-formula)         │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     v
┌─────────────────────────────────────────────────────────────────┐
│         INTERPRETATION & VISUALIZATION                           │
│  - SHAP values for representation interpretation                │
│  - Subgroup discovery                                           │
│  - Counterfactual explanations                                  │
│  - Effect heterogeneity analysis                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Guide

### Prerequisites

**Python Environment**:
```bash
# Create conda environment
conda create -n causal_rep python=3.9
conda activate causal_rep

# Install core packages
pip install torch torchvision torchaudio
pip install tensorflow tensorflow-probability
pip install pyro-ppl  # For variational inference
pip install econml dowhy causalnex  # Causal inference libraries
pip install pycox lifelines scikit-survival  # Survival analysis
pip install shap lime  # Interpretability
pip install pandas numpy scipy scikit-learn
pip install matplotlib seaborn plotly
```

**R Integration**:
```r
# Install reticulate for R-Python interface
install.packages("reticulate")

# Configure Python environment
library(reticulate)
use_condaenv("causal_rep")
```

### File Structure

```
Survival-Analysis/
├── scripts/
│   ├── python/
│   │   ├── 00_data_preprocessing.py      # Data loading and preprocessing
│   │   ├── 01_cevae_survival.py          # CEVAE implementation
│   │   ├── 02_deep_cfr_survival.py       # Deep counterfactual networks
│   │   ├── 03_neural_causal_survival.py  # Neural survival models
│   │   ├── 04_causal_forest_embeddings.py # Hybrid causal forest
│   │   ├── 05_variational_ci.py          # Variational causal inference
│   │   ├── 06_evaluation.py              # Model evaluation
│   │   ├── 07_visualization.py           # Results visualization
│   │   └── utils/
│   │       ├── models.py                  # Neural network architectures
│   │       ├── losses.py                  # Custom loss functions
│   │       └── metrics.py                 # Evaluation metrics
│   └── r/
│       └── 09_run_python_causal.r        # R wrapper for Python scripts
├── models/
│   └── causal_rep/                        # Saved models
├── output/
│   └── causal_rep/                        # Results and tables
└── figures/
    └── causal_rep/                        # Plots and visualizations
```

### Workflow

#### Step 1: Data Preprocessing
```python
# scripts/python/00_data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv("data/covid_sample_data.txt", sep=";")

# Create features
X = data[['age', 'gender', 'wave', 'hosp_id']]
A = (data['Ct'] <= 24).astype(int)  # Treatment: high viral load
Y = data['mort_hospital']
T = data['LoS']

# Train/val/test split
X_train, X_test, A_train, A_test, Y_train, Y_test, T_train, T_test = \
    train_test_split(X, A, Y, T, test_size=0.2, random_state=42)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

#### Step 2: Train CEVAE
```python
# scripts/python/01_cevae_survival.py
from models import CEVAE
import torch

# Initialize model
cevae = CEVAE(
    input_dim=X_train.shape[1],
    latent_dim=10,
    hidden_dims=[64, 32],
    treatment_dim=1,
    outcome_type='survival'
)

# Train
cevae.fit(
    X=X_train_scaled,
    A=A_train,
    Y=Y_train,
    T=T_train,
    epochs=100,
    batch_size=128
)

# Estimate counterfactuals
Y0_hat, Y1_hat = cevae.predict_counterfactuals(X_test_scaled)

# Individual treatment effects
ITE = Y1_hat - Y0_hat
ATE = ITE.mean()
```

#### Step 3: Train Deep CFR
```python
# scripts/python/02_deep_cfr_survival.py
from models import DeepCFR

cfr = DeepCFR(
    input_dim=X_train.shape[1],
    repr_dim=50,
    hidden_dims=[128, 64, 32],
    alpha=1.0  # Balancing regularization weight
)

cfr.fit(X_train_scaled, A_train, Y_train, T_train)

# Predict potential outcomes
S0, S1 = cfr.predict_survival_curves(X_test_scaled, time_points=[7, 14, 28])

# Treatment effects at specific timepoints
TE_day7 = S1[:, 0] - S0[:, 0]
```

#### Step 4: Evaluate
```python
# scripts/python/06_evaluation.py
from metrics import pehe, policy_value, calibration_score

# Compare methods
methods = {
    'CEVAE': cevae,
    'DeepCFR': cfr,
    'IPW': ipw_model  # Traditional method
}

for name, model in methods.items():
    ate = model.estimate_ate(X_test_scaled, A_test)
    print(f"{name} ATE: {ate:.3f}")
```

#### Step 5: Interpret
```python
# scripts/python/07_visualization.py
import shap

# SHAP for representation interpretation
explainer = shap.DeepExplainer(cevae.encoder, X_train_scaled)
shap_values = explainer.shap_values(X_test_scaled)

# Plot
shap.summary_plot(shap_values, X_test)
```

---

## Comparison with Traditional Methods

| Method | Unmeasured Confounding | Non-linearity | Heterogeneity | Uncertainty | Scalability |
|--------|------------------------|---------------|---------------|-------------|-------------|
| **IPW** | ❌ | ⚠️ (manual) | ❌ | ⚠️ (bootstrap) | ✅ |
| **G-formula** | ❌ | ⚠️ (manual) | ⚠️ | ⚠️ (bootstrap) | ✅ |
| **CEVAE** | ✅ | ✅ | ✅ | ✅ (Bayesian) | ✅ |
| **Deep CFR** | ⚠️ | ✅ | ✅ | ⚠️ | ✅ |
| **Neural Survival** | ❌ | ✅ | ✅ | ⚠️ | ✅ |
| **Causal Forest + DL** | ❌ | ✅ | ✅ | ✅ (CI) | ⚠️ |

**Legend**: ✅ = Strong, ⚠️ = Moderate, ❌ = Weak

---

## Evaluation and Validation

### Metrics

1. **ATE Estimation Error**
   - Compare to randomized trial (if available)
   - Bootstrap confidence intervals

2. **PEHE (Precision in Estimation of Heterogeneous Effects)**
   ```
   PEHE = sqrt(E[(τ(x) - τ̂(x))²])
   ```
   - Requires ground truth (semi-synthetic data)

3. **Policy Value**
   - Expected outcome under learned treatment policy
   - Compared to random/observed policy

4. **Calibration**
   - Are predicted treatment effects well-calibrated?
   - Residual analysis

5. **Representation Quality**
   - Balance: Are representations balanced across treatment groups?
   - Predictiveness: Do representations predict outcomes well?
   - Interpretability: Are latent dimensions meaningful?

### Validation Strategies

1. **Cross-validation**: K-fold CV for hyperparameter tuning
2. **Sensitivity analysis**: Vary model specifications
3. **Comparison with benchmarks**: IPW, g-formula, doubly robust
4. **Semi-synthetic experiments**: Add synthetic confounding
5. **Negative controls**: Test on outcomes that should NOT be affected

---

## Key Advantages for COVID-19 Data

### 1. Handling Unmeasured Confounders
**Traditional**: Assumes all confounders measured → **Biased estimates**
**Causal Rep Learning**: Learns latent confounders from proxies → **Less biased**

**Example**:
- Unmeasured: Disease severity, comorbidity index
- CEVAE learns latent "frailty" from age, Ct, hospital patterns

### 2. Complex Interactions
**Traditional**: Manual interaction terms (age × Ct)
**Causal Rep Learning**: Automatically discovers interactions in hidden layers

**Example**:
- Neural nets discover that effect of viral load depends on age × hospital × wave in non-linear ways

### 3. Individual-Level Predictions
**Traditional**: Population-level ATE only
**Causal Rep Learning**: Individual treatment effects (ITE) for personalized medicine

**Example**:
- Patient A: ITE = +20% mortality reduction from antiviral
- Patient B: ITE = +5% (less benefit due to latent factors)

### 4. Counterfactual Reasoning
**Traditional**: Predict under observed treatment only
**Causal Rep Learning**: Generate full counterfactual distributions

**Example**:
- What would patient's survival curve be if they had low viral load?
- Full distribution, not just point estimate

---

## Limitations and Considerations

### 1. Identifiability Assumptions
- Still require **ignorability** in latent space
- Proxy variable assumption may not hold
- Results sensitive to architecture choices

### 2. Computational Cost
- Training deep models is expensive (GPUs recommended)
- Hyperparameter tuning requires extensive search
- May be overkill for small datasets (<10k observations)

### 3. Interpretability
- Black-box models harder to interpret than linear models
- Need post-hoc explanation methods (SHAP, LIME)
- Latent dimensions may not have clear meaning

### 4. Validation Challenges
- Hard to validate causal claims without RCT
- Semi-synthetic experiments may not reflect reality
- Model selection criteria unclear

---

## Next Steps

### Immediate (Phase 1)
1. ✅ Set up Python environment
2. ✅ Implement data preprocessing pipeline
3. ✅ Train CEVAE on COVID-19 data
4. ✅ Evaluate against traditional IPW/g-formula

### Medium-term (Phase 2)
5. Implement Deep CFR and neural survival models
6. Hyperparameter tuning and model selection
7. Comprehensive evaluation on held-out data
8. Sensitivity analyses

### Long-term (Phase 3)
9. Deploy best models for treatment effect prediction
10. Generate patient-level treatment recommendations
11. Validate findings with clinical experts
12. Prepare manuscript with causal representation learning results

---

## References

### Foundational Papers

1. **Louizos et al. (2017)** - "Causal Effect Inference with Deep Latent-Variable Models"
   - CEVAE framework

2. **Shalit et al. (2017)** - "Estimating individual treatment effect"
   - Deep counterfactual representations

3. **Kaddour et al. (2022)** - "Causal Machine Learning: A Survey and Open Problems"
   - Comprehensive review

4. **Curth & van der Schaar (2021)** - "Nonparametric Estimation of Heterogeneous Treatment Effects"
   - Neural causal models

5. **Künzel et al. (2019)** - "Metalearners for estimating heterogeneous treatment effects"
   - S-learner, T-learner, X-learner

### Software

- **PyTorch**: Deep learning framework
- **EconML**: Causal ML library (Microsoft)
- **DoWhy**: Causal inference framework (Microsoft)
- **CausalNex**: Bayesian networks for causality (QuantumBlack)
- **Pyro**: Probabilistic programming (Uber)
- **pycox**: Neural survival models

---

**Document Version**: 1.0
**Last Updated**: 2025-10-21
**Author**: Geoffrey Manda
**License**: CC BY 4.0
