# Causal Inference Framework for COVID-19 Survival Analysis

## Overview

This document outlines a **causal inference framework** for analyzing the COVID-19 patient survival data. While the current analysis uses associational methods (Cox regression, Kaplan-Meier curves), this framework enables us to answer **causal questions** such as:

- "What would happen to mortality if we could reduce viral load?"
- "What is the effect of being admitted during different pandemic waves?"
- "How much of the age-mortality relationship is direct versus mediated?"

## Table of Contents

1. [Causal vs Associational Inference](#causal-vs-associational-inference)
2. [Proposed Causal Estimands](#proposed-causal-estimands)
3. [Identification Strategies](#identification-strategies)
4. [Directed Acyclic Graphs (DAGs)](#directed-acyclic-graphs-dags)
5. [Statistical Methods](#statistical-methods)
6. [Implementation Guide](#implementation-guide)
7. [Limitations and Sensitivity Analysis](#limitations-and-sensitivity-analysis)

---

## Causal vs Associational Inference

### Associational Analysis (Current Approach)

Current methods answer questions like:
- "Are patients with high viral load more likely to die?" (descriptive)
- "Is age associated with mortality?" (predictive)

**Limitations:**
- Cannot distinguish causation from confounding
- Cannot answer "what if" questions
- Results may be misleading for policy decisions

### Causal Analysis (Proposed Framework)

Causal methods answer questions like:
- "Would reducing viral load reduce mortality?" (interventional)
- "What would mortality be if all patients were young?" (counterfactual)

**Advantages:**
- Guides interventions and policy
- Quantifies effects of hypothetical changes
- Makes assumptions explicit (via DAGs)

---

## Proposed Causal Estimands

### Estimand 1: Effect of Viral Load on Mortality

#### Causal Question
> What is the causal effect of high viral load (low Ct value) versus low viral load (high Ct value) on hospital mortality?

#### Target Estimand
**Average Treatment Effect (ATE):**
```
ATE = E[Y(1) - Y(0)]
```
where:
- Y(a) = potential outcome under Ct level a
- For binary treatment: ATE = P(Mortality | do(Ct = High)) - P(Mortality | do(Ct = Low))

#### Why This Matters
- **Clinical**: Informs early intervention strategies (e.g., antiviral timing)
- **Public Health**: Guides testing and isolation protocols
- **Biological**: Quantifies dose-response relationship

#### Variables
- **Treatment**: Ct value (continuous) or ct_gp (categorical: Strongly/Moderately/Weakly positive)
- **Outcome**: Hospital mortality (mort_hospital)
- **Time scale**: Length of stay (LoS)

#### Confounders to Adjust For
1. **Age**: Older patients may have both worse outcomes and different viral loads
2. **Gender**: Biological differences in immune response
3. **Wave**: Viral variants and treatment protocols changed over time
4. **Hospital**: Resource availability, quality of care varies
5. **Disease severity** (partially unmeasured): Sicker patients may have higher viral loads

#### Challenges
- **Selection bias**: Ct measured at admission, not randomized
- **Timing**: Viral load changes over time post-infection
- **Unmeasured confounding**: Comorbidities, exact disease severity, time from symptom onset
- **Competing risks**: Discharge versus death

#### Proposed Methods
1. **Inverse Probability Weighting (IPW)**: Weight observations to create pseudo-randomized sample
2. **G-formula (standardization)**: Predict survival under each treatment scenario
3. **Doubly Robust Estimation**: Combine IPW and outcome modeling for robustness
4. **Instrumental Variables**: If available (e.g., testing protocol changes)
5. **Sensitivity Analysis**: Quantify impact of unmeasured confounding

---

### Estimand 2: Effect of Pandemic Wave on Mortality

#### Causal Question
> What would mortality rates be if all patients had been admitted during Wave 3 versus Wave 1?

#### Target Estimand
**ATE for Wave Comparison:**
```
ATE(Wave 3 vs Wave 1) = E[Y(Wave=3) - Y(Wave=1)]
```

#### Why This Matters
- **Viral variants** had different virulence (Alpha, Delta, Omicron)
- **Treatment protocols** improved (corticosteroids, antivirals)
- **Healthcare capacity** varied (ventilator availability, staffing)
- **Population immunity** changed (vaccination, prior infection)

#### Identification Challenges
1. **Time confounding**: Wave assignment is not random (unidirectional time)
2. **Compositional changes**: Patient characteristics may differ across waves
3. **Concurrent changes**: Multiple factors changed simultaneously
4. **Selection bias**: Testing and hospitalization criteria evolved

#### Confounders
- Age and gender distribution (may shift across waves)
- Baseline viral load (different variants)
- Hospital capacity indicators
- Testing policies (selection into sample)

#### Proposed Methods
1. **IPW with baseline confounders**: Adjust for patient characteristics
2. **Marginal Structural Models (MSMs)**: Handle time-varying confounding
3. **Difference-in-Differences**: Compare trends if control group available
4. **Interrupted Time Series**: Assess before-after wave transitions
5. **Regression Discontinuity**: If sharp transition points exist

---

### Estimand 3: Direct and Indirect Effects of Age

#### Causal Question
> How much of the age effect on mortality is:
> - (a) **Direct** (biological frailty, immune senescence)
> - (b) **Indirect** through viral load
> - (c) **Indirect** through unmeasured pathways (comorbidities)

#### Target Estimands
**Mediation Analysis:**
```
Total Effect (TE) = E[Y(a) - Y(a')]
Natural Direct Effect (NDE) = E[Y(a, M(a')) - Y(a', M(a'))]
Natural Indirect Effect (NIE) = E[Y(a, M(a)) - Y(a, M(a'))]
```
where:
- a, a' = age levels (e.g., 80 vs 40)
- M = mediator (Ct value)

**Decomposition:** TE = NDE + NIE

#### Example Interpretation
If we find:
- Total effect of age 80 vs 40 on mortality = **35 percentage points**
- Direct effect (not through Ct) = **28 percentage points** (80%)
- Indirect effect (through Ct) = **7 percentage points** (20%)

**Conclusion**: Most of age effect is direct (biological), not mediated by viral load.

#### Identification Assumptions
1. No unmeasured exposure-outcome confounding
2. No unmeasured mediator-outcome confounding
3. No unmeasured exposure-mediator confounding
4. No mediator-outcome confounder affected by exposure

#### Methods
1. Regression-based mediation with survival outcomes
2. IPW for mediation
3. G-formula for mediation
4. **Sensitivity analysis crucial** (unmeasured confounding likely)

---

### Estimand 4: Hospital Quality Effect

#### Causal Question
> What would happen to patient outcomes if low-performing hospitals adopted practices of high-performing hospitals?

#### Target Estimand
**Hospital-level ATE:**
```
E[Y(Hospital = Best) - Y(Hospital = Worst)]
```
for a randomly selected patient

#### Challenges
1. **Non-random assignment**: Geographic proximity, referral patterns
2. **Case-mix variation**: Patient populations differ across hospitals
3. **Multi-level effects**: Resources, expertise, protocols, patient population

#### What Makes Hospital Effects Causal?
Hospital effects may be due to:
- Resources and capacity (beds, equipment)
- Staff expertise and training
- Treatment protocols
- **OR** patient population characteristics (confounding)

#### Confounders
- Patient age, gender, viral load at admission
- Disease severity (partially captured by Ct)
- Socioeconomic factors (unmeasured)
- Distance to hospital (selection)

#### Proposed Methods
1. **Hospital Fixed Effects**: Control time-invariant hospital factors
2. **Random Effects + Propensity Matching**: Model hospital selection
3. **Instrumental Variables**: Geographic distance (affects hospital choice, not directly outcome)
4. **Regression Discontinuity**: Catchment area boundaries
5. **Difference-in-Differences**: If policy change at some hospitals

---

### Estimand 5: Effect Modification by Gender

#### Causal Questions
1. Does the effect of viral load on mortality **differ** for males versus females?
2. Does the effect of age on mortality **differ** by gender?

#### Target Estimands
**Conditional Average Treatment Effect (CATE):**
```
CATE(Male) = E[Y(1) - Y(0) | Gender = Male]
CATE(Female) = E[Y(1) - Y(0) | Gender = Female]

Effect Modification = CATE(Male) - CATE(Female)
```

#### Example
Effect of high viral load on mortality:
- **Males**: HR = 2.5 (95% CI: 2.0-3.1)
- **Females**: HR = 1.8 (95% CI: 1.4-2.3)
- **Interaction p-value** < 0.05

**Interpretation**: High viral load has a **stronger effect** on mortality in males than females.

#### Methods
1. **Stratified Analysis**: Separate models by gender
2. **Interaction Terms**: Include treatment × gender in regression
3. **Causal Forests**: Machine learning for heterogeneous effects
4. **Meta-learners**: S-learner, T-learner, X-learner

---

## Directed Acyclic Graphs (DAGs)

DAGs make **causal assumptions explicit** and help identify:
- Which variables to adjust for (confounders)
- Which variables NOT to adjust for (colliders, mediators)
- What assumptions are required for causal identification

### DAG 1: Viral Load Effect on Mortality

```
TimeFromOnset     Age ---------> Comorbidities
      |            |                   |
      v            v                   v
   Ct Value --> Severity ----------> Mortality
      |            |                   ^
      v            v                   |
    LoS -----> Hospital ---------------+
```

**Key Insights:**
- Adjust for: Age, Gender, Wave, Hospital
- DO NOT adjust for: LoS (collider), Severity (mediator if asking total effect)
- Unmeasured: Comorbidities, Time from onset

**Minimal Adjustment Set**: {Age, Gender, Wave, Hospital}

### DAG 2: Pandemic Wave Effect

```
           Wave
          /  |  \
         /   |   \
        v    v    v
   Variant  Treatment  Hospital
        \    |    /
         v   v   v
        Mortality
```

**Challenges:**
- Wave affects BOTH exposure and outcome pathways
- Selection bias from changing testing policies
- Multiple simultaneous changes (variants + treatment + capacity)

**Adjustment Strategy**: Control for baseline patient characteristics, but recognize residual confounding from unmeasured factors.

### DAG 3: Age Effect with Mediation

```
Age --> Immune Function --> Mortality
 |           |                 ^
 |           v                 |
 +-----> Comorbidities --------+
 |                             |
 +---------> Ct Value ---------+
```

**Paths from Age to Mortality:**
1. Age → Mortality (direct effect)
2. Age → Ct → Mortality (mediated by viral load)
3. Age → Comorbidities → Mortality (unmeasured mediation)
4. Age → Immune Function → Mortality (unmeasured mediation)

**For Total Effect**: Adjust for baseline confounders, NOT mediators
**For Direct Effect**: Adjust for mediator (Ct) plus confounders

### DAG 4: Hospital Effect with Instrumental Variable

```
Geography --> Hospital --> Mortality
                 ^            ^
                 |            |
            SES (unmeasured)  |
                 |            |
                 +------------+
                      Age, Gender, Ct
```

**Instrumental Variable Assumptions:**
1. **Relevance**: Geography strongly predicts Hospital
2. **Exclusion**: Geography affects Mortality ONLY through Hospital
3. **Exchangeability**: Geography independent of confounders

**Potential Instruments:**
- Geographic distance to hospitals
- Hospital catchment area boundaries
- Ambulance routing protocols

---

## Statistical Methods

### 1. Inverse Probability Weighting (IPW)

**Concept**: Create a pseudo-population where treatment is independent of confounders.

**Steps:**
1. Estimate propensity score: P(Treatment = 1 | Confounders)
2. Calculate weights:
   - Treated: w = 1 / PS
   - Control: w = 1 / (1 - PS)
3. Fit weighted outcome model

**Advantages:**
- Estimates marginal effects (population-level)
- Flexible (works with any outcome model)

**Disadvantages:**
- Unstable if extreme weights
- Requires positivity (overlap)

**Implementation:**
```r
# Propensity score model
ps_model <- glm(Treatment ~ Age + Gender + Wave,
                family = binomial, data = data)

# Weights
data$weight <- ifelse(data$Treatment == 1,
                     1 / ps,
                     1 / (1 - ps))

# Weighted Cox model
cox_ipw <- coxph(Surv(time, event) ~ Treatment,
                 weights = weight, data = data)
```

### 2. G-Formula (Standardization)

**Concept**: Predict outcomes under each treatment scenario, then average over confounder distribution.

**Steps:**
1. Fit outcome model: E[Y | Treatment, Confounders]
2. Set Treatment = 1 for all, predict outcomes, average
3. Set Treatment = 0 for all, predict outcomes, average
4. Compare averages

**Advantages:**
- Intuitive (simulates intervention)
- No extreme weights

**Disadvantages:**
- Requires correct outcome model
- More complex for time-to-event

**Implementation:**
```r
# Outcome model
model <- coxph(Surv(time, event) ~ Treatment + Age + Gender,
               data = data)

# Counterfactual datasets
data1 <- data %>% mutate(Treatment = 1)
data0 <- data %>% mutate(Treatment = 0)

# Predict survival
surv1 <- survfit(model, newdata = data1)
surv0 <- survfit(model, newdata = data0)

# Average treatment effect
ATE <- mean(surv1$surv) - mean(surv0$surv)
```

### 3. Doubly Robust Estimation

**Concept**: Combine IPW and outcome modeling for robustness.

**Key Property**: Correct if EITHER propensity model OR outcome model is correct (not necessarily both).

**Implementation:**
```r
# Augmented IPW estimator
library(stdReg)
model <- coxph(Surv(time, event) ~ Treatment + Age + Gender,
               data = data)

ate <- stdCoxph(model, data = data,
                X = "Treatment", x = c(0, 1))
```

### 4. Instrumental Variable (IV) Methods

**When to Use**: Strong unmeasured confounding, but valid instrument available

**Example Instrument**: Geographic distance to high-quality hospital

**Two-Stage Approach:**
1. First stage: Hospital ~ Geography + Controls
2. Second stage: Mortality ~ Predicted(Hospital) + Controls

**Implementation:**
```r
library(ivtools)
# First stage
stage1 <- lm(Hospital ~ Geography + Age + Gender, data = data)

# Second stage
data$Hospital_pred <- predict(stage1)
stage2 <- coxph(Surv(time, event) ~ Hospital_pred + Age + Gender,
                data = data)
```

### 5. Sensitivity Analysis

**E-value**: Minimum strength of unmeasured confounding required to explain away observed effect.

**Implementation:**
```r
library(EValue)

# Observed hazard ratio
HR <- 1.8

# Calculate E-value
evalue(HR(HR, rare = FALSE))
```

**Interpretation**: An unmeasured confounder would need HR ≈ 2.5 with both treatment and outcome to nullify observed HR of 1.8.

---

## Implementation Guide

### Step 1: Define Research Question
- State causal question precisely
- Identify treatment, outcome, time scale
- Specify target population

### Step 2: Draw DAG
- List all relevant variables (measured and unmeasured)
- Draw causal arrows based on domain knowledge
- Identify confounders, mediators, colliders

### Step 3: Identify Adjustment Set
- Use `dagitty` package in R
- Find minimal sufficient adjustment set
- Check for:
  - Confounders (adjust for)
  - Mediators (adjust for direct effect only)
  - Colliders (DO NOT adjust for)

### Step 4: Check Assumptions
**Consistency**: Well-defined interventions
**Positivity**: Overlap in propensity scores
**Exchangeability**: No unmeasured confounding (after adjustment)

### Step 5: Estimate Causal Effect
- Use IPW, g-formula, or doubly robust methods
- Report both point estimates and uncertainty
- Compare to unadjusted estimates

### Step 6: Sensitivity Analysis
- Calculate E-values
- Vary model specifications
- Check robustness to:
  - Different confounder sets
  - Different functional forms
  - Trimming extreme weights

### Step 7: Interpret Causally
- Report estimand clearly
- State limitations
- Distinguish causal from associational findings

---

## Limitations and Sensitivity Analysis

### Data Limitations

**Missing Variables:**
- Comorbidities (diabetes, hypertension, obesity)
- Disease severity markers at admission
- Time from symptom onset
- Vaccination status (later waves)
- Treatment received (antivirals, steroids, oxygen)

**Impact**: Residual confounding likely; estimates may be biased.

### Assumption Violations

**Unmeasured Confounding**
- **Concern**: Comorbidities affect both viral load and mortality
- **Sensitivity**: Calculate E-values; conduct quantitative bias analysis

**Positivity**
- **Concern**: Some covariate combinations may have no treated/control observations
- **Check**: Examine propensity score overlap
- **Solution**: Trim non-overlapping regions

**Consistency**
- **Concern**: "High viral load" may mean different things in different contexts
- **Solution**: Clearly define intervention; consider multiple operationalizations

### Recommended Sensitivity Analyses

1. **E-values**: Quantify robustness to unmeasured confounding
2. **Multiple adjustment sets**: Try different confounder combinations
3. **Subgroup analyses**: Check consistency across subpopulations
4. **Alternative methods**: Compare IPW, g-formula, regression results
5. **Negative controls**: Use outcomes that SHOULD NOT be affected by treatment

---

## Next Steps

### Immediate Actions
1. Review DAGs with clinical domain experts
2. Conduct exploratory data analysis to check assumptions
3. Implement IPW for viral load estimand (Estimand 1)
4. Compare causal vs associational estimates

### Medium-Term Goals
1. Implement g-formula for wave effects (Estimand 2)
2. Conduct mediation analysis for age (Estimand 3)
3. Explore hospital IV analysis (Estimand 4)
4. Test for effect modification by gender (Estimand 5)

### Long-Term Objectives
1. Supplement data with medical records (comorbidities, treatments)
2. Develop comprehensive sensitivity analysis framework
3. Write up findings with causal interpretation
4. Submit for peer review / publication

---

## References

### Key Papers on Causal Inference in Survival Analysis

1. **Hernán MA (2010).** "The hazards of hazard ratios." *Epidemiology*, 21(1):13-15.
   - Discusses causal interpretation of hazard ratios

2. **Cole SR, Hernán MA (2008).** "Constructing inverse probability weights for marginal structural models." *Am J Epidemiol*, 168(6):656-664.
   - IPW methods for time-to-event data

3. **Young JG, et al. (2020).** "Identification, estimation and approximation of risk under interventions that depend on the natural value of treatment using observational data." *Epidemiologic Methods*, 9(1).
   - G-formula for causal effects

4. **VanderWeele TJ, Ding P (2017).** "Sensitivity analysis in observational research: introducing the E-value." *Ann Intern Med*, 167(4):268-274.
   - E-value methodology

5. **Pearl J (2009).** *Causality: Models, Reasoning, and Inference.* Cambridge University Press.
   - Foundational text on causal DAGs

### R Packages

- `dagitty`: DAG construction and analysis
- `ggdag`: DAG visualization
- `ipw`: Inverse probability weighting
- `stdReg`: Standardization and doubly robust methods
- `EValue`: Sensitivity analysis
- `mediation`: Mediation analysis
- `survival`, `survminer`: Survival analysis

---

## Contact

For questions about this causal inference framework:
- **Author**: Geoffrey Manda
- **GitHub**: [https://github.com/GeoffreyManda](https://github.com/GeoffreyManda)
- **LinkedIn**: [https://www.linkedin.com/in/geoffreymanda/](https://www.linkedin.com/in/geoffreymanda/)

---

**Document Version**: 1.0
**Last Updated**: 2025-10-21
**License**: CC BY 4.0
