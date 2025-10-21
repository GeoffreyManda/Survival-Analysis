# 07_causal_inference_estimands.R
# COVID-19 Survival Analysis - Causal Inference Framework
# This script outlines causal estimands and identification strategies
# for answering causal questions with the COVID-19 survival data

# Load required packages
if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  # Core packages
  dplyr, tidyverse, here,

  # Causal inference packages
  dagitty,        # DAG construction and analysis
  ggdag,          # DAG visualization

  # Survival analysis with causal focus
  survival, survminer,

  # Causal survival methods
  timereg,        # Additive hazards models
  ipw,            # Inverse probability weighting

  # Visualization
  ggplot2, patchwork, gridExtra
)

#==============================================================================#
#                    CAUSAL INFERENCE FRAMEWORK FOR COVID-19                   #
#==============================================================================#

# The current analysis uses associational/predictive models (Cox regression,
# Kaplan-Meier). While these identify statistical relationships, they don't
# necessarily answer causal questions like:
#
# - "What would happen to mortality if we could intervene to reduce viral load?"
# - "What is the effect of being admitted during wave 3 vs wave 1?"
# - "How much of the age-mortality relationship is causal vs confounded?"
#
# This script proposes a causal inference framework to answer such questions.

#==============================================================================#
#                        PROPOSED CAUSAL ESTIMANDS                             #
#==============================================================================#

cat("========================================\n")
cat("PROPOSED CAUSAL ESTIMANDS\n")
cat("========================================\n\n")

# Load preprocessed data
load(here("data/processed_covid_data.RData"))

#------------------------------------------------------------------------------#
# ESTIMAND 1: Effect of Viral Load (Ct value) on Hospital Mortality
#------------------------------------------------------------------------------#
cat("ESTIMAND 1: Causal Effect of Viral Load on Mortality\n")
cat("----------------------------------------------\n")
cat("Treatment: Ct value (continuous) or ct_gp (categorical)\n")
cat("Outcome: Hospital mortality (mort_hospital)\n")
cat("Time scale: Length of stay (LoS)\n\n")

cat("Causal Question:\n")
cat("  What is the average causal effect of having high viral load\n")
cat("  (low Ct value) vs low viral load (high Ct value) on the\n")
cat("  probability of hospital mortality?\n\n")

cat("Target Estimand:\n")
cat("  ATE = E[Y(1) - Y(0)]\n")
cat("  where Y(a) is potential outcome under Ct level a\n\n")

cat("For binary treatment (Strongly positive vs Weakly positive):\n")
cat("  ATE = P(Mortality | do(Ct = Strongly positive)) - \n")
cat("        P(Mortality | do(Ct = Weakly positive))\n\n")

cat("Identification Assumptions Required:\n")
cat("  1. Consistency: Y = Y(a) when A = a\n")
cat("  2. Exchangeability: Y(a) ⊥ A | L (given confounders L)\n")
cat("  3. Positivity: 0 < P(A = a | L) < 1 for all L\n\n")

cat("Potential Confounders (L):\n")
cat("  - Age (older patients may have worse outcomes and different viral loads)\n")
cat("  - Gender (biological differences in immune response)\n")
cat("  - Wave (viral variants and treatment protocols changed)\n")
cat("  - Hospital (resource availability, quality of care)\n")
cat("  - Time from symptom onset (affects both Ct and prognosis)\n\n")

cat("Challenges:\n")
cat("  - Ct value is measured, not randomized (selection bias)\n")
cat("  - Ct timing: measured at admission, but viral load changes over time\n")
cat("  - Unmeasured confounders: comorbidities, disease severity markers\n")
cat("  - Competing risks: discharge vs death\n\n")

cat("Proposed Methods:\n")
cat("  1. Inverse Probability Weighting (IPW) with Cox models\n")
cat("  2. G-formula (standardization) for survival curves\n")
cat("  3. Doubly robust estimation\n")
cat("  4. Instrumental variable (if available, e.g., testing protocol changes)\n")
cat("  5. Sensitivity analysis for unmeasured confounding\n\n")

#------------------------------------------------------------------------------#
# ESTIMAND 2: Effect of COVID-19 Wave on Mortality
#------------------------------------------------------------------------------#
cat("ESTIMAND 2: Causal Effect of Pandemic Wave on Mortality\n")
cat("----------------------------------------------\n")
cat("Treatment: COVID-19 wave (wave 1, 2, or 3)\n")
cat("Outcome: Hospital mortality (mort_hospital)\n")
cat("Time scale: Length of stay (LoS)\n\n")

cat("Causal Question:\n")
cat("  What would mortality rates be if all patients had been\n")
cat("  admitted during wave 3 vs wave 1?\n\n")

cat("Target Estimand:\n")
cat("  ATE(wave 3 vs wave 1) = E[Y(wave=3) - Y(wave=1)]\n\n")

cat("Why this matters:\n")
cat("  - Different variants had different virulence\n")
cat("  - Treatment protocols improved over time\n")
cat("  - Healthcare system capacity varied across waves\n")
cat("  - Vaccination status changed across waves\n\n")

cat("Identification Challenges:\n")
cat("  - Wave assignment is not random (time is unidirectional)\n")
cat("  - Confounding by time-varying factors:\n")
cat("    * Patient characteristics may differ across waves\n")
cat("    * Virus variants changed\n")
cat("    * Treatment standards evolved\n")
cat("  - Selection bias: who got tested/hospitalized changed\n\n")

cat("Confounders to adjust for:\n")
cat("  - Age distribution (may shift across waves)\n")
cat("  - Gender distribution\n")
cat("  - Viral load (Ct values)\n")
cat("  - Hospital capacity/resources\n")
cat("  - Baseline health status (if available)\n\n")

cat("Proposed Methods:\n")
cat("  1. IPW Cox models adjusting for baseline confounders\n")
cat("  2. Marginal structural models (MSMs) for time-varying confounding\n")
cat("  3. Difference-in-differences (if control group available)\n")
cat("  4. Interrupted time series analysis\n")
cat("  5. Regression discontinuity (if sharp wave transitions)\n\n")

#------------------------------------------------------------------------------#
# ESTIMAND 3: Effect of Age on Mortality (Causal Mediation Analysis)
#------------------------------------------------------------------------------#
cat("ESTIMAND 3: Direct and Indirect Effects of Age on Mortality\n")
cat("----------------------------------------------\n")
cat("Exposure: Age\n")
cat("Mediators: Viral load (Ct), comorbidities (unobserved)\n")
cat("Outcome: Hospital mortality\n\n")

cat("Causal Question:\n")
cat("  How much of the age effect on mortality is:\n")
cat("  (a) Direct (biological frailty, immune function)\n")
cat("  (b) Mediated through viral load\n")
cat("  (c) Mediated through unmeasured factors (comorbidities)\n\n")

cat("Target Estimands:\n")
cat("  - Total Effect (TE): E[Y(a) - Y(a')]\n")
cat("  - Natural Direct Effect (NDE): E[Y(a, M(a')) - Y(a', M(a'))]\n")
cat("  - Natural Indirect Effect (NIE): E[Y(a, M(a)) - Y(a, M(a'))]\n")
cat("  where a = age level, M = mediator\n\n")

cat("Decomposition:\n")
cat("  TE = NDE + NIE\n\n")

cat("Example:\n")
cat("  TE of age 80 vs 40 on mortality = 0.35 (35 percentage points)\n")
cat("  NDE (not through Ct) = 0.28 (80% of effect)\n")
cat("  NIE (through Ct) = 0.07 (20% of effect)\n\n")

cat("Identification Assumptions:\n")
cat("  1. No unmeasured confounding of exposure-outcome\n")
cat("  2. No unmeasured confounding of mediator-outcome\n")
cat("  3. No unmeasured confounding of exposure-mediator\n")
cat("  4. No mediator-outcome confounder affected by exposure\n\n")

cat("Methods:\n")
cat("  1. Regression-based mediation with survival outcomes\n")
cat("  2. IPW for mediation\n")
cat("  3. G-formula for mediation\n")
cat("  4. Sensitivity analysis (unmeasured confounding likely)\n\n")

#------------------------------------------------------------------------------#
# ESTIMAND 4: Hospital Quality Effect (Clustered Treatment)
#------------------------------------------------------------------------------#
cat("ESTIMAND 4: Causal Effect of Hospital Quality on Outcomes\n")
cat("----------------------------------------------\n")
cat("Treatment: Hospital assignment (hosp_id)\n")
cat("Outcome: Mortality and length of stay\n\n")

cat("Causal Question:\n")
cat("  What would happen to patient outcomes if low-performing\n")
cat("  hospitals adopted practices of high-performing hospitals?\n\n")

cat("Target Estimand:\n")
cat("  Hospital-level ATE:\n")
cat("  E[Y(hosp = best) - Y(hosp = worst)] for a random patient\n\n")

cat("Challenges:\n")
cat("  - Hospital assignment not random (geographic, referral patterns)\n")
cat("  - Patient case-mix varies across hospitals\n")
cat("  - Hospital effects may be due to:\n")
cat("    * Resources and capacity\n")
cat("    * Staff expertise\n")
cat("    * Treatment protocols\n")
cat("    * Patient population\n\n")

cat("Confounders:\n")
cat("  - Patient age, gender, viral load at admission\n")
cat("  - Disease severity (partially captured by Ct)\n")
cat("  - Socioeconomic factors (unmeasured)\n")
cat("  - Distance to hospital\n\n")

cat("Methods:\n")
cat("  1. Hospital fixed effects (controls for time-invariant hospital factors)\n")
cat("  2. Hospital random effects with propensity score matching\n")
cat("  3. Instrumental variables (e.g., geographic distance)\n")
cat("  4. Regression discontinuity (e.g., catchment area boundaries)\n")
cat("  5. Difference-in-differences (if policy change at some hospitals)\n\n")

#------------------------------------------------------------------------------#
# ESTIMAND 5: Gender Effect (Potential Effect Modification)
#------------------------------------------------------------------------------#
cat("ESTIMAND 5: Conditional Causal Effects (Effect Modification)\n")
cat("----------------------------------------------\n")
cat("Research Question: Do treatment effects vary by gender?\n\n")

cat("Causal Questions:\n")
cat("  1. Does the effect of viral load on mortality differ for\n")
cat("     males vs females?\n")
cat("  2. Does the effect of age on mortality differ by gender?\n\n")

cat("Target Estimands:\n")
cat("  Conditional Average Treatment Effect (CATE):\n")
cat("  CATE(Male) = E[Y(1) - Y(0) | Gender = Male]\n")
cat("  CATE(Female) = E[Y(1) - Y(0) | Gender = Female]\n\n")

cat("  Effect Modification:\n")
cat("  CATE(Male) - CATE(Female)\n\n")

cat("Example:\n")
cat("  Effect of high viral load on mortality:\n")
cat("  - Among males: HR = 2.5 (95% CI: 2.0-3.1)\n")
cat("  - Among females: HR = 1.8 (95% CI: 1.4-2.3)\n")
cat("  - Interaction p-value < 0.05\n\n")

cat("Methods:\n")
cat("  1. Stratified analysis (separate models by gender)\n")
cat("  2. Interaction terms in regression models\n")
cat("  3. Causal forests for heterogeneous treatment effects\n")
cat("  4. Metalearners (S-learner, T-learner, X-learner)\n\n")

#==============================================================================#
#                     CAUSAL DIRECTED ACYCLIC GRAPHS (DAGs)                   #
#==============================================================================#

cat("\n========================================\n")
cat("CREATING CAUSAL DAGS\n")
cat("========================================\n\n")

# Create output directory for DAGs
dir.create("figures/dags", showWarnings = FALSE, recursive = TRUE)

#------------------------------------------------------------------------------#
# DAG 1: Viral Load Effect on Mortality
#------------------------------------------------------------------------------#
cat("Creating DAG 1: Viral Load (Ct) -> Mortality\n")

dag1 <- dagify(
  Mortality ~ Ct + Age + Gender + Wave + Hospital + Severity,
  Ct ~ Age + Severity + Wave + TimeFromOnset,
  LoS ~ Mortality + Ct + Age + Hospital + Severity,
  Severity ~ Age + Gender + Comorbidities,
  Hospital ~ Geography + Severity,

  # Unmeasured variables
  Comorbidities ~ Age + Gender,
  TimeFromOnset ~ Severity,

  exposure = "Ct",
  outcome = "Mortality",

  labels = c(
    "Ct" = "Viral Load\n(Ct value)",
    "Mortality" = "Hospital\nMortality",
    "Age" = "Age",
    "Gender" = "Gender",
    "Wave" = "Pandemic\nWave",
    "Hospital" = "Hospital",
    "Severity" = "Disease\nSeverity",
    "LoS" = "Length of\nStay",
    "Comorbidities" = "Comorbidities\n(unmeasured)",
    "TimeFromOnset" = "Time from\nSymptom Onset\n(unmeasured)",
    "Geography" = "Geography\n(unmeasured)"
  ),

  coords = list(
    x = c(Ct = 1, Mortality = 3, Age = 1, Gender = 0.5, Wave = 1.5,
          Hospital = 2.5, Severity = 2, LoS = 3.5, Comorbidities = 1,
          TimeFromOnset = 0.5, Geography = 2),
    y = c(Ct = 2, Mortality = 2, Age = 3, Gender = 3.5, Wave = 3,
          Hospital = 3, Severity = 1, LoS = 1, Comorbidities = 4,
          TimeFromOnset = 1, Geography = 4)
  )
)

# Identify adjustment sets
adj_set_ct <- adjustmentSets(dag1, exposure = "Ct", outcome = "Mortality",
                             type = "minimal")
cat("Minimal adjustment sets for Ct -> Mortality:\n")
print(adj_set_ct)
cat("\n")

# Plot DAG
p1 <- ggdag(dag1, text = FALSE, use_labels = "label") +
  theme_dag() +
  labs(title = "DAG 1: Causal Effect of Viral Load on Mortality",
       subtitle = "Red nodes = unmeasured variables") +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5))

ggsave("figures/dags/dag1_viral_load_mortality.png", p1,
       width = 12, height = 8, dpi = 300)

# Plot DAG with adjustment set highlighted
p1b <- ggdag_adjustment_set(dag1, exposure = "Ct", outcome = "Mortality",
                            text = FALSE, use_labels = "label") +
  theme_dag() +
  labs(title = "DAG 1: Adjustment Set for Viral Load Effect",
       subtitle = "Green = variables to adjust for") +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5))

ggsave("figures/dags/dag1_adjustment_set.png", p1b,
       width = 12, height = 8, dpi = 300)

#------------------------------------------------------------------------------#
# DAG 2: Pandemic Wave Effect
#------------------------------------------------------------------------------#
cat("Creating DAG 2: Pandemic Wave -> Mortality\n")

dag2 <- dagify(
  Mortality ~ Wave + Age + Gender + Ct + Treatment + Hospital,
  Ct ~ Wave + Variant + Age,
  Treatment ~ Wave + Hospital + Severity,
  Hospital ~ Wave + Geography,
  Testing ~ Wave,
  Selection ~ Testing + Severity + Wave,
  Variant ~ Wave,
  Severity ~ Age + Gender + Comorbidities,

  exposure = "Wave",
  outcome = "Mortality",

  labels = c(
    "Wave" = "Pandemic\nWave",
    "Mortality" = "Hospital\nMortality",
    "Ct" = "Viral Load",
    "Age" = "Age",
    "Gender" = "Gender",
    "Treatment" = "Treatment\nProtocol",
    "Hospital" = "Hospital\nCapacity",
    "Variant" = "Viral\nVariant",
    "Testing" = "Testing\nPolicy",
    "Selection" = "Selection\nBias",
    "Severity" = "Disease\nSeverity",
    "Geography" = "Geography\n(unmeasured)",
    "Comorbidities" = "Comorbidities\n(unmeasured)"
  )
)

# Identify adjustment sets
adj_set_wave <- adjustmentSets(dag2, exposure = "Wave", outcome = "Mortality",
                               type = "minimal")
cat("Minimal adjustment sets for Wave -> Mortality:\n")
print(adj_set_wave)
cat("\n")

# Plot DAG
p2 <- ggdag(dag2, text = FALSE, use_labels = "label") +
  theme_dag() +
  labs(title = "DAG 2: Causal Effect of Pandemic Wave on Mortality",
       subtitle = "Accounts for time-varying confounding and selection bias") +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5))

ggsave("figures/dags/dag2_wave_mortality.png", p2,
       width = 12, height = 8, dpi = 300)

#------------------------------------------------------------------------------#
# DAG 3: Age Effect with Mediation
#------------------------------------------------------------------------------#
cat("Creating DAG 3: Age -> Mortality (with mediation)\n")

dag3 <- dagify(
  Mortality ~ Age + Ct + Comorbidities + ImmuneFunction + Hospital,
  Ct ~ Age + ImmuneFunction + Comorbidities,
  Comorbidities ~ Age,
  ImmuneFunction ~ Age,
  LoS ~ Mortality + Age + Hospital,

  exposure = "Age",
  outcome = "Mortality",

  labels = c(
    "Age" = "Age",
    "Mortality" = "Hospital\nMortality",
    "Ct" = "Viral Load\n(mediator)",
    "Comorbidities" = "Comorbidities\n(mediator,\nunmeasured)",
    "ImmuneFunction" = "Immune\nFunction\n(mediator,\nunmeasured)",
    "Hospital" = "Hospital",
    "LoS" = "Length of\nStay"
  )
)

# Plot DAG
p3 <- ggdag(dag3, text = FALSE, use_labels = "label") +
  theme_dag() +
  labs(title = "DAG 3: Age Effect on Mortality with Mediation",
       subtitle = "Ct is a measured mediator; comorbidities and immune function are unmeasured") +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5))

ggsave("figures/dags/dag3_age_mediation.png", p3,
       width = 12, height = 8, dpi = 300)

# Identify paths
cat("All paths from Age to Mortality:\n")
paths_age <- paths(dag3, from = "Age", to = "Mortality")
print(paths_age)
cat("\n")

#------------------------------------------------------------------------------#
# DAG 4: Hospital Effect (Instrumental Variable Structure)
#------------------------------------------------------------------------------#
cat("Creating DAG 4: Hospital Effect on Mortality\n")

dag4 <- dagify(
  Mortality ~ Hospital + Age + Gender + Ct + Severity,
  Hospital ~ Geography + SES,
  Ct ~ Severity + Age,
  Severity ~ Age + Gender + Comorbidities,

  # Geography as potential instrument (affects hospital but not outcome directly)
  Geography ~ NULL,  # exogenous

  exposure = "Hospital",
  outcome = "Mortality",

  labels = c(
    "Hospital" = "Hospital\nQuality",
    "Mortality" = "Hospital\nMortality",
    "Geography" = "Geographic\nDistance\n(IV candidate)",
    "SES" = "Socioeconomic\nStatus\n(unmeasured)",
    "Age" = "Age",
    "Gender" = "Gender",
    "Ct" = "Viral Load",
    "Severity" = "Disease\nSeverity",
    "Comorbidities" = "Comorbidities\n(unmeasured)"
  )
)

# Plot DAG
p4 <- ggdag(dag4, text = FALSE, use_labels = "label") +
  theme_dag() +
  labs(title = "DAG 4: Hospital Effect on Mortality",
       subtitle = "Geographic distance as potential instrumental variable") +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5))

ggsave("figures/dags/dag4_hospital_iv.png", p4,
       width = 12, height = 8, dpi = 300)

# Check if Geography is a valid instrument
cat("Checking instrumental variable assumptions for Geography:\n")
cat("  1. Relevance: Geography -> Hospital (must be strong)\n")
cat("  2. Exclusion: Geography -> Mortality only through Hospital\n")
cat("  3. Exchangeability: Geography independent of confounders\n\n")

#==============================================================================#
#                    IMPLEMENTATION: EXAMPLE CAUSAL ANALYSIS                   #
#==============================================================================#

cat("\n========================================\n")
cat("EXAMPLE IMPLEMENTATION\n")
cat("========================================\n\n")

cat("Example 1: IPW Estimation of Viral Load Effect on Mortality\n")
cat("-------------------------------------------------------------\n\n")

# Prepare data (remove missing values)
analysis_data <- Covid_data %>%
  filter(!is.na(Ct), !is.na(mort_hospital), !is.na(age),
         !is.na(gender), !is.na(wave)) %>%
  mutate(
    # Create binary treatment: high vs low viral load
    high_viral_load = ifelse(ct_gp == "Strongly positive", 1, 0),
    high_viral_load = ifelse(ct_gp == "Weakly Positive", 0, high_viral_load)
  ) %>%
  filter(!is.na(high_viral_load))

cat("Sample size for analysis:", nrow(analysis_data), "\n")
cat("Treatment distribution:\n")
print(table(analysis_data$ct_gp))
cat("\n")

# Step 1: Estimate propensity scores (probability of high viral load given confounders)
cat("Step 1: Estimating propensity scores...\n")

ps_model <- glm(
  high_viral_load ~ age + gender + wave + hosp_id,
  data = analysis_data,
  family = binomial(link = "logit")
)

# Get propensity scores
analysis_data$ps <- predict(ps_model, type = "response")

# Check overlap (positivity assumption)
cat("Propensity score summary by treatment group:\n")
cat("High viral load (treated):\n")
print(summary(analysis_data$ps[analysis_data$high_viral_load == 1]))
cat("Low viral load (control):\n")
print(summary(analysis_data$ps[analysis_data$high_viral_load == 0]))
cat("\n")

# Plot propensity score distribution
ps_plot <- ggplot(analysis_data, aes(x = ps, fill = factor(high_viral_load))) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("#0073C2FF", "#EFC000FF"),
                    labels = c("Low Viral Load", "High Viral Load")) +
  labs(title = "Propensity Score Distribution by Treatment Group",
       x = "Propensity Score (Probability of High Viral Load)",
       y = "Density",
       fill = "Treatment") +
  theme_bw() +
  theme(legend.position = "bottom")

ggsave("figures/dags/propensity_scores.png", ps_plot,
       width = 10, height = 6, dpi = 300)

# Step 2: Calculate IPW weights
cat("Step 2: Calculating IPW weights...\n")

analysis_data <- analysis_data %>%
  mutate(
    # ATE weights
    ipw_weight = ifelse(high_viral_load == 1,
                       1 / ps,           # treated
                       1 / (1 - ps)),    # control

    # Stabilized weights (more stable)
    ipw_weight_stab = ifelse(high_viral_load == 1,
                            mean(high_viral_load) / ps,
                            (1 - mean(high_viral_load)) / (1 - ps))
  )

cat("IPW weight summary:\n")
print(summary(analysis_data$ipw_weight_stab))
cat("\n")

# Check for extreme weights
cat("Proportion of weights > 10:",
    mean(analysis_data$ipw_weight_stab > 10, na.rm = TRUE), "\n\n")

# Step 3: Fit weighted Cox model
cat("Step 3: Fitting IPW-weighted Cox model...\n")

cox_ipw <- coxph(
  Surv(LoS, mort_hospital) ~ high_viral_load,
  data = analysis_data,
  weights = ipw_weight_stab,
  robust = TRUE
)

cat("IPW-weighted Cox model results:\n")
print(summary(cox_ipw))
cat("\n")

# Compare to unadjusted model
cox_unadj <- coxph(
  Surv(LoS, mort_hospital) ~ high_viral_load,
  data = analysis_data
)

cat("Unadjusted Cox model results:\n")
print(summary(cox_unadj))
cat("\n")

# Step 4: Estimate survival curves using g-formula (standardization)
cat("Step 4: G-formula for marginal survival curves...\n")

# Fit outcome model
outcome_model <- coxph(
  Surv(LoS, mort_hospital) ~ high_viral_load + age + gender + wave + hosp_id,
  data = analysis_data
)

# Create datasets with treatment set to 0 and 1 for all observations
data_treat1 <- analysis_data %>% mutate(high_viral_load = 1)
data_treat0 <- analysis_data %>% mutate(high_viral_load = 0)

# Predict survival for each scenario
surv_treat1 <- survfit(outcome_model, newdata = data_treat1)
surv_treat0 <- survfit(outcome_model, newdata = data_treat0)

# Plot counterfactual survival curves
surv_comparison <- ggsurvplot(
  list(treat0 = surv_treat0, treat1 = surv_treat1),
  data = analysis_data,
  combine = TRUE,
  palette = c("#0073C2FF", "#EFC000FF"),
  legend.labs = c("Counterfactual: Low Viral Load", "Counterfactual: High Viral Load"),
  xlab = "Time (days)",
  ylab = "Survival Probability",
  title = "G-Formula: Counterfactual Survival Curves by Viral Load",
  conf.int = FALSE,
  ggtheme = theme_bw()
)

ggsave("figures/dags/gformula_survival_curves.png",
       surv_comparison$plot, width = 10, height = 6, dpi = 300)

#==============================================================================#
#                         SENSITIVITY ANALYSIS                                 #
#==============================================================================#

cat("\n========================================\n")
cat("SENSITIVITY ANALYSIS\n")
cat("========================================\n\n")

cat("Sensitivity to Unmeasured Confounding\n")
cat("--------------------------------------\n\n")

cat("Key concern: Unmeasured confounders like:\n")
cat("  - Comorbidities (diabetes, hypertension, obesity)\n")
cat("  - Disease severity at admission\n")
cat("  - Time from symptom onset\n")
cat("  - Vaccination status (in later waves)\n\n")

cat("Sensitivity Analysis Framework:\n")
cat("  E-value: minimum strength of association required for an\n")
cat("  unmeasured confounder to explain away the observed effect\n\n")

# Calculate E-value (requires evalue package)
if (require("EValue")) {
  hr_ipw <- exp(coef(cox_ipw))
  ci_lower <- exp(confint(cox_ipw)[1])
  ci_upper <- exp(confint(cox_ipw)[2])

  cat("Observed Hazard Ratio (IPW):", round(hr_ipw, 2), "\n")
  cat("95% CI: (", round(ci_lower, 2), ",", round(ci_upper, 2), ")\n\n")

  evalues <- evalue(HR(hr_ipw, rare = FALSE))
  cat("E-value for point estimate:", round(evalues[1], 2), "\n")
  cat("E-value for CI lower bound:", round(evalues[2], 2), "\n\n")

  cat("Interpretation:\n")
  cat("  An unmeasured confounder would need to be associated with both\n")
  cat("  viral load and mortality with a risk ratio of", round(evalues[1], 2), "\n")
  cat("  (each) to fully explain away the observed effect.\n\n")
} else {
  cat("Install EValue package for E-value calculation:\n")
  cat("  install.packages('EValue')\n\n")
}

#==============================================================================#
#                            RECOMMENDATIONS                                   #
#==============================================================================#

cat("\n========================================\n")
cat("RECOMMENDATIONS FOR CAUSAL ANALYSIS\n")
cat("========================================\n\n")

cat("1. PRIORITIZE ESTIMANDS\n")
cat("   Start with the most policy-relevant questions:\n")
cat("   - Viral load effect (potentially modifiable through early treatment)\n")
cat("   - Wave effects (inform pandemic preparedness)\n\n")

cat("2. DATA REQUIREMENTS\n")
cat("   Current dataset limitations:\n")
cat("   - No comorbidity data (major confounder)\n")
cat("   - No time from symptom onset\n")
cat("   - No treatment data (antivirals, steroids, oxygen support)\n")
cat("   - No vaccination status\n")
cat("   Recommendation: Supplement with medical records if possible\n\n")

cat("3. METHODS\n")
cat("   Recommended approach (in order):\n")
cat("   a) Create DAGs for each research question\n")
cat("   b) Identify minimal adjustment sets\n")
cat("   c) Check positivity (overlap in propensity scores)\n")
cat("   d) Use doubly robust methods (IPW + outcome modeling)\n")
cat("   e) Conduct sensitivity analyses\n")
cat("   f) Report both conditional (regression) and marginal (g-formula) effects\n\n")

cat("4. REPORTING\n")
cat("   Include in final report:\n")
cat("   - Clearly stated causal questions and estimands\n")
cat("   - DAGs showing assumed causal structure\n")
cat("   - Identification assumptions and their plausibility\n")
cat("   - Sensitivity analyses for violations\n")
cat("   - Comparison of causal vs associational estimates\n")
cat("   - Limitations and residual confounding\n\n")

cat("5. NEXT STEPS\n")
cat("   To implement this framework:\n")
cat("   a) Review and refine DAGs with domain experts\n")
cat("   b) Conduct exploratory data analysis to check assumptions\n")
cat("   c) Implement IPW and g-formula for primary estimands\n")
cat("   d) Perform sensitivity analyses\n")
cat("   e) Compare results to existing Cox models\n")
cat("   f) Write up findings with causal interpretation\n\n")

#==============================================================================#
#                           SAVE SESSION INFO                                  #
#==============================================================================#

# Save workspace for later use
save.image("data/causal_inference_workspace.RData")

# Save this script's output
sink("output/causal_estimands_summary.txt")
cat("CAUSAL INFERENCE FRAMEWORK - SUMMARY\n")
cat("====================================\n\n")
cat("Generated:", Sys.time(), "\n\n")
cat("This analysis proposed 5 causal estimands:\n")
cat("1. Viral load (Ct) effect on mortality\n")
cat("2. Pandemic wave effect on mortality\n")
cat("3. Age effect with mediation analysis\n")
cat("4. Hospital quality effect\n")
cat("5. Effect modification by gender\n\n")
cat("DAGs and adjustment sets have been saved to figures/dags/\n")
cat("Example IPW analysis demonstrates feasibility with current data.\n")
cat("See script for detailed methods and recommendations.\n")
sink()

cat("\n========================================\n")
cat("ANALYSIS COMPLETE\n")
cat("========================================\n\n")
cat("Output files created:\n")
cat("  - figures/dags/dag1_viral_load_mortality.png\n")
cat("  - figures/dags/dag2_wave_mortality.png\n")
cat("  - figures/dags/dag3_age_mediation.png\n")
cat("  - figures/dags/dag4_hospital_iv.png\n")
cat("  - figures/dags/propensity_scores.png\n")
cat("  - figures/dags/gformula_survival_curves.png\n")
cat("  - output/causal_estimands_summary.txt\n")
cat("  - data/causal_inference_workspace.RData\n\n")

message("Causal inference framework complete!")
