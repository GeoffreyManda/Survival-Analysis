# 08_causal_analysis_implementation.R
# COVID-19 Survival Analysis - Practical Causal Inference Implementation
# This script provides step-by-step implementation of causal analyses
# Based on the framework in 07-causal-inference-estimands.r

# Load required packages
if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  # Core packages
  dplyr, tidyverse, here,

  # Survival analysis
  survival, survminer,

  # Causal inference
  boot,           # Bootstrap confidence intervals
  MatchIt,        # Propensity score matching
  cobalt,         # Balance assessment

  # Visualization
  ggplot2, patchwork, gridExtra,
  RColorBrewer
)

#==============================================================================#
#                           LOAD AND PREPARE DATA                              #
#==============================================================================#

message("Loading data...")
load(here("data/processed_covid_data.RData"))

# Create analysis dataset with complete cases
analysis_data <- Covid_data %>%
  filter(
    !is.na(Ct),
    !is.na(mort_hospital),
    !is.na(LoS),
    !is.na(age),
    !is.na(gender),
    !is.na(wave),
    !is.na(hosp_id)
  ) %>%
  mutate(
    # Ensure factors are properly ordered
    wave = factor(wave, levels = c("1", "2", "3")),
    hosp_id = factor(hosp_id),

    # Create binary treatment variables
    high_viral_load = case_when(
      ct_gp == "Strongly positive" ~ 1,
      ct_gp == "Weakly Positive" ~ 0,
      TRUE ~ NA_real_
    ),

    # Later wave vs early wave
    late_wave = ifelse(wave == "3", 1, 0),

    # Elderly vs younger
    elderly = ifelse(age_gp == "Elderly", 1, 0)
  )

# Remove rows with NA in treatment variables
analysis_data <- analysis_data %>%
  filter(!is.na(high_viral_load))

message("Analysis dataset prepared:")
message("  Total observations: ", nrow(analysis_data))
message("  Events (deaths): ", sum(analysis_data$mort_hospital))
message("  Event rate: ", round(mean(analysis_data$mort_hospital) * 100, 1), "%")

# Create directories
dir.create("output/causal", showWarnings = FALSE, recursive = TRUE)
dir.create("figures/causal", showWarnings = FALSE, recursive = TRUE)
dir.create("models/causal", showWarnings = FALSE, recursive = TRUE)

#==============================================================================#
#        CAUSAL ANALYSIS 1: VIRAL LOAD EFFECT ON MORTALITY                    #
#==============================================================================#

message("\n=======================================================")
message("CAUSAL ANALYSIS 1: Viral Load Effect on Mortality")
message("=======================================================\n")

message("Research Question: What is the causal effect of high viral load")
message("(strongly positive Ct) vs low viral load (weakly positive Ct)")
message("on hospital mortality?")
message("")

#------------------------------------------------------------------------------#
# Step 1: Descriptive Statistics
#------------------------------------------------------------------------------#

message("Step 1: Descriptive statistics by treatment group")
message("--------------------------------------------------")

desc_table <- analysis_data %>%
  group_by(high_viral_load) %>%
  summarise(
    n = n(),
    deaths = sum(mort_hospital),
    mortality_rate = mean(mort_hospital),
    mean_age = mean(age),
    sd_age = sd(age),
    pct_male = mean(gender == 1),
    mean_los = mean(LoS, na.rm = TRUE),
    median_los = median(LoS, na.rm = TRUE)
  )

print(desc_table)
write.csv(desc_table, "output/causal/descriptive_by_viral_load.csv",
          row.names = FALSE)

# Chi-square test for mortality difference
chisq_result <- chisq.test(
  table(analysis_data$high_viral_load, analysis_data$mort_hospital)
)
message("\nChi-square test for mortality difference:")
print(chisq_result)

#------------------------------------------------------------------------------#
# Step 2: Propensity Score Estimation
#------------------------------------------------------------------------------#

message("\nStep 2: Estimating propensity scores")
message("--------------------------------------")

# Propensity score model: P(high viral load | confounders)
ps_formula <- high_viral_load ~ age + gender + wave + hosp_id

ps_model <- glm(
  ps_formula,
  data = analysis_data,
  family = binomial(link = "logit")
)

# Model summary
message("Propensity score model summary:")
print(summary(ps_model))

# Save model
saveRDS(ps_model, "models/causal/propensity_score_model.rds")

# Extract propensity scores
analysis_data$ps <- predict(ps_model, type = "response")

# Summary statistics
message("\nPropensity score distribution:")
message("Overall: ",
        sprintf("Mean=%.3f, SD=%.3f, Range=[%.3f, %.3f]",
                mean(analysis_data$ps), sd(analysis_data$ps),
                min(analysis_data$ps), max(analysis_data$ps)))

ps_summary <- analysis_data %>%
  group_by(high_viral_load) %>%
  summarise(
    n = n(),
    mean_ps = mean(ps),
    sd_ps = sd(ps),
    min_ps = min(ps),
    max_ps = max(ps)
  )
print(ps_summary)

#------------------------------------------------------------------------------#
# Step 3: Assess Overlap (Positivity Assumption)
#------------------------------------------------------------------------------#

message("\nStep 3: Checking positivity (propensity score overlap)")
message("-------------------------------------------------------")

# Propensity score distribution plot
ps_plot <- ggplot(analysis_data,
                  aes(x = ps, fill = factor(high_viral_load))) +
  geom_histogram(alpha = 0.6, position = "identity", bins = 30) +
  scale_fill_manual(
    values = c("#0073C2FF", "#EFC000FF"),
    labels = c("Low Viral Load", "High Viral Load")
  ) +
  labs(
    title = "Propensity Score Distribution by Treatment Group",
    subtitle = "Checking for overlap (positivity assumption)",
    x = "Propensity Score (Probability of High Viral Load)",
    y = "Count",
    fill = "Treatment"
  ) +
  theme_bw() +
  theme(legend.position = "bottom")

ggsave("figures/causal/propensity_scores_histogram.png", ps_plot,
       width = 10, height = 6, dpi = 300)

# Density plot (better for overlap visualization)
ps_density_plot <- ggplot(analysis_data,
                          aes(x = ps, fill = factor(high_viral_load))) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(
    values = c("#0073C2FF", "#EFC000FF"),
    labels = c("Low Viral Load", "High Viral Load")
  ) +
  labs(
    title = "Propensity Score Overlap",
    x = "Propensity Score",
    y = "Density",
    fill = "Treatment"
  ) +
  theme_bw() +
  theme(legend.position = "bottom")

ggsave("figures/causal/propensity_scores_density.png", ps_density_plot,
       width = 10, height = 6, dpi = 300)

# Check for extreme propensity scores (positivity violations)
extreme_ps <- analysis_data %>%
  filter(ps < 0.05 | ps > 0.95) %>%
  nrow()

message("Observations with extreme PS (<0.05 or >0.95): ", extreme_ps,
        " (", round(100 * extreme_ps / nrow(analysis_data), 1), "%)")

#------------------------------------------------------------------------------#
# Step 4: Covariate Balance Before Weighting
#------------------------------------------------------------------------------#

message("\nStep 4: Assessing covariate balance before weighting")
message("----------------------------------------------------")

# Standardized mean differences (SMD) before weighting
calc_smd <- function(data, var, treatment) {
  mean_treat <- mean(data[[var]][data[[treatment]] == 1], na.rm = TRUE)
  mean_control <- mean(data[[var]][data[[treatment]] == 0], na.rm = TRUE)
  sd_pooled <- sqrt(
    (var(data[[var]][data[[treatment]] == 1], na.rm = TRUE) +
     var(data[[var]][data[[treatment]] == 0], na.rm = TRUE)) / 2
  )
  smd <- (mean_treat - mean_control) / sd_pooled
  return(abs(smd))
}

smd_before <- data.frame(
  variable = c("age", "gender", "wave", "hosp_id"),
  smd = c(
    calc_smd(analysis_data, "age", "high_viral_load"),
    calc_smd(analysis_data, "gender", "high_viral_load"),
    NA,  # categorical
    NA   # categorical
  )
)

message("Standardized Mean Differences (continuous variables):")
print(smd_before[!is.na(smd_before$smd), ])
message("Note: SMD > 0.1 suggests meaningful imbalance")

#------------------------------------------------------------------------------#
# Step 5: Calculate IPW Weights
#------------------------------------------------------------------------------#

message("\nStep 5: Calculating inverse probability weights")
message("-----------------------------------------------")

analysis_data <- analysis_data %>%
  mutate(
    # Unstabilized ATE weights
    ipw_weight = case_when(
      high_viral_load == 1 ~ 1 / ps,
      high_viral_load == 0 ~ 1 / (1 - ps),
      TRUE ~ NA_real_
    ),

    # Stabilized ATE weights (more stable)
    ipw_weight_stab = case_when(
      high_viral_load == 1 ~ mean(high_viral_load) / ps,
      high_viral_load == 0 ~ (1 - mean(high_viral_load)) / (1 - ps),
      TRUE ~ NA_real_
    )
  )

# Weight diagnostics
message("IPW weight summary (stabilized):")
print(summary(analysis_data$ipw_weight_stab))

# Check for extreme weights
extreme_weights <- analysis_data %>%
  filter(ipw_weight_stab > 10) %>%
  nrow()

message("Observations with weight > 10: ", extreme_weights,
        " (", round(100 * extreme_weights / nrow(analysis_data), 1), "%)")

# Effective sample size
ess <- sum(analysis_data$ipw_weight_stab)^2 /
       sum(analysis_data$ipw_weight_stab^2)
message("Effective sample size: ", round(ess),
        " (", round(100 * ess / nrow(analysis_data), 1), "% of original)")

# Weight distribution plot
weight_plot <- ggplot(analysis_data,
                      aes(x = ipw_weight_stab,
                          fill = factor(high_viral_load))) +
  geom_histogram(alpha = 0.6, position = "identity", bins = 50) +
  scale_fill_manual(
    values = c("#0073C2FF", "#EFC000FF"),
    labels = c("Low Viral Load", "High Viral Load")
  ) +
  labs(
    title = "Distribution of IPW Weights",
    x = "Stabilized IPW Weight",
    y = "Count",
    fill = "Treatment"
  ) +
  xlim(0, quantile(analysis_data$ipw_weight_stab, 0.99)) +
  theme_bw() +
  theme(legend.position = "bottom")

ggsave("figures/causal/ipw_weights_distribution.png", weight_plot,
       width = 10, height = 6, dpi = 300)

#------------------------------------------------------------------------------#
# Step 6: Covariate Balance After Weighting
#------------------------------------------------------------------------------#

message("\nStep 6: Assessing covariate balance after weighting")
message("---------------------------------------------------")

# Weighted means
weighted_balance <- analysis_data %>%
  group_by(high_viral_load) %>%
  summarise(
    weighted_mean_age = weighted.mean(age, ipw_weight_stab),
    weighted_mean_gender = weighted.mean(gender, ipw_weight_stab),
    .groups = "drop"
  )

message("Weighted covariate means by treatment group:")
print(weighted_balance)

#------------------------------------------------------------------------------#
# Step 7: Estimate Causal Effect with IPW
#------------------------------------------------------------------------------#

message("\nStep 7: Estimating causal effect using IPW")
message("------------------------------------------")

# Unadjusted Cox model (for comparison)
cox_unadj <- coxph(
  Surv(LoS, mort_hospital) ~ high_viral_load,
  data = analysis_data
)

message("UNADJUSTED Cox model:")
print(summary(cox_unadj))

# Standard adjusted Cox model (for comparison)
cox_adj <- coxph(
  Surv(LoS, mort_hospital) ~ high_viral_load + age + gender + wave + hosp_id,
  data = analysis_data
)

message("\nSTANDARD ADJUSTED Cox model:")
print(summary(cox_adj))

# IPW-weighted Cox model (CAUSAL ESTIMATE)
cox_ipw <- coxph(
  Surv(LoS, mort_hospital) ~ high_viral_load,
  data = analysis_data,
  weights = ipw_weight_stab,
  robust = TRUE  # Robust variance for weighted data
)

message("\nIPW-WEIGHTED Cox model (Causal ATE):")
print(summary(cox_ipw))

# Save models
saveRDS(cox_unadj, "models/causal/cox_unadjusted.rds")
saveRDS(cox_adj, "models/causal/cox_adjusted.rds")
saveRDS(cox_ipw, "models/causal/cox_ipw.rds")

# Extract and compare estimates
results_comparison <- data.frame(
  Model = c("Unadjusted", "Covariate-Adjusted", "IPW (Causal ATE)"),
  HR = c(
    exp(coef(cox_unadj)),
    exp(coef(cox_adj)["high_viral_load"]),
    exp(coef(cox_ipw))
  ),
  Lower_CI = c(
    exp(confint(cox_unadj)[1]),
    exp(confint(cox_adj)["high_viral_load", 1]),
    exp(confint(cox_ipw)[1])
  ),
  Upper_CI = c(
    exp(confint(cox_unadj)[2]),
    exp(confint(cox_adj)["high_viral_load", 2]),
    exp(confint(cox_ipw)[2])
  )
)

message("\nCOMPARISON OF ESTIMATES:")
print(results_comparison)

write.csv(results_comparison,
          "output/causal/viral_load_effect_comparison.csv",
          row.names = FALSE)

# Forest plot comparison
forest_data <- results_comparison %>%
  mutate(
    Model = factor(Model, levels = rev(Model))
  )

forest_plot <- ggplot(forest_data, aes(x = HR, y = Model)) +
  geom_point(size = 4) +
  geom_errorbarh(aes(xmin = Lower_CI, xmax = Upper_CI), height = 0.2) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "red") +
  scale_x_continuous(trans = "log", breaks = c(0.5, 1, 2, 3, 4)) +
  labs(
    title = "Comparison of Viral Load Effect Estimates",
    subtitle = "Hazard Ratio: High vs Low Viral Load",
    x = "Hazard Ratio (95% CI)",
    y = ""
  ) +
  theme_bw() +
  theme(panel.grid.major.y = element_blank())

ggsave("figures/causal/effect_estimates_forest_plot.png", forest_plot,
       width = 10, height = 6, dpi = 300)

#------------------------------------------------------------------------------#
# Step 8: G-Formula (Standardization)
#------------------------------------------------------------------------------#

message("\nStep 8: G-formula for counterfactual survival curves")
message("----------------------------------------------------")

# Fit outcome model with all confounders
outcome_model <- coxph(
  Surv(LoS, mort_hospital) ~ high_viral_load + age + gender + wave + hosp_id,
  data = analysis_data
)

# Create counterfactual datasets
data_treat1 <- analysis_data %>% mutate(high_viral_load = 1)
data_treat0 <- analysis_data %>% mutate(high_viral_load = 0)

# Predict survival for each counterfactual scenario
surv_treat1 <- survfit(outcome_model, newdata = data_treat1)
surv_treat0 <- survfit(outcome_model, newdata = data_treat0)

# Extract survival curves
times <- surv_treat1$time
surv_df <- data.frame(
  time = c(times, times),
  survival = c(surv_treat1$surv, surv_treat0$surv),
  treatment = rep(c("High Viral Load", "Low Viral Load"),
                  c(length(times), length(times))),
  lower = c(surv_treat1$lower, surv_treat0$lower),
  upper = c(surv_treat1$upper, surv_treat0$upper)
)

# Plot counterfactual survival curves
gformula_plot <- ggplot(surv_df, aes(x = time, y = survival,
                                     color = treatment, fill = treatment)) +
  geom_line(size = 1) +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2, color = NA) +
  scale_color_manual(values = c("#EFC000FF", "#0073C2FF")) +
  scale_fill_manual(values = c("#EFC000FF", "#0073C2FF")) +
  labs(
    title = "G-Formula: Counterfactual Survival Curves",
    subtitle = "What would survival be if everyone had high vs low viral load?",
    x = "Time (days)",
    y = "Survival Probability",
    color = "Counterfactual Scenario",
    fill = "Counterfactual Scenario"
  ) +
  theme_bw() +
  theme(legend.position = "bottom")

ggsave("figures/causal/gformula_survival_curves.png", gformula_plot,
       width = 10, height = 6, dpi = 300)

# Calculate survival at specific timepoints
timepoints <- c(7, 14, 28)
survival_estimates <- lapply(timepoints, function(t) {
  idx1 <- which.min(abs(surv_treat1$time - t))
  idx0 <- which.min(abs(surv_treat0$time - t))

  data.frame(
    day = t,
    surv_high_vl = surv_treat1$surv[idx1],
    surv_low_vl = surv_treat0$surv[idx0],
    risk_diff = surv_treat0$surv[idx0] - surv_treat1$surv[idx1]
  )
})

survival_table <- do.call(rbind, survival_estimates)
message("\nG-formula survival estimates at key timepoints:")
print(survival_table)

write.csv(survival_table,
          "output/causal/gformula_survival_estimates.csv",
          row.names = FALSE)

#==============================================================================#
#         CAUSAL ANALYSIS 2: PANDEMIC WAVE EFFECT                             #
#==============================================================================#

message("\n=======================================================")
message("CAUSAL ANALYSIS 2: Pandemic Wave Effect on Mortality")
message("=======================================================\n")

message("Research Question: What would mortality be if all patients")
message("were admitted during Wave 3 vs Wave 1?")
message("")

# Create binary treatment: Wave 3 vs Wave 1
wave_data <- analysis_data %>%
  filter(wave %in% c("1", "3")) %>%
  mutate(wave3 = ifelse(wave == "3", 1, 0))

message("Sample size for wave analysis: ", nrow(wave_data))

# Propensity score for being in Wave 3
ps_wave_model <- glm(
  wave3 ~ age + gender + ct_gp + hosp_id,
  data = wave_data,
  family = binomial
)

wave_data$ps_wave <- predict(ps_wave_model, type = "response")
wave_data$ipw_wave <- ifelse(
  wave_data$wave3 == 1,
  mean(wave_data$wave3) / wave_data$ps_wave,
  (1 - mean(wave_data$wave3)) / (1 - wave_data$ps_wave)
)

# IPW Cox model for wave effect
cox_wave_ipw <- coxph(
  Surv(LoS, mort_hospital) ~ wave3,
  data = wave_data,
  weights = ipw_wave,
  robust = TRUE
)

message("IPW estimate of Wave 3 vs Wave 1 effect:")
print(summary(cox_wave_ipw))

# Save results
saveRDS(cox_wave_ipw, "models/causal/cox_wave_ipw.rds")

wave_effect <- data.frame(
  Comparison = "Wave 3 vs Wave 1",
  HR = exp(coef(cox_wave_ipw)),
  Lower_CI = exp(confint(cox_wave_ipw)[1]),
  Upper_CI = exp(confint(cox_wave_ipw)[2])
)

write.csv(wave_effect, "output/causal/wave_effect_estimate.csv",
          row.names = FALSE)

#==============================================================================#
#            SENSITIVITY ANALYSIS: E-VALUES                                    #
#==============================================================================#

message("\n=======================================================")
message("SENSITIVITY ANALYSIS: E-values")
message("=======================================================\n")

message("E-value quantifies robustness to unmeasured confounding.")
message("It is the minimum strength of association (risk ratio scale)")
message("that an unmeasured confounder would need to have with both")
message("treatment and outcome to fully explain away the observed effect.")
message("")

# Calculate E-value for viral load effect
if (require("EValue")) {
  hr_ipw <- exp(coef(cox_ipw))
  ci_lower <- exp(confint(cox_ipw)[1])

  message("Observed HR (IPW): ", round(hr_ipw, 2))
  message("95% CI lower bound: ", round(ci_lower, 2))
  message("")

  evalues <- evalue(HR(hr_ipw, rare = FALSE))

  message("E-value for point estimate: ", round(evalues[1], 2))
  message("E-value for CI lower bound: ", round(evalues[2], 2))
  message("")
  message("Interpretation:")
  message("  An unmeasured confounder would need to be associated with")
  message("  both high viral load AND mortality with a risk ratio of")
  message("  ", round(evalues[1], 2), " each to fully explain away the observed effect.")
  message("")
  message("  For the confidence interval to include the null, the")
  message("  confounder associations would need RR ≥ ", round(evalues[2], 2))

  # Save E-values
  evalue_results <- data.frame(
    Estimate = "High vs Low Viral Load",
    HR = hr_ipw,
    CI_Lower = ci_lower,
    Evalue_Point = evalues[1],
    Evalue_CI = evalues[2]
  )

  write.csv(evalue_results, "output/causal/evalue_sensitivity.csv",
            row.names = FALSE)
} else {
  message("Install EValue package for E-value calculation:")
  message("  install.packages('EValue')")
}

#==============================================================================#
#                        SUMMARY REPORT                                        #
#==============================================================================#

message("\n=======================================================")
message("SUMMARY REPORT")
message("=======================================================\n")

sink("output/causal/causal_analysis_summary.txt")
cat("COVID-19 SURVIVAL ANALYSIS - CAUSAL INFERENCE REPORT\n")
cat("=====================================================\n\n")
cat("Generated:", as.character(Sys.time()), "\n\n")

cat("RESEARCH QUESTION 1: Viral Load Effect on Mortality\n")
cat("----------------------------------------------------\n")
cat("Causal question: What is the effect of high viral load on mortality?\n\n")

cat("Sample:\n")
cat("  N = ", nrow(analysis_data), "\n")
cat("  High viral load: ", sum(analysis_data$high_viral_load == 1), "\n")
cat("  Low viral load: ", sum(analysis_data$high_viral_load == 0), "\n\n")

cat("Effect Estimates (Hazard Ratio):\n")
cat("  Unadjusted:          ", sprintf("%.2f (%.2f - %.2f)\n",
                                       results_comparison$HR[1],
                                       results_comparison$Lower_CI[1],
                                       results_comparison$Upper_CI[1]))
cat("  Covariate-Adjusted:  ", sprintf("%.2f (%.2f - %.2f)\n",
                                       results_comparison$HR[2],
                                       results_comparison$Lower_CI[2],
                                       results_comparison$Upper_CI[2]))
cat("  IPW (Causal ATE):    ", sprintf("%.2f (%.2f - %.2f)\n",
                                       results_comparison$HR[3],
                                       results_comparison$Lower_CI[3],
                                       results_comparison$Upper_CI[3]))

cat("\nInterpretation:\n")
cat("  The causal average treatment effect (ATE) estimate from IPW\n")
cat("  suggests that high viral load increases the hazard of death by\n")
cat("  a factor of ", sprintf("%.2f", results_comparison$HR[3]), ".\n\n")

if (exists("evalue_results")) {
  cat("Sensitivity to Unmeasured Confounding:\n")
  cat("  E-value (point): ", sprintf("%.2f\n", evalue_results$Evalue_Point))
  cat("  E-value (CI):    ", sprintf("%.2f\n", evalue_results$Evalue_CI))
  cat("  An unmeasured confounder would need RR ≥ ",
      sprintf("%.2f", evalue_results$Evalue_Point),
      " with both\n  treatment and outcome to explain away the effect.\n\n")
}

cat("\nREFERENCES:\n")
cat("-----------\n")
cat("Methods based on:\n")
cat("- Hernán MA, Robins JM (2020). Causal Inference: What If.\n")
cat("- Cole SR, Hernán MA (2008). Am J Epidemiol, 168(6):656-664.\n")
cat("- VanderWeele TJ, Ding P (2017). Ann Intern Med, 167(4):268-274.\n")

sink()

message("\n=======================================================")
message("ANALYSIS COMPLETE!")
message("=======================================================\n")
message("Output files created in output/causal/ and figures/causal/")
message("Summary report: output/causal/causal_analysis_summary.txt")
message("\nNext steps:")
message("  1. Review DAGs and assumptions")
message("  2. Assess covariate balance")
message("  3. Interpret causal estimates")
message("  4. Conduct additional sensitivity analyses")
message("  5. Compare with associational analyses")
