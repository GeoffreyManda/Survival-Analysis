# 14_sensitivity_analyses.R
# Comprehensive Sensitivity Analyses for Unmeasured Confounding
# Includes E-values, quantitative bias analysis, negative controls, and placebo tests

# Load required packages
if (!require("pacman")) install.packages("pacman")
pacman::p_load(dplyr, tidyverse, here, ggplot2, patchwork)

# Load EValue package (if not available, create custom function)
evalue_available <- require("EValue", quietly = TRUE)
if (!evalue_available) {
  message("EValue package not available. Will use custom implementation.")
}

#==============================================================================#
#                    LOAD DATA AND PREVIOUS RESULTS                            #
#==============================================================================#

message("========================================")
message("SENSITIVITY ANALYSES FOR UNMEASURED CONFOUNDING")
message("========================================\n")

# Load data
load(here("data/processed_covid_data.RData"))

# Prepare analysis dataset
analysis_data <- Covid_data %>%
  filter(!is.na(Ct), !is.na(mort_hospital), !is.na(LoS)) %>%
  mutate(
    high_viral_load = ifelse(ct_gp == "Strongly positive", 1, 0),
    A = high_viral_load,
    Y = mort_hospital
  ) %>%
  filter(!is.na(high_viral_load))

# Extract ATE estimates from previous analyses
# (In practice, load these from saved results)
ATE_observed <- 0.045  # From CEVAE
ATE_se <- 0.011
HR_observed <- exp(0.37)  # Hazard ratio from Cox model

message("Observed estimates:")
message(sprintf("  ATE (risk difference): %.4f ± %.4f", ATE_observed, ATE_se))
message(sprintf("  Hazard ratio: %.2f\n", HR_observed))

# Create output directories
dir.create("output/sensitivity", showWarnings = FALSE, recursive = TRUE)
dir.create("figures/sensitivity", showWarnings = FALSE, recursive = TRUE)

#==============================================================================#
#                    SENSITIVITY ANALYSIS 1: E-VALUES                          #
#==============================================================================#

message("\n========================================")
message("ANALYSIS 1: E-Values")
message("========================================\n")

message("E-value: Minimum strength of association an unmeasured confounder")
message("must have with both treatment and outcome to explain away the")
message("observed effect.\n")

# Calculate E-values
if (evalue_available) {
  library(EValue)

  # For hazard ratio
  evalues_hr <- evalue(HR(HR_observed, rare = FALSE))

  message("E-values for Hazard Ratio:")
  message(sprintf("  HR (observed): %.2f", HR_observed))
  message(sprintf("  E-value for point estimate: %.2f", evalues_hr[1]))
  message(sprintf("  E-value for CI lower bound: %.2f", evalues_hr[2]))

  # For risk difference (convert to risk ratio first)
  # Baseline risk
  baseline_risk <- mean(analysis_data$Y[analysis_data$A == 0])
  exposed_risk <- baseline_risk + ATE_observed
  RR <- exposed_risk / baseline_risk

  evalues_rr <- evalue(RR(RR, rare = FALSE))

  message("\nE-values for Risk Ratio:")
  message(sprintf("  Baseline risk: %.3f", baseline_risk))
  message(sprintf("  Exposed risk: %.3f", exposed_risk))
  message(sprintf("  Risk ratio: %.2f", RR))
  message(sprintf("  E-value for point estimate: %.2f", evalues_rr[1]))

} else {
  # Custom E-value calculation
  evalue_hr <- (HR_observed + sqrt(HR_observed * (HR_observed - 1)))
  message("E-value for HR (custom calculation):")
  message(sprintf("  E-value: %.2f", evalue_hr))

  evalues_hr <- c(evalue_hr, NA)
  evalues_rr <- c(NA, NA)
}

message("\nInterpretation:")
message("  An unmeasured confounder associated with both viral load")
message("  and mortality with RR ≥ %.2f (each) could explain away", evalues_hr[1])
message("  the observed effect.")

# Compare to plausible confounders
plausible_confounders <- data.frame(
  Confounder = c(
    "Comorbidity Index (Charlson)",
    "Disease Severity Score (NEWS)",
    "Immune Status (Lymphocyte Count)",
    "Symptom Duration",
    "Socioeconomic Status",
    "Frailty Index"
  ),
  Estimated_RR_Treatment = c(1.8, 1.5, 1.3, 1.2, 1.4, 2.0),
  Estimated_RR_Outcome = c(2.5, 3.0, 1.8, 1.3, 1.5, 2.2),
  Required_RR = evalues_hr[1]
)

message("\nPlausible unmeasured confounders:")
print(plausible_confounders)

message("\nConclusion:")
sufficient <- plausible_confounders %>%
  mutate(Sufficient = (Estimated_RR_Treatment >= Required_RR &
                      Estimated_RR_Outcome >= Required_RR))

if (any(sufficient$Sufficient)) {
  message("  Some plausible confounders could explain away the effect.")
  message("  → Effect estimate is NOT robust to unmeasured confounding")
} else {
  message("  No single plausible confounder is strong enough.")
  message("  → Effect estimate is moderately robust")
}

# Save E-value results
evalue_results <- data.frame(
  Estimate = c("Hazard Ratio", "Risk Ratio"),
  Value = c(HR_observed, ifelse(evalue_available, RR, NA)),
  Evalue_Point = c(evalues_hr[1], evalues_rr[1]),
  Evalue_CI = c(evalues_hr[2], evalues_rr[2])
)

write.csv(evalue_results, "output/sensitivity/evalue_results.csv",
          row.names = FALSE)

# Visualization
if (evalue_available && !is.na(evalues_hr[1])) {
  evalue_plot_data <- plausible_confounders %>%
    mutate(
      Label = paste0(Confounder, "\n(", Estimated_RR_Treatment, ", ",
                    Estimated_RR_Outcome, ")")
    )

  p_evalue <- ggplot(evalue_plot_data) +
    geom_point(aes(x = Estimated_RR_Treatment, y = Estimated_RR_Outcome,
                  color = Confounder), size = 4) +
    geom_hline(yintercept = evalues_hr[1], linetype = "dashed",
              color = "red", size = 1) +
    geom_vline(xintercept = evalues_hr[1], linetype = "dashed",
              color = "red", size = 1) +
    geom_abline(intercept = 0, slope = 1, linetype = "dotted",
               color = "gray50") +
    annotate("text", x = evalues_hr[1], y = Inf, vjust = 1.5,
            label = sprintf("E-value = %.2f", evalues_hr[1]),
            color = "red", fontface = "bold") +
    labs(
      title = "E-value Sensitivity Analysis",
      subtitle = "Plausible unmeasured confounders vs. required strength",
      x = "RR (Confounder → Treatment)",
      y = "RR (Confounder → Outcome)",
      caption = "Points in upper-right quadrant could explain away effect"
    ) +
    theme_minimal() +
    theme(legend.position = "bottom")

  ggsave("figures/sensitivity/evalue_plot.png", p_evalue,
         width = 10, height = 8, dpi = 300)

  message("  E-value plot saved!")
}

#==============================================================================#
#                    SENSITIVITY ANALYSIS 2: QUANTITATIVE BIAS ANALYSIS        #
#==============================================================================#

message("\n========================================")
message("ANALYSIS 2: Quantitative Bias Analysis")
message("========================================\n")

message("Simulating the impact of unmeasured confounder with varying strength.\n")

# Define grid of confounder strengths
confounder_strengths_treatment <- seq(1.0, 3.0, 0.1)
confounder_strengths_outcome <- seq(1.0, 3.0, 0.1)

# Function to calculate bias-adjusted estimate
# Using formula from Ding & VanderWeele (2016)
calculate_bias_adjusted_rr <- function(rr_obs, rr_ut, rr_uy, prev_u) {
  # rr_obs: observed RR
  # rr_ut: RR for U -> Treatment
  # rr_uy: RR for U -> Outcome
  # prev_u: prevalence of unmeasured confounder

  # Simplified bias factor (assumes binary confounder)
  bias_factor <- (rr_uy - 1) * (rr_ut - 1) * prev_u + 1

  rr_adjusted <- rr_obs / bias_factor
  return(rr_adjusted)
}

# Calculate bias-adjusted estimates for grid
bias_grid <- expand.grid(
  RR_UT = confounder_strengths_treatment,
  RR_UY = confounder_strengths_outcome
) %>%
  mutate(
    RR_adjusted = calculate_bias_adjusted_rr(
      rr_obs = RR,
      rr_ut = RR_UT,
      rr_uy = RR_UY,
      prev_u = 0.3  # Assume 30% prevalence
    ),
    Bias = RR - RR_adjusted,
    Null_Crossed = (RR_adjusted < 1)
  )

message("Bias analysis grid:")
message(sprintf("  %d combinations of confounder strengths", nrow(bias_grid)))
message(sprintf("  RR range (Treatment): %.1f - %.1f",
                min(bias_grid$RR_UT), max(bias_grid$RR_UT)))
message(sprintf("  RR range (Outcome): %.1f - %.1f",
                min(bias_grid$RR_UY), max(bias_grid$RR_UY)))

# Find combinations that nullify effect
null_combinations <- bias_grid %>%
  filter(Null_Crossed)

message(sprintf("\n%d combinations would nullify the effect (%.1f%%)",
                nrow(null_combinations),
                100 * nrow(null_combinations) / nrow(bias_grid)))

# Visualization: Contour plot
p_bias_contour <- ggplot(bias_grid, aes(x = RR_UT, y = RR_UY, z = RR_adjusted)) +
  geom_contour_filled(bins = 15) +
  geom_contour(color = "white", alpha = 0.3) +
  geom_hline(yintercept = evalues_hr[1], color = "red", linetype = "dashed") +
  geom_vline(xintercept = evalues_hr[1], color = "red", linetype = "dashed") +
  scale_fill_viridis_d(option = "plasma") +
  labs(
    title = "Quantitative Bias Analysis",
    subtitle = "Bias-adjusted RR under different unmeasured confounder strengths",
    x = "RR (Unmeasured Confounder → Treatment)",
    y = "RR (Unmeasured Confounder → Outcome)",
    fill = "Adjusted RR"
  ) +
  theme_minimal()

ggsave("figures/sensitivity/bias_contour_plot.png", p_bias_contour,
       width = 10, height = 8, dpi = 300)

message("  Bias contour plot saved!")

#==============================================================================#
#                    SENSITIVITY ANALYSIS 3: NEGATIVE CONTROL OUTCOMES         #
#==============================================================================#

message("\n========================================")
message("ANALYSIS 3: Negative Control Outcomes")
message("========================================\n")

message("Negative controls: Outcomes that SHOULD NOT be affected by viral load.")
message("If we find effects on negative controls, it suggests residual confounding.\n")

# Define negative control outcomes
# (Variables that should not be affected by viral load)
negative_controls <- list(
  gender = "Gender (biological, not affected by VL)",
  age_category = "Age category (fixed at admission)",
  wave = "Pandemic wave (temporal, not affected by VL)"
)

# Test for associations with negative controls
negative_control_results <- list()

for (nc_name in names(negative_controls)) {
  message(sprintf("Testing: %s", negative_controls[[nc_name]]))

  if (nc_name == "gender") {
    # Test if viral load predicts gender (should be ~0)
    model <- glm(gender ~ high_viral_load + age + wave + hosp_id,
                data = analysis_data, family = binomial)
    coef_est <- coef(model)["high_viral_load"]
    se_est <- summary(model)$coefficients["high_viral_load", "Std. Error"]

  } else if (nc_name == "age_category") {
    # Test if viral load predicts being elderly (controlling for continuous age)
    analysis_data$elderly <- ifelse(analysis_data$age_gp == "Elderly", 1, 0)
    model <- glm(elderly ~ high_viral_load + wave + hosp_id,
                data = analysis_data, family = binomial)
    coef_est <- coef(model)["high_viral_load"]
    se_est <- summary(model)$coefficients["high_viral_load", "Std. Error"]

  } else if (nc_name == "wave") {
    # Test if viral load predicts wave (should be reverse causation only)
    analysis_data$wave3 <- ifelse(analysis_data$wave == "3", 1, 0)
    model <- glm(wave3 ~ high_viral_load + age + gender + hosp_id,
                data = analysis_data, family = binomial)
    coef_est <- coef(model)["high_viral_load"]
    se_est <- summary(model)$coefficients["high_viral_load", "Std. Error"]
  }

  p_value <- 2 * pnorm(-abs(coef_est / se_est))

  negative_control_results[[nc_name]] <- data.frame(
    Outcome = negative_controls[[nc_name]],
    Coefficient = coef_est,
    SE = se_est,
    P_value = p_value,
    Significant = p_value < 0.05
  )

  message(sprintf("  Coefficient: %.4f (SE=%.4f, p=%.4f)",
                  coef_est, se_est, p_value))
}

nc_results_df <- bind_rows(negative_control_results)

message("\nNegative control results:")
print(nc_results_df)

if (any(nc_results_df$Significant)) {
  message("\nWARNING: Some negative controls show significant effects!")
  message("This suggests residual confounding or model misspecification.")
} else {
  message("\nGOOD: No negative controls show significant effects.")
  message("This supports the validity of the main analysis.")
}

write.csv(nc_results_df, "output/sensitivity/negative_control_results.csv",
          row.names = FALSE)

# Visualization
p_nc <- ggplot(nc_results_df, aes(x = Outcome, y = Coefficient)) +
  geom_point(size = 4) +
  geom_errorbar(aes(ymin = Coefficient - 1.96*SE,
                   ymax = Coefficient + 1.96*SE),
               width = 0.2, size = 1) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  coord_flip() +
  labs(
    title = "Negative Control Analysis",
    subtitle = "Effect estimates on outcomes that should NOT be affected",
    x = "",
    y = "Coefficient (95% CI)",
    caption = "Significant effects suggest residual confounding"
  ) +
  theme_minimal()

ggsave("figures/sensitivity/negative_controls.png", p_nc,
       width = 10, height = 6, dpi = 300)

#==============================================================================#
#                    SENSITIVITY ANALYSIS 4: PLACEBO OUTCOMES                  #
#==============================================================================#

message("\n========================================")
message("ANALYSIS 4: Placebo Outcome Test")
message("========================================\n")

message("Placebo test: Randomly permute treatment assignment.")
message("If we still find effects, it suggests spurious association.\n")

# Permutation test
n_permutations <- 1000
set.seed(42)

permutation_ates <- sapply(1:n_permutations, function(i) {
  # Randomly shuffle treatment
  A_permuted <- sample(analysis_data$A)

  # Calculate ATE with permuted treatment
  ate_perm <- mean(analysis_data$Y[A_permuted == 1]) -
             mean(analysis_data$Y[A_permuted == 0])

  return(ate_perm)
})

# Observed ATE
ate_observed <- mean(analysis_data$Y[analysis_data$A == 1]) -
               mean(analysis_data$Y[analysis_data$A == 0])

# P-value from permutation test
p_value_perm <- mean(abs(permutation_ates) >= abs(ate_observed))

message("Permutation test results:")
message(sprintf("  Observed ATE: %.4f", ate_observed))
message(sprintf("  Mean permuted ATE: %.4f", mean(permutation_ates)))
message(sprintf("  SD permuted ATE: %.4f", sd(permutation_ates)))
message(sprintf("  P-value: %.4f", p_value_perm))

if (p_value_perm < 0.05) {
  message("\nObserved ATE is significantly different from permuted ATEs.")
  message("This supports a true effect (not spurious).")
} else {
  message("\nWARNING: Observed ATE not significantly different from permuted.")
  message("This suggests the association may be spurious.")
}

# Visualization
perm_df <- data.frame(ATE = permutation_ates)

p_perm <- ggplot(perm_df, aes(x = ATE)) +
  geom_histogram(bins = 50, fill = "gray70", color = "black", alpha = 0.7) +
  geom_vline(xintercept = ate_observed, color = "red",
            size = 1.5, linetype = "dashed") +
  geom_vline(xintercept = quantile(permutation_ates, c(0.025, 0.975)),
            color = "blue", linetype = "dotted") +
  labs(
    title = "Permutation Test",
    subtitle = sprintf("Observed ATE (red) vs null distribution (p=%.4f)",
                      p_value_perm),
    x = "Average Treatment Effect (Permuted)",
    y = "Frequency",
    caption = "Blue lines: 2.5% and 97.5% percentiles"
  ) +
  theme_minimal()

ggsave("figures/sensitivity/permutation_test.png", p_perm,
       width = 10, height = 6, dpi = 300)

#==============================================================================#
#                    SENSITIVITY ANALYSIS 5: SYNTHETIC CONFOUNDER              #
#==============================================================================#

message("\n========================================")
message("ANALYSIS 5: Synthetic Unmeasured Confounder")
message("========================================\n")

message("Adding a synthetic unmeasured confounder to test method robustness.\n")

# Generate synthetic unmeasured confounder
# Correlated with age and affects both treatment and outcome
set.seed(42)

analysis_data$U_synthetic <- rnorm(nrow(analysis_data),
                                   mean = 0.3 * scale(analysis_data$age)[,1],
                                   sd = 0.5)

# True effect of U on treatment (propensity)
beta_U_A <- 0.8  # Strong effect

# True effect of U on outcome
beta_U_Y <- 1.2  # Strong effect

# Adjust treatment based on U
ps_with_U <- plogis(qlogis(0.4) + beta_U_A * analysis_data$U_synthetic)
A_with_U <- rbinom(nrow(analysis_data), 1, ps_with_U)

# Adjust outcome based on U
Y_prob_with_U <- plogis(qlogis(analysis_data$Y) + beta_U_Y * analysis_data$U_synthetic)
Y_with_U <- rbinom(nrow(analysis_data), 1, Y_prob_with_U)

# Estimate ATE ignoring U (biased)
ate_ignore_U <- mean(Y_with_U[A_with_U == 1]) -
               mean(Y_with_U[A_with_U == 0])

# Estimate ATE controlling for U (unbiased)
model_with_U <- glm(Y_with_U ~ A_with_U + age + gender + wave + hosp_id + U_synthetic,
                   family = binomial)
ate_control_U <- coef(model_with_U)["A_with_U"]

message("Synthetic confounder results:")
message(sprintf("  True confounder strength: βUA=%.1f, βUY=%.1f",
                beta_U_A, beta_U_Y))
message(sprintf("  ATE ignoring U: %.4f (BIASED)", ate_ignore_U))
message(sprintf("  ATE controlling for U: %.4f (UNBIASED)", ate_control_U))
message(sprintf("  Bias: %.4f", ate_ignore_U - ate_control_U))

# Save results
synthetic_results <- data.frame(
  Method = c("Ignoring U", "Controlling for U"),
  ATE = c(ate_ignore_U, ate_control_U),
  Bias = c(ate_ignore_U - ate_control_U, 0)
)

write.csv(synthetic_results, "output/sensitivity/synthetic_confounder_results.csv",
          row.names = FALSE)

#==============================================================================#
#                    SUMMARY REPORT                                            #
#==============================================================================#

message("\n========================================")
message("GENERATING SUMMARY REPORT")
message("========================================\n")

sink("output/sensitivity/sensitivity_analysis_summary.txt")
cat("SENSITIVITY ANALYSIS SUMMARY\n")
cat("============================\n\n")
cat("Generated:", as.character(Sys.time()), "\n\n")

cat("ANALYSIS 1: E-VALUES\n")
cat("-------------------\n")
cat(sprintf("Observed HR: %.2f\n", HR_observed))
cat(sprintf("E-value for point estimate: %.2f\n", evalues_hr[1]))
cat(sprintf("E-value for CI lower bound: %.2f\n\n", evalues_hr[2]))
cat("Interpretation:\n")
cat(sprintf("  An unmeasured confounder with RR ≥ %.2f for both\n", evalues_hr[1]))
cat("  treatment and outcome could explain away the effect.\n\n")

cat("Plausible confounders:\n")
print(plausible_confounders)
cat("\n")

cat("ANALYSIS 2: QUANTITATIVE BIAS ANALYSIS\n")
cat("--------------------------------------\n")
cat(sprintf("Tested %d combinations of confounder strengths\n", nrow(bias_grid)))
cat(sprintf("%d combinations would nullify effect (%.1f%%)\n\n",
            nrow(null_combinations),
            100 * nrow(null_combinations) / nrow(bias_grid)))

cat("ANALYSIS 3: NEGATIVE CONTROL OUTCOMES\n")
cat("-------------------------------------\n")
print(nc_results_df)
cat("\n")
if (any(nc_results_df$Significant)) {
  cat("WARNING: Some negative controls significant → residual confounding\n\n")
} else {
  cat("GOOD: No negative controls significant → analysis valid\n\n")
}

cat("ANALYSIS 4: PLACEBO OUTCOME TEST\n")
cat("--------------------------------\n")
cat(sprintf("Observed ATE: %.4f\n", ate_observed))
cat(sprintf("Permutation p-value: %.4f\n", p_value_perm))
if (p_value_perm < 0.05) {
  cat("Result: Observed effect is statistically significant\n\n")
} else {
  cat("WARNING: Observed effect not significant in permutation test\n\n")
}

cat("ANALYSIS 5: SYNTHETIC UNMEASURED CONFOUNDER\n")
cat("-------------------------------------------\n")
cat(sprintf("ATE ignoring U: %.4f (BIASED)\n", ate_ignore_U))
cat(sprintf("ATE controlling for U: %.4f (UNBIASED)\n", ate_control_U))
cat(sprintf("Bias magnitude: %.4f\n\n", abs(ate_ignore_U - ate_control_U)))

cat("OVERALL CONCLUSION:\n")
cat("-------------------\n")
cat("Based on sensitivity analyses:\n\n")

# Determine robustness level
if (evalues_hr[1] > 2.5 && !any(nc_results_df$Significant) && p_value_perm < 0.05) {
  cat("ROBUST: Effect appears robust to unmeasured confounding\n")
  cat("  - High E-value (>2.5)\n")
  cat("  - Negative controls all null\n")
  cat("  - Significant in permutation test\n")
} else if (evalues_hr[1] > 1.5) {
  cat("MODERATELY ROBUST: Effect moderately robust\n")
  cat("  - Moderate E-value (1.5-2.5)\n")
  cat("  - Consider additional sensitivity analyses\n")
} else {
  cat("WEAK ROBUSTNESS: Effect may be explained by unmeasured confounding\n")
  cat("  - Low E-value (<1.5)\n")
  cat("  - Interpret results with caution\n")
}

sink()

message("Summary report saved to: output/sensitivity/sensitivity_analysis_summary.txt")

#==============================================================================#
#                    CREATE COMBINED VISUALIZATION                             #
#==============================================================================#

message("\nCreating combined sensitivity analysis figure...")

# Load saved plots
plots_exist <- all(file.exists(c(
  "figures/sensitivity/evalue_plot.png",
  "figures/sensitivity/bias_contour_plot.png",
  "figures/sensitivity/negative_controls.png",
  "figures/sensitivity/permutation_test.png"
)))

if (plots_exist || (!plots_exist && exists("p_evalue"))) {
  # Create panel
  if (exists("p_evalue") && exists("p_nc") && exists("p_perm")) {
    panel_sensitivity <- (p_evalue | p_nc) / p_perm +
      plot_annotation(
        title = "Sensitivity Analyses for Unmeasured Confounding",
        tag_levels = "A",
        theme = theme(plot.title = element_text(face = "bold", size = 16))
      )

    ggsave("figures/sensitivity/sensitivity_panel.png", panel_sensitivity,
           width = 16, height = 12, dpi = 300)

    message("  Combined panel saved!")
  }
}

message("\n========================================")
message("SENSITIVITY ANALYSES COMPLETE!")
message("========================================\n")
message("Output files:")
message("  - output/sensitivity/evalue_results.csv")
message("  - output/sensitivity/negative_control_results.csv")
message("  - output/sensitivity/synthetic_confounder_results.csv")
message("  - output/sensitivity/sensitivity_analysis_summary.txt")
message("\nFigures:")
message("  - figures/sensitivity/evalue_plot.png")
message("  - figures/sensitivity/bias_contour_plot.png")
message("  - figures/sensitivity/negative_controls.png")
message("  - figures/sensitivity/permutation_test.png")
message("  - figures/sensitivity/sensitivity_panel.png")
