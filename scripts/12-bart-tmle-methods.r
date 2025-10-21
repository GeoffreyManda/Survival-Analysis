# 12_bart_tmle_methods.R
# Advanced Causal Inference Methods: BART and TMLE
# Bayesian Additive Regression Trees and Targeted Maximum Likelihood Estimation

# Load required packages
if (!require("pacman")) install.packages("pacman")
pacman::p_load(dplyr, tidyverse, here, survival, ggplot2)

# Try to load BART and tmle packages
bart_available <- FALSE
tmle_available <- FALSE

tryCatch({
  library(BART)
  bart_available <- TRUE
  message("BART package loaded successfully")
}, error = function(e) {
  message("BART package not available. Attempting installation...")
  tryCatch({
    install.packages("BART")
    library(BART)
    bart_available <- TRUE
  }, error = function(e2) {
    message("Could not install BART. Will skip BART analysis.")
  })
})

tryCatch({
  library(tmle)
  tmle_available <- TRUE
  message("tmle package loaded successfully")
}, error = function(e) {
  message("tmle package not available. Attempting installation...")
  tryCatch({
    install.packages("tmle")
    library(tmle)
    tmle_available <- TRUE
  }, error = function(e2) {
    message("Could not install tmle. Will skip TMLE analysis.")
  })
})

# Also try SuperLearner for TMLE
if (tmle_available) {
  tryCatch({
    library(SuperLearner)
  }, error = function(e) {
    install.packages("SuperLearner")
    library(SuperLearner)
  })
}

#==============================================================================#
#                    LOAD DATA                                                 #
#==============================================================================#

message("\n========================================")
message("ADVANCED CAUSAL METHODS: BART & TMLE")
message("========================================\n")

# Load preprocessed data
load(here("data/processed_covid_data.RData"))

# Prepare analysis dataset
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
    # Treatment: high viral load
    A = ifelse(ct_gp == "Strongly positive", 1, 0),
    # Outcome: mortality
    Y = mort_hospital,
    # Covariates
    age_scaled = scale(age)[,1],
    wave_num = as.numeric(as.character(wave)),
    hosp_num = as.numeric(as.character(hosp_id))
  ) %>%
  filter(!is.na(A))

# Create covariate matrix
X <- analysis_data %>%
  select(age_scaled, gender, wave_num, hosp_num) %>%
  as.matrix()

A <- analysis_data$A
Y <- analysis_data$Y

message(sprintf("Total observations: %d", nrow(analysis_data)))
message(sprintf("Treated (high viral load): %d (%.1f%%)",
                sum(A), 100 * mean(A)))
message(sprintf("Deaths: %d (%.1f%%)", sum(Y), 100 * mean(Y)))

# Create output directories
dir.create("output/advanced_causal", showWarnings = FALSE, recursive = TRUE)
dir.create("figures/advanced_causal", showWarnings = FALSE, recursive = TRUE)
dir.create("models/advanced_causal", showWarnings = FALSE, recursive = TRUE)

# Train/test split
set.seed(42)
train_idx <- sample(1:nrow(X), 0.8 * nrow(X))
test_idx <- setdiff(1:nrow(X), train_idx)

X_train <- X[train_idx, ]
X_test <- X[test_idx, ]
A_train <- A[train_idx]
A_test <- A[test_idx]
Y_train <- Y[train_idx]
Y_test <- Y[test_idx]

#==============================================================================#
#                    METHOD 1: BAYESIAN ADDITIVE REGRESSION TREES (BART)       #
#==============================================================================#

if (bart_available) {
  message("\n========================================")
  message("METHOD 1: Bayesian Additive Regression Trees (BART)")
  message("========================================\n")

  message("BART is a Bayesian ensemble method that:")
  message("  - Uses sum of regression trees to model outcomes")
  message("  - Provides uncertainty quantification via posterior distributions")
  message("  - Automatically handles non-linearities and interactions")
  message("  - Can estimate heterogeneous treatment effects\n")

  # Fit BART for treatment model (propensity score)
  message("Step 1: Fitting propensity score model with BART...")

  bart_ps <- pbart(
    x.train = X_train,
    y.train = A_train,
    x.test = X_test,
    ntree = 200,
    nskip = 1000,
    ndpost = 2000,
    printevery = 1000
  )

  # Extract propensity scores
  ps_train <- colMeans(pnorm(bart_ps$yhat.train))
  ps_test <- colMeans(pnorm(bart_ps$yhat.test))

  message(sprintf("  Propensity score range: [%.3f, %.3f]",
                  min(ps_test), max(ps_test)))

  # Fit BART for outcome models (separate for treated and control)
  message("\nStep 2: Fitting outcome models for treated and control groups...")

  # Model for treated
  treated_idx <- which(A_train == 1)
  message(sprintf("  Training BART for TREATED (n=%d)...", length(treated_idx)))

  bart_treated <- pbart(
    x.train = X_train[treated_idx, ],
    y.train = Y_train[treated_idx],
    x.test = X_test,
    ntree = 200,
    nskip = 1000,
    ndpost = 2000,
    printevery = 1000
  )

  # Model for control
  control_idx <- which(A_train == 0)
  message(sprintf("  Training BART for CONTROL (n=%d)...", length(control_idx)))

  bart_control <- pbart(
    x.train = X_train[control_idx, ],
    y.train = Y_train[control_idx],
    x.test = X_test,
    ntree = 200,
    nskip = 1000,
    ndpost = 2000,
    printevery = 1000
  )

  # Extract posterior mean predictions
  Y1_pred_bart <- colMeans(pnorm(bart_treated$yhat.test))
  Y0_pred_bart <- colMeans(pnorm(bart_control$yhat.test))

  # Individual treatment effects with full posterior
  ITE_posterior <- pnorm(bart_treated$yhat.test) - pnorm(bart_control$yhat.test)
  ITE_bart <- colMeans(ITE_posterior)
  ITE_bart_sd <- apply(ITE_posterior, 2, sd)
  ITE_bart_lower <- apply(ITE_posterior, 2, quantile, 0.025)
  ITE_bart_upper <- apply(ITE_posterior, 2, quantile, 0.975)

  # Average treatment effect
  ATE_posterior <- rowMeans(ITE_posterior)
  ATE_bart <- mean(ATE_posterior)
  ATE_bart_sd <- sd(ATE_posterior)
  ATE_bart_ci <- quantile(ATE_posterior, c(0.025, 0.975))

  # ATT and ATC
  ATT_bart <- mean(ITE_bart[A_test == 1])
  ATC_bart <- mean(ITE_bart[A_test == 0])

  message("\n----------------------------------------")
  message("BART Results:")
  message("----------------------------------------")
  message(sprintf("ATE (Average Treatment Effect): %.4f", ATE_bart))
  message(sprintf("  Posterior SD: %.4f", ATE_bart_sd))
  message(sprintf("  95%% Credible Interval: [%.4f, %.4f]",
                  ATE_bart_ci[1], ATE_bart_ci[2]))
  message(sprintf("ATT (Effect on Treated): %.4f", ATT_bart))
  message(sprintf("ATC (Effect on Controls): %.4f", ATC_bart))
  message(sprintf("ITE mean: %.4f", mean(ITE_bart)))
  message(sprintf("ITE SD (heterogeneity): %.4f", sd(ITE_bart)))

  # Save BART results
  bart_results <- list(
    model_ps = bart_ps,
    model_treated = bart_treated,
    model_control = bart_control,
    ATE = ATE_bart,
    ATE_posterior = ATE_posterior,
    ATE_sd = ATE_bart_sd,
    ATE_ci = ATE_bart_ci,
    ATT = ATT_bart,
    ATC = ATC_bart,
    ITE = ITE_bart,
    ITE_sd = ITE_bart_sd,
    ITE_lower = ITE_bart_lower,
    ITE_upper = ITE_bart_upper,
    ps_test = ps_test
  )

  saveRDS(bart_results, "models/advanced_causal/bart_model.rds")

  # Visualizations
  message("\nCreating BART visualizations...")

  # 1. ITE posterior distribution
  ite_post_sample <- sample(1:nrow(ITE_posterior), 100)
  ite_post_df <- data.frame(
    ITE = as.vector(ITE_posterior[ite_post_sample, ]),
    Patient = rep(1:ncol(ITE_posterior), each = length(ite_post_sample))
  )

  p_bart_1 <- ggplot(data.frame(ITE = ITE_bart), aes(x = ITE)) +
    geom_histogram(aes(y = ..density..), bins = 50, fill = "#56B4E9", alpha = 0.7) +
    geom_density(size = 1.2, color = "black") +
    geom_vline(xintercept = ATE_bart, color = "red", linetype = "dashed", size = 1) +
    geom_vline(xintercept = ATE_bart_ci[1], color = "red", linetype = "dotted") +
    geom_vline(xintercept = ATE_bart_ci[2], color = "red", linetype = "dotted") +
    labs(
      title = "BART: Individual Treatment Effect Distribution",
      subtitle = sprintf("ATE = %.4f (95%% CI: [%.4f, %.4f])",
                         ATE_bart, ATE_bart_ci[1], ATE_bart_ci[2]),
      x = "Individual Treatment Effect (ITE)",
      y = "Density"
    ) +
    theme_minimal()

  ggsave("figures/advanced_causal/bart_ite_distribution.png", p_bart_1,
         width = 10, height = 6, dpi = 300)

  # 2. ATE posterior distribution
  p_bart_2 <- ggplot(data.frame(ATE = ATE_posterior), aes(x = ATE)) +
    geom_histogram(aes(y = ..density..), bins = 50, fill = "#E69F00", alpha = 0.7) +
    geom_density(size = 1.2, color = "black") +
    geom_vline(xintercept = ATE_bart, color = "red", size = 1.2) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
    labs(
      title = "BART: ATE Posterior Distribution",
      subtitle = sprintf("Posterior mean = %.4f, SD = %.4f",
                         ATE_bart, ATE_bart_sd),
      x = "Average Treatment Effect (ATE)",
      y = "Posterior Density"
    ) +
    theme_minimal()

  ggsave("figures/advanced_causal/bart_ate_posterior.png", p_bart_2,
         width = 10, height = 6, dpi = 300)

  # 3. Propensity score distribution
  ps_df <- data.frame(
    PS = ps_test,
    Treatment = factor(A_test, levels = c(0, 1),
                      labels = c("Low Viral Load", "High Viral Load"))
  )

  p_bart_3 <- ggplot(ps_df, aes(x = PS, fill = Treatment)) +
    geom_density(alpha = 0.6) +
    scale_fill_manual(values = c("#00BFC4", "#F8766D")) +
    labs(
      title = "BART: Propensity Score Distribution",
      x = "Propensity Score (P(Treatment | Covariates))",
      y = "Density",
      fill = "Actual Treatment"
    ) +
    theme_minimal()

  ggsave("figures/advanced_causal/bart_propensity_scores.png", p_bart_3,
         width = 10, height = 6, dpi = 300)

  message("  BART visualizations saved!")

} else {
  message("\nBART analysis skipped (package not available)")
  bart_results <- NULL
}

#==============================================================================#
#                    METHOD 2: TARGETED MAXIMUM LIKELIHOOD ESTIMATION (TMLE)   #
#==============================================================================#

if (tmle_available) {
  message("\n========================================")
  message("METHOD 2: Targeted Maximum Likelihood Estimation (TMLE)")
  message("========================================\n")

  message("TMLE is a doubly robust method that:")
  message("  - Combines outcome regression and propensity score models")
  message("  - Uses targeted learning to reduce bias")
  message("  - Provides valid inference under model misspecification")
  message("  - Asymptotically efficient\n")

  message("Step 1: Preparing data for TMLE...")

  # TMLE requires data frame format
  tmle_data <- data.frame(
    Y = Y,
    A = A,
    W = X
  )

  # Split data
  tmle_train <- tmle_data[train_idx, ]
  tmle_test <- tmle_data[test_idx, ]

  # Fit TMLE on training data
  message("Step 2: Fitting TMLE model...")

  # Create SuperLearner library
  SL.library <- c("SL.glm", "SL.glm.interaction", "SL.gam", "SL.ranger")

  # Fit TMLE
  tmle_fit <- tmle(
    Y = tmle_train$Y,
    A = tmle_train$A,
    W = tmle_train[, grep("^W\\.", names(tmle_train))],
    Q.SL.library = SL.library,
    g.SL.library = SL.library,
    family = "binomial"
  )

  # Extract results
  ATE_tmle <- tmle_fit$estimates$ATE$psi
  ATE_tmle_se <- sqrt(tmle_fit$estimates$ATE$var.psi)
  ATE_tmle_ci <- tmle_fit$estimates$ATE$CI

  # Risk under treatment and control
  EY1 <- tmle_fit$estimates$EY1$psi
  EY0 <- tmle_fit$estimates$EY0$psi

  message("\n----------------------------------------")
  message("TMLE Results:")
  message("----------------------------------------")
  message(sprintf("ATE (Average Treatment Effect): %.4f", ATE_tmle))
  message(sprintf("  Standard Error: %.4f", ATE_tmle_se))
  message(sprintf("  95%% CI: [%.4f, %.4f]", ATE_tmle_ci[1], ATE_tmle_ci[2]))
  message(sprintf("  P-value: %.4f", tmle_fit$estimates$ATE$pvalue))
  message(sprintf("E[Y(1)] (Risk under treatment): %.4f", EY1))
  message(sprintf("E[Y(0)] (Risk under control): %.4f", EY0))

  # Predict on test set (using fitted Q and g)
  message("\nStep 3: Predicting on test set...")

  # For individual predictions, we need to refit or use the models
  # Here we'll do a simple approach: fit on full data and extract ITE

  tmle_full <- tmle(
    Y = tmle_data$Y,
    A = tmle_data$A,
    W = tmle_data[, grep("^W\\.", names(tmle_data))],
    Q.SL.library = SL.library,
    g.SL.library = SL.library,
    family = "binomial"
  )

  # Individual treatment effects (approximate)
  # TMLE doesn't directly give ITE, but we can extract from Q model
  ITE_tmle <- rep(ATE_tmle, length(Y_test))  # Simplified

  # Save TMLE results
  tmle_results <- list(
    model = tmle_fit,
    model_full = tmle_full,
    ATE = ATE_tmle,
    ATE_se = ATE_tmle_se,
    ATE_ci = ATE_tmle_ci,
    pvalue = tmle_fit$estimates$ATE$pvalue,
    EY1 = EY1,
    EY0 = EY0,
    ITE = ITE_tmle
  )

  saveRDS(tmle_results, "models/advanced_causal/tmle_model.rds")

  # Visualization
  message("\nCreating TMLE visualizations...")

  # Point estimate with CI
  tmle_plot_data <- data.frame(
    Estimate = c("ATE", "E[Y(1)]", "E[Y(0)]"),
    Value = c(ATE_tmle, EY1, EY0),
    Lower = c(ATE_tmle_ci[1], NA, NA),
    Upper = c(ATE_tmle_ci[2], NA, NA)
  )

  p_tmle_1 <- ggplot(tmle_plot_data[1, ], aes(x = Estimate, y = Value)) +
    geom_point(size = 5, color = "#56B4E9") +
    geom_errorbar(aes(ymin = Lower, ymax = Upper), width = 0.2, size = 1.2) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
    labs(
      title = "TMLE: Average Treatment Effect Estimate",
      subtitle = sprintf("ATE = %.4f (95%% CI: [%.4f, %.4f]), p = %.4f",
                         ATE_tmle, ATE_tmle_ci[1], ATE_tmle_ci[2],
                         tmle_fit$estimates$ATE$pvalue),
      y = "Treatment Effect (Risk Difference)",
      x = ""
    ) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(size = 12, face = "bold"),
      plot.title = element_text(size = 14, face = "bold")
    )

  ggsave("figures/advanced_causal/tmle_ate_estimate.png", p_tmle_1,
         width = 8, height = 6, dpi = 300)

  message("  TMLE visualizations saved!")

} else {
  message("\nTMLE analysis skipped (package not available)")
  tmle_results <- NULL
}

#==============================================================================#
#                    COMPARISON ACROSS ALL METHODS                             #
#==============================================================================#

message("\n========================================")
message("COMPARISON ACROSS ALL METHODS")
message("========================================\n")

# Collect all ATE estimates
all_estimates <- data.frame(
  Method = c("BART", "TMLE"),
  ATE = c(
    ifelse(bart_available, ATE_bart, NA),
    ifelse(tmle_available, ATE_tmle, NA)
  ),
  SE = c(
    ifelse(bart_available, ATE_bart_sd, NA),
    ifelse(tmle_available, ATE_tmle_se, NA)
  ),
  Lower_CI = c(
    ifelse(bart_available, ATE_bart_ci[1], NA),
    ifelse(tmle_available, ATE_tmle_ci[1], NA)
  ),
  Upper_CI = c(
    ifelse(bart_available, ATE_bart_ci[2], NA),
    ifelse(tmle_available, ATE_tmle_ci[2], NA)
  ),
  Type = c("Bayesian", "Frequentist")
) %>%
  filter(!is.na(ATE))

# Load previous results if available
if (file.exists("output/causal_ml/method_comparison.csv")) {
  previous_results <- read.csv("output/causal_ml/method_comparison.csv") %>%
    select(Method, ATE) %>%
    mutate(
      SE = NA,
      Lower_CI = NA,
      Upper_CI = NA,
      Type = "Machine Learning"
    )

  all_estimates <- bind_rows(all_estimates, previous_results)
}

if (nrow(all_estimates) > 0) {
  message("All Method Estimates:")
  print(all_estimates)

  # Save comparison
  write.csv(all_estimates, "output/advanced_causal/all_methods_comparison.csv",
            row.names = FALSE)

  # Forest plot
  p_comparison <- ggplot(all_estimates, aes(x = ATE, y = reorder(Method, ATE),
                                             color = Type)) +
    geom_point(size = 4) +
    geom_errorbarh(aes(xmin = Lower_CI, xmax = Upper_CI),
                   height = 0.3, size = 1) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "red") +
    scale_color_brewer(palette = "Set1") +
    labs(
      title = "Comparison of All Causal Inference Methods",
      subtitle = "Average Treatment Effect (ATE) estimates with 95% CI/Credible Intervals",
      x = "Average Treatment Effect (Risk Difference)",
      y = "",
      color = "Method Type"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      legend.position = "bottom"
    )

  ggsave("figures/advanced_causal/all_methods_forest_plot.png", p_comparison,
         width = 12, height = 8, dpi = 300)

  message("\nComparison plot saved!")
}

#==============================================================================#
#                    SUMMARY                                                   #
#==============================================================================#

message("\n========================================")
message("SUMMARY")
message("========================================\n")

sink("output/advanced_causal/analysis_summary.txt")
cat("ADVANCED CAUSAL METHODS ANALYSIS\n")
cat("=================================\n\n")
cat("Generated:", as.character(Sys.time()), "\n\n")

if (bart_available) {
  cat("BART RESULTS:\n")
  cat("-------------\n")
  cat(sprintf("ATE: %.4f (95%% Cred Int: [%.4f, %.4f])\n",
              ATE_bart, ATE_bart_ci[1], ATE_bart_ci[2]))
  cat(sprintf("Posterior SD: %.4f\n", ATE_bart_sd))
  cat(sprintf("ATT: %.4f\n", ATT_bart))
  cat(sprintf("ATC: %.4f\n\n", ATC_bart))
}

if (tmle_available) {
  cat("TMLE RESULTS:\n")
  cat("-------------\n")
  cat(sprintf("ATE: %.4f (95%% CI: [%.4f, %.4f])\n",
              ATE_tmle, ATE_tmle_ci[1], ATE_tmle_ci[2]))
  cat(sprintf("SE: %.4f\n", ATE_tmle_se))
  cat(sprintf("P-value: %.4f\n", tmle_fit$estimates$ATE$pvalue))
  cat(sprintf("E[Y(1)]: %.4f\n", EY1))
  cat(sprintf("E[Y(0)]: %.4f\n\n", EY0))
}

cat("OUTPUT FILES:\n")
cat("-------------\n")
cat("  Models: models/advanced_causal/\n")
cat("  Results: output/advanced_causal/\n")
cat("  Figures: figures/advanced_causal/\n")

sink()

message("Analysis complete!")
message("Results saved to: output/advanced_causal/")
