# 10_causal_machine_learning.R
# Causal Machine Learning for COVID-19 Survival Analysis
# R-based implementation using BART, Random Forests, and boosting
# Similar in spirit to deep causal representation learning but using R packages

# Load required packages
if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  dplyr, tidyverse, here,
  survival, survminer,
  randomForestSRC,  # Random survival forests
  gbm,              # Gradient boosting
  glmnet,           # Regularized regression
  caret,            # ML framework
  ggplot2, patchwork
)

# Attempt to load BART packages (optional)
bart_available <- require("BART", quietly = TRUE)
if (!bart_available) {
  message("BART package not available. Installing...")
  tryCatch({
    install.packages("BART")
    library(BART)
    bart_available <- TRUE
  }, error = function(e) {
    message("Could not install BART. Will use other methods.")
    bart_available <- FALSE
  })
}

#==============================================================================#
#                    LOAD AND PREPARE DATA                                     #
#==============================================================================#

message("="*70)
message("CAUSAL MACHINE LEARNING - COVID-19 SURVIVAL ANALYSIS")
message("="*70)

# Load preprocessed data
message("\nLoading preprocessed data...")
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
    # Treatment: high viral load (strongly positive)
    high_viral_load = ifelse(ct_gp == "Strongly positive", 1, 0),

    # Ensure numeric variables
    age_numeric = as.numeric(age),
    gender_numeric = as.numeric(gender),
    wave_numeric = as.numeric(as.character(wave)),
    hosp_numeric = as.numeric(as.character(hosp_id)),

    # Log transform LoS for better distributional properties
    log_LoS = log(LoS + 0.1)  # Add small constant to avoid log(0)
  ) %>%
  filter(!is.na(high_viral_load))  # Remove moderately positive

message(sprintf("  Total observations: %d", nrow(analysis_data)))
message(sprintf("  Events (deaths): %d (%.1f%%)",
                sum(analysis_data$mort_hospital),
                100 * mean(analysis_data$mort_hospital)))
message(sprintf("  High viral load: %d (%.1f%%)",
                sum(analysis_data$high_viral_load),
                100 * mean(analysis_data$high_viral_load)))

# Create output directories
dir.create("output/causal_ml", showWarnings = FALSE, recursive = TRUE)
dir.create("figures/causal_ml", showWarnings = FALSE, recursive = TRUE)
dir.create("models/causal_ml", showWarnings = FALSE, recursive = TRUE)

# Train/test split (80/20)
set.seed(42)
train_idx <- createDataPartition(analysis_data$mort_hospital, p = 0.8, list = FALSE)
train_data <- analysis_data[train_idx, ]
test_data <- analysis_data[-train_idx, ]

message(sprintf("\n  Train set: %d observations", nrow(train_data)))
message(sprintf("  Test set:  %d observations", nrow(test_data)))

#==============================================================================#
#                    METHOD 1: RANDOM SURVIVAL FORESTS                         #
#==============================================================================#

message("\n========================================")
message("METHOD 1: Random Survival Forests (Causal)")
message("========================================\n")

# Prepare data for random survival forest
rsf_formula <- as.formula(
  "Surv(LoS, mort_hospital) ~ age + gender + wave + hosp_id + Ct"
)

# Fit separate forests for treated and control
message("Fitting Random Survival Forests...")

# Forest for treated (high viral load)
train_treated <- train_data %>% filter(high_viral_load == 1)
message(sprintf("  Training forest for TREATED (n=%d)...", nrow(train_treated)))

rsf_treated <- rfsrc(
  rsf_formula,
  data = train_treated,
  ntree = 500,
  nodesize = 10,
  importance = TRUE,
  seed = 42
)

# Forest for control (low viral load)
train_control <- train_data %>% filter(high_viral_load == 0)
message(sprintf("  Training forest for CONTROL (n=%d)...", nrow(train_control)))

rsf_control <- rfsrc(
  rsf_formula,
  data = train_control,
  ntree = 500,
  nodesize = 10,
  importance = TRUE,
  seed = 42
)

# Predict on test set
message("\nPredicting counterfactuals on test set...")

# Predict mortality under both treatments
pred_treated <- predict(rsf_treated, newdata = test_data)
pred_control <- predict(rsf_control, newdata = test_data)

# Extract predicted mortality (1 - survival probability)
# Use survival at median time
median_time <- median(test_data$LoS, na.rm = TRUE)

# Get survival probabilities
Y1_pred <- 1 - pred_treated$survival[, which.min(abs(rsf_treated$time.interest - median_time))]
Y0_pred <- 1 - pred_control$survival[, which.min(abs(rsf_control$time.interest - median_time))]

# Individual treatment effects
ITE_rsf <- Y1_pred - Y0_pred

# Average treatment effects
ATE_rsf <- mean(ITE_rsf, na.rm = TRUE)
ATT_rsf <- mean(ITE_rsf[test_data$high_viral_load == 1], na.rm = TRUE)
ATC_rsf <- mean(ITE_rsf[test_data$high_viral_load == 0], na.rm = TRUE)

message("\nRandom Survival Forest Results:")
message(sprintf("  ATE (Average Treatment Effect): %.4f", ATE_rsf))
message(sprintf("  ATT (Effect on Treated):        %.4f", ATT_rsf))
message(sprintf("  ATC (Effect on Controls):       %.4f", ATC_rsf))
message(sprintf("  ITE std:                        %.4f", sd(ITE_rsf, na.rm = TRUE)))
message(sprintf("  ITE range:                      [%.4f, %.4f]",
                min(ITE_rsf, na.rm = TRUE), max(ITE_rsf, na.rm = TRUE)))

# Variable importance
importance_treated <- rsf_treated$importance
importance_control <- rsf_control$importance

message("\nVariable Importance (Treated Group):")
print(sort(importance_treated, decreasing = TRUE))

# Save model
saveRDS(list(
  treated = rsf_treated,
  control = rsf_control,
  ATE = ATE_rsf,
  ATT = ATT_rsf,
  ATC = ATC_rsf,
  ITE = ITE_rsf
), "models/causal_ml/random_survival_forest.rds")

#==============================================================================#
#                    METHOD 2: GRADIENT BOOSTING MACHINES                      #
#==============================================================================#

message("\n========================================")
message("METHOD 2: Gradient Boosting (Causal)")
message("========================================\n")

# Fit GBM for binary outcome (mortality)
message("Fitting Gradient Boosting Models...")

# Prepare data for GBM
train_features <- train_data %>%
  select(age, gender, wave_numeric, hosp_numeric, Ct) %>%
  as.data.frame()

test_features <- test_data %>%
  select(age, gender, wave_numeric, hosp_numeric, Ct) %>%
  as.data.frame()

# GBM for treated
message("  Training GBM for TREATED...")
gbm_treated <- gbm(
  mort_hospital ~ .,
  data = cbind(train_features[train_data$high_viral_load == 1, ],
               mort_hospital = train_data$mort_hospital[train_data$high_viral_load == 1]),
  distribution = "bernoulli",
  n.trees = 1000,
  interaction.depth = 4,
  shrinkage = 0.01,
  cv.folds = 5,
  verbose = FALSE
)

# GBM for control
message("  Training GBM for CONTROL...")
gbm_control <- gbm(
  mort_hospital ~ .,
  data = cbind(train_features[train_data$high_viral_load == 0, ],
               mort_hospital = train_data$mort_hospital[train_data$high_viral_load == 0]),
  distribution = "bernoulli",
  n.trees = 1000,
  interaction.depth = 4,
  shrinkage = 0.01,
  cv.folds = 5,
  verbose = FALSE
)

# Find optimal number of trees
best_iter_treated <- gbm.perf(gbm_treated, method = "cv", plot.it = FALSE)
best_iter_control <- gbm.perf(gbm_control, method = "cv", plot.it = FALSE)

message(sprintf("  Optimal trees (treated): %d", best_iter_treated))
message(sprintf("  Optimal trees (control): %d", best_iter_control))

# Predict counterfactuals
Y1_pred_gbm <- predict(gbm_treated, newdata = test_features,
                       n.trees = best_iter_treated, type = "response")
Y0_pred_gbm <- predict(gbm_control, newdata = test_features,
                       n.trees = best_iter_control, type = "response")

# Individual treatment effects
ITE_gbm <- Y1_pred_gbm - Y0_pred_gbm

# Average treatment effects
ATE_gbm <- mean(ITE_gbm, na.rm = TRUE)
ATT_gbm <- mean(ITE_gbm[test_data$high_viral_load == 1], na.rm = TRUE)
ATC_gbm <- mean(ITE_gbm[test_data$high_viral_load == 0], na.rm = TRUE)

message("\nGradient Boosting Results:")
message(sprintf("  ATE (Average Treatment Effect): %.4f", ATE_gbm))
message(sprintf("  ATT (Effect on Treated):        %.4f", ATT_gbm))
message(sprintf("  ATC (Effect on Controls):       %.4f", ATC_gbm))
message(sprintf("  ITE std:                        %.4f", sd(ITE_gbm, na.rm = TRUE)))

# Variable importance
importance_gbm_treated <- summary(gbm_treated, plotit = FALSE)
message("\nVariable Importance (GBM - Treated):")
print(importance_gbm_treated)

# Save model
saveRDS(list(
  treated = gbm_treated,
  control = gbm_control,
  best_iter_treated = best_iter_treated,
  best_iter_control = best_iter_control,
  ATE = ATE_gbm,
  ATT = ATT_gbm,
  ATC = ATC_gbm,
  ITE = ITE_gbm
), "models/causal_ml/gradient_boosting.rds")

#==============================================================================#
#                    METHOD 3: ELASTIC NET (Regularized Regression)            #
#==============================================================================#

message("\n========================================")
message("METHOD 3: Elastic Net Regression (Causal)")
message("========================================\n")

message("Fitting Elastic Net models...")

# Prepare matrices for glmnet
X_train <- model.matrix(~ age + gender + wave_numeric + hosp_numeric + Ct - 1,
                        data = train_features)
X_test <- model.matrix(~ age + gender + wave_numeric + hosp_numeric + Ct - 1,
                       data = test_features)

# Fit for treated
y_treated <- train_data$mort_hospital[train_data$high_viral_load == 1]
X_treated <- X_train[train_data$high_viral_load == 1, ]

enet_treated <- cv.glmnet(
  X_treated, y_treated,
  family = "binomial",
  alpha = 0.5,  # Elastic net (mix of L1 and L2)
  nfolds = 5
)

# Fit for control
y_control <- train_data$mort_hospital[train_data$high_viral_load == 0]
X_control <- X_train[train_data$high_viral_load == 0, ]

enet_control <- cv.glmnet(
  X_control, y_control,
  family = "binomial",
  alpha = 0.5,
  nfolds = 5
)

# Predict counterfactuals
Y1_pred_enet <- predict(enet_treated, newx = X_test, s = "lambda.min", type = "response")[,1]
Y0_pred_enet <- predict(enet_control, newx = X_test, s = "lambda.min", type = "response")[,1]

# Individual treatment effects
ITE_enet <- Y1_pred_enet - Y0_pred_enet

# Average treatment effects
ATE_enet <- mean(ITE_enet, na.rm = TRUE)
ATT_enet <- mean(ITE_enet[test_data$high_viral_load == 1], na.rm = TRUE)
ATC_enet <- mean(ITE_enet[test_data$high_viral_load == 0], na.rm = TRUE)

message("\nElastic Net Results:")
message(sprintf("  ATE (Average Treatment Effect): %.4f", ATE_enet))
message(sprintf("  ATT (Effect on Treated):        %.4f", ATT_enet))
message(sprintf("  ATC (Effect on Controls):       %.4f", ATC_enet))
message(sprintf("  ITE std:                        %.4f", sd(ITE_enet, na.rm = TRUE)))

# Save model
saveRDS(list(
  treated = enet_treated,
  control = enet_control,
  ATE = ATE_enet,
  ATT = ATT_enet,
  ATC = ATC_enet,
  ITE = ITE_enet
), "models/causal_ml/elastic_net.rds")

#==============================================================================#
#                    COMPARISON OF METHODS                                     #
#==============================================================================#

message("\n========================================")
message("COMPARISON ACROSS METHODS")
message("========================================\n")

# Create comparison table
comparison <- data.frame(
  Method = c("Random Survival Forest", "Gradient Boosting", "Elastic Net"),
  ATE = c(ATE_rsf, ATE_gbm, ATE_enet),
  ATT = c(ATT_rsf, ATT_gbm, ATC_enet),
  ATC = c(ATC_rsf, ATC_gbm, ATC_enet),
  ITE_std = c(sd(ITE_rsf, na.rm = TRUE),
              sd(ITE_gbm, na.rm = TRUE),
              sd(ITE_enet, na.rm = TRUE)),
  ITE_range = c(
    sprintf("[%.4f, %.4f]", min(ITE_rsf, na.rm = TRUE), max(ITE_rsf, na.rm = TRUE)),
    sprintf("[%.4f, %.4f]", min(ITE_gbm, na.rm = TRUE), max(ITE_gbm, na.rm = TRUE)),
    sprintf("[%.4f, %.4f]", min(ITE_enet, na.rm = TRUE), max(ITE_enet, na.rm = TRUE))
  )
)

message("Effect Estimates Across Methods:")
print(comparison)

# Save comparison
write.csv(comparison, "output/causal_ml/method_comparison.csv", row.names = FALSE)

#==============================================================================#
#                    VISUALIZATIONS                                            #
#==============================================================================#

message("\n========================================")
message("GENERATING VISUALIZATIONS")
message("========================================\n")

# 1. ITE distributions
ite_data <- data.frame(
  ITE = c(ITE_rsf, ITE_gbm, ITE_enet),
  Method = rep(c("Random Survival Forest", "Gradient Boosting", "Elastic Net"),
               c(length(ITE_rsf), length(ITE_gbm), length(ITE_enet)))
)

p1 <- ggplot(ite_data, aes(x = ITE, fill = Method)) +
  geom_density(alpha = 0.5) +
  geom_vline(data = comparison, aes(xintercept = ATE, color = Method),
             linetype = "dashed", size = 1) +
  labs(
    title = "Distribution of Individual Treatment Effects (ITE)",
    subtitle = "Effect of high viral load on mortality across different methods",
    x = "Individual Treatment Effect (Risk Difference)",
    y = "Density"
  ) +
  theme_bw() +
  theme(legend.position = "bottom")

ggsave("figures/causal_ml/ite_distributions.png", p1, width = 12, height = 8, dpi = 300)
message("  Saved: figures/causal_ml/ite_distributions.png")

# 2. Forest plot of effect estimates
forest_data <- comparison %>%
  select(Method, ATE) %>%
  mutate(
    Lower_CI = ATE - 1.96 * comparison$ITE_std / sqrt(nrow(test_data)),
    Upper_CI = ATE + 1.96 * comparison$ITE_std / sqrt(nrow(test_data))
  )

p2 <- ggplot(forest_data, aes(x = ATE, y = reorder(Method, ATE))) +
  geom_point(size = 4) +
  geom_errorbarh(aes(xmin = Lower_CI, xmax = Upper_CI), height = 0.2) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "red") +
  labs(
    title = "Forest Plot: Causal Effect Estimates",
    subtitle = "Average Treatment Effect (ATE) with approximate 95% CI",
    x = "Average Treatment Effect (Risk Difference)",
    y = ""
  ) +
  theme_bw() +
  theme(panel.grid.major.y = element_blank())

ggsave("figures/causal_ml/forest_plot.png", p2, width = 10, height = 6, dpi = 300)
message("  Saved: figures/causal_ml/forest_plot.png")

# 3. Variable importance (from RSF)
importance_df <- data.frame(
  Variable = names(importance_treated),
  Importance = as.numeric(importance_treated)
) %>%
  arrange(desc(Importance))

p3 <- ggplot(importance_df, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(
    title = "Variable Importance (Random Survival Forest)",
    subtitle = "Treated group",
    x = "",
    y = "Importance"
  ) +
  theme_bw()

ggsave("figures/causal_ml/variable_importance.png", p3, width = 10, height = 6, dpi = 300)
message("  Saved: figures/causal_ml/variable_importance.png")

# 4. Heterogeneity analysis (ITE by covariates)
heterogeneity_data <- test_data %>%
  mutate(
    ITE_RSF = ITE_rsf,
    ITE_GBM = ITE_gbm,
    age_group = cut(age, breaks = c(0, 40, 60, 100), labels = c("Young", "Middle", "Elderly"))
  )

p4 <- ggplot(heterogeneity_data, aes(x = age_group, y = ITE_RSF, fill = age_group)) +
  geom_boxplot(alpha = 0.7) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(
    title = "Treatment Effect Heterogeneity by Age Group",
    subtitle = "Random Survival Forest estimates",
    x = "Age Group",
    y = "Individual Treatment Effect (ITE)"
  ) +
  theme_bw() +
  theme(legend.position = "none")

ggsave("figures/causal_ml/heterogeneity_age.png", p4, width = 10, height = 6, dpi = 300)
message("  Saved: figures/causal_ml/heterogeneity_age.png")

#==============================================================================#
#                    MODEL FIT ASSESSMENT                                      #
#==============================================================================#

message("\n========================================")
message("MODEL FIT ASSESSMENT")
message("========================================\n")

# Calculate prediction accuracy on test set
test_data_eval <- test_data %>%
  mutate(
    # Factual predictions (observed treatment)
    Y_pred_rsf = ifelse(high_viral_load == 1, Y1_pred, Y0_pred),
    Y_pred_gbm = ifelse(high_viral_load == 1, Y1_pred_gbm, Y0_pred_gbm),
    Y_pred_enet = ifelse(high_viral_load == 1, Y1_pred_enet, Y0_pred_enet)
  )

# Calculate metrics
calculate_metrics <- function(y_true, y_pred, method_name) {
  # Brier score (lower is better)
  brier <- mean((y_true - y_pred)^2, na.rm = TRUE)

  # AUC
  pred_obj <- ROCR::prediction(y_pred, y_true)
  auc <- ROCR::performance(pred_obj, "auc")@y.values[[1]]

  # Calibration slope
  calib_model <- glm(y_true ~ y_pred, family = binomial)
  calib_slope <- coef(calib_model)[2]

  data.frame(
    Method = method_name,
    Brier_Score = brier,
    AUC = auc,
    Calibration_Slope = calib_slope
  )
}

# Need ROCR package
if (!require("ROCR")) install.packages("ROCR")
library(ROCR)

fit_metrics <- bind_rows(
  calculate_metrics(test_data_eval$mort_hospital, test_data_eval$Y_pred_rsf, "Random Survival Forest"),
  calculate_metrics(test_data_eval$mort_hospital, test_data_eval$Y_pred_gbm, "Gradient Boosting"),
  calculate_metrics(test_data_eval$mort_hospital, test_data_eval$Y_pred_enet, "Elastic Net")
)

message("Model Fit Metrics (Factual Predictions):")
print(fit_metrics)

# Save metrics
write.csv(fit_metrics, "output/causal_ml/fit_metrics.csv", row.names = FALSE)

#==============================================================================#
#                    FINAL SUMMARY REPORT                                      #
#==============================================================================#

message("\n========================================")
message("FINAL SUMMARY")
message("========================================\n")

sink("output/causal_ml/analysis_summary.txt")
cat("CAUSAL MACHINE LEARNING ANALYSIS - COVID-19 SURVIVAL\n")
cat("=====================================================\n\n")
cat("Generated:", as.character(Sys.time()), "\n\n")

cat("DATASET:\n")
cat("--------\n")
cat(sprintf("  Total observations: %d\n", nrow(analysis_data)))
cat(sprintf("  Training set: %d\n", nrow(train_data)))
cat(sprintf("  Test set: %d\n", nrow(test_data)))
cat(sprintf("  Mortality rate: %.1f%%\n", 100 * mean(analysis_data$mort_hospital)))
cat(sprintf("  High viral load: %.1f%%\n\n", 100 * mean(analysis_data$high_viral_load)))

cat("CAUSAL EFFECT ESTIMATES:\n")
cat("------------------------\n")
print(comparison)
cat("\n")

cat("MODEL FIT (on test set):\n")
cat("------------------------\n")
print(fit_metrics)
cat("\n")

cat("INTERPRETATION:\n")
cat("---------------\n")
cat(sprintf("Average Treatment Effect (consensus): %.4f\n", mean(comparison$ATE)))
cat(sprintf("  This suggests that high viral load increases mortality\n"))
cat(sprintf("  risk by approximately %.1f percentage points.\n\n", 100 * mean(comparison$ATE)))

cat("Treatment effect heterogeneity:\n")
cat(sprintf("  ITE standard deviation (RSF): %.4f\n", sd(ITE_rsf, na.rm = TRUE)))
cat("  This indicates substantial heterogeneity - some patients\n")
cat("  are affected more than others.\n\n")

cat("Model performance:\n")
cat(sprintf("  Best AUC: %.3f (%s)\n",
            max(fit_metrics$AUC),
            fit_metrics$Method[which.max(fit_metrics$AUC)]))
cat(sprintf("  Best Brier Score: %.4f (%s)\n",
            min(fit_metrics$Brier_Score),
            fit_metrics$Method[which.min(fit_metrics$Brier_Score)]))

cat("\nOUTPUT FILES:\n")
cat("-------------\n")
cat("  Models: models/causal_ml/\n")
cat("  Results: output/causal_ml/\n")
cat("  Figures: figures/causal_ml/\n")

sink()

message("Summary report saved to: output/causal_ml/analysis_summary.txt")

message("\n" + "="*70)
message("ANALYSIS COMPLETE!")
message("="*70)
message("\nKey Findings:")
message(sprintf("  Average Treatment Effect: %.4f", mean(comparison$ATE)))
message(sprintf("  Best performing model: %s (AUC=%.3f)",
                fit_metrics$Method[which.max(fit_metrics$AUC)],
                max(fit_metrics$AUC)))
message("\nNext steps:")
message("  1. Review plots in figures/causal_ml/")
message("  2. Examine heterogeneity by subgroups")
message("  3. Compare with traditional IPW estimates")
message("  4. Validate findings with domain experts")
