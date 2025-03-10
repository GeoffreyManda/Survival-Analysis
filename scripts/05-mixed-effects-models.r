# 05_mixed_effects_models.R
# COVID-19 Survival Analysis - Mixed-Effects Models
# This script fits mixed-effects models to account for hospital clustering

# Load required packages
if (!require("pacman")) install.packages("pacman")
pacman::p_load(coxme, lme4, dplyr, ggplot2, survival, lattice, sjPlot)

# Load preprocessed data
if (!file.exists("data/processed_covid_data.RData")) {
  stop("Processed data file not found. Run 01_data_preprocessing.R first.")
}

load("data/processed_covid_data.RData")

# Create output directories
dir.create("figures", showWarnings = FALSE)
dir.create("output", showWarnings = FALSE)
dir.create("models", showWarnings = FALSE)

#---------------------------#
# 1. Mixed-Effects Cox Model #
#---------------------------#
message("Fitting mixed-effects Cox model...")

# 1.1 Fit model with hospital as random effect
cox_mixed <- tryCatch({
  coxme(Surv(LoS, mort_hospital) ~ gender + age + wave + Ct + (1 | hosp_id), 
      data = Covid_data)
}, error = function(e) {
  message("Error fitting mixed-effects Cox model: ", e$message)
  return(NULL)
})

if (!is.null(cox_mixed)) {
  # Save model summary
  sink("output/cox_mixed_effects.txt")
  print(summary(cox_mixed))
  sink()
  
  # Save model object
  saveRDS(cox_mixed, "models/cox_mixed_effects.rds")
  
  # Check proportional hazards assumption if possible
  tryCatch({
    ph_test <- cox.zph(cox_mixed)
    sink("output/cox_mixed_ph_test.txt")
    print(ph_test)
    sink()
  }, error = function(e) {
    message("Could not test proportional hazards assumption for mixed model: ", e$message)
  })
}

#-------------------------------------#
# 2. Mixed-Effects Logistic Regression #
#-------------------------------------#
message("Fitting mixed-effects logistic regression for mortality...")

# 2.1 Fit generalized linear mixed model for binary outcome
glmm_model <- glmer(
  mort_hospital ~ age + gender + Ct + LoS + wave + (1 | hosp_id),
  family = binomial,
  data = Covid_data
)

# Save model summary
sink("output/glmm_mortality.txt")
print(summary(glmm_model))
sink()

# Save model object
saveRDS(glmm_model, "models/glmm_mortality.rds")

# Create odds ratio plot
or_plot <- sjPlot::plot_model(
  glmm_model, 
  title = "Odds Ratios for Mortality (Mixed-Effects Logistic Regression)",
  show.values = TRUE, 
  value.offset = 0.3
)

# Save odds ratio plot
suppressMessages(
  ggsave("figures/glmm_odds_ratios.png", or_plot, width = 10, height = 6)
)
is_singular <- isSingular(glmm_model)
print(is_singular)
# Calculate ICC (Intraclass Correlation Coefficient)
icc_value <- performance::icc(glmm_model)
sink("output/glmm_icc.txt")
cat("Intraclass Correlation Coefficient (ICC):", icc_value, "\n")
cat("This represents the proportion of variance explained by the hospital clustering.\n")
sink()

#-------------------------------------#
# 3. Mixed-Effects Model for Length of Stay #
#-------------------------------------#
message("Fitting mixed-effects linear models for length of stay...")

# 3.1 Log-transform length of stay for better normality
Covid_data$log_LoS <- log(Covid_data$LoS)

# 3.2 Fit linear mixed model
lmm_los <- lmer(log_LoS ~ age + gender + Ct + wave + (1 | hosp_id), 
               data = Covid_data)

# Save model summary
sink("output/lmm_length_of_stay.txt")
print(summary(lmm_los))
sink()

# Save model object
saveRDS(lmm_los, "models/lmm_length_of_stay.rds")

# 3.3 Create diagnostic plots
# Residual plots
png("figures/lmm_residual_plots.png", width = 1000, height = 800)
par(mfrow = c(2, 2))
plot(lmm_los)
dev.off()

# Random effects plot
png("figures/lmm_random_effects.png", width = 900, height = 600)
dotplot(ranef(lmm_los, condVar=TRUE))
dev.off()

# Fixed effects plot
fe_plot <- sjPlot::plot_model(
  lmm_los, 
  title = "Fixed Effects for Length of Stay (Mixed-Effects Model)",
  show.values = TRUE, 
  value.offset = 0.3
)

# Save fixed effects plot
suppressMessages(
  ggsave("figures/lmm_fixed_effects.png", fe_plot, width = 10, height = 6)
)

# 3.4 Fit alternative models for sensitivity analysis
# Model with different variable ordering
lmm_los_alt1 <- lmer(log_LoS ~ gender + age + Ct + wave + (1 | hosp_id), 
                    data = Covid_data)

lmm_los_alt2 <- lmer(log_LoS ~ Ct + age + gender + wave + (1 | hosp_id), 
                    data = Covid_data)

# Compare models
anova_result <- anova(lmm_los, lmm_los_alt1, lmm_los_alt2)

# Save comparison
sink("output/lmm_model_comparison.txt")
print(anova_result)
sink()

# 3.5 Calculate ICC for length of stay model
icc_los <- performance::icc(lmm_los)
icc_los <- as.data.frame(icc_los)
sink("output/lmm_los_icc.txt")
cat("This represents the proportion of variance in length of stay explained by hospital clustering.\n")
sink()

# 3.6 Make predictions from the model
# Create a grid of typical values
prediction_grid <- expand.grid(
  age = c(30, 50, 70),  # Young, middle-aged, elderly
  gender = factor(c(0, 1), levels = c(0, 1)),  # Female, Male
  Ct = c(20, 25, 30),  # Different viral loads
  wave = factor(1:3)  # Three waves
)

# Make predictions
prediction_grid$predicted_log_los <- predict(lmm_los, newdata = prediction_grid, re.form = NA)
prediction_grid$predicted_los <- exp(prediction_grid$predicted_log_los)

# Save predictions
write.csv(prediction_grid, "output/los_predictions.csv", row.names = FALSE)

# Create a prediction visualization
pred_plot <- ggplot(prediction_grid, aes(x = age, y = predicted_los, color = wave, shape = gender)) +
  geom_point(size = 3) +
  geom_line(aes(group = interaction(wave, gender))) +
  facet_wrap(~ Ct, labeller = labeller(Ct = function(x) paste("Ct =", x))) +
  labs(
    title = "Predicted Length of Stay by Age, Gender, CT Value, and Wave",
    x = "Age (years)",
    y = "Predicted Length of Stay (days)",
    color = "Wave",
    shape = "Gender"
  ) +
  scale_shape_manual(values = c(16, 17), labels = c("Female", "Male")) +
  theme_bw()
dev.new()
pred_plot
# Save prediction plot
ggsave("figures/los_predictions.png", pred_plot, width = 10, height = 8)

message("Mixed-effects modeling complete.")

