# 04_parametric_models.R
# COVID-19 Survival Analysis - Parametric Models 
# This script fits parametric survival models, including flexible parametric and AFT models

# Load required packages
if (!require("pacman")) install.packages("pacman")
pacman::p_load(flexsurv, survival, gtsummary, dplyr, ggplot2, MASS)

# Load preprocessed data
if (!file.exists("data/processed_covid_data.RData")) {
  stop("Processed data file not found. Run 01_data_preprocessing.R first.")
}

load("data/processed_covid_data.RData")

# Create output directories
dir.create("figures", showWarnings = FALSE)
dir.create("output", showWarnings = FALSE)
dir.create("models", showWarnings = FALSE)

#-------------------------------#
# 1. Flexible Parametric Models #
#-------------------------------#
message("Fitting flexible parametric survival models...")

# 1.1 Find best distribution for flexible model
message("  Comparing distributions for flexible parametric model...")

# Define distributions to try
distributions <- c("gengamma", "gengamma.orig", "genf", "genf.orig", "weibull", 
                  "gamma", "exp", "llogis", "lnorm", "gompertz")

# Track the best model
best_model <- NULL
best_aic <- Inf
best_dist <- ""
model_comparison <- data.frame(
  Distribution = character(),
  AIC = numeric(),
  BIC = numeric(),
  LogLik = numeric(),
  stringsAsFactors = FALSE
)

# Try each distribution
for (dist in distributions) {
  tryCatch({
    message(paste("    Trying", dist, "distribution..."))
    # Fit model with current distribution
    current_model <- flexsurvreg(
      Surv(LoS, mort_hospital == 1) ~ age + sex + Ct + wave, 
      dist = dist, data = Covid_data
    )
    
    # Calculate criteria
    current_aic <- AIC(current_model)
    current_bic <- BIC(current_model)
    current_loglik <- current_model$loglik
    
    # Add to comparison table
    model_comparison <- rbind(
      model_comparison,
      data.frame(
        Distribution = dist,
        AIC = current_aic,
        BIC = current_bic,
        LogLik = current_loglik,
        stringsAsFactors = FALSE
      )
    )
    
    # Update best model if this one is better
    if (current_aic < best_aic) {
      best_model <- current_model
      best_aic <- current_aic
      best_dist <- dist
    }
  }, error = function(e) {
    message(paste("    Error with", dist, "distribution:", e$message))
    model_comparison <- rbind(
      model_comparison,
      data.frame(
        Distribution = dist,
        AIC = NA,
        BIC = NA,
        LogLik = NA,
        stringsAsFactors = FALSE
      )
    )
  })
}

# Save model comparison
write.csv(model_comparison, "output/flexsurv_distribution_comparison.csv", row.names = FALSE)

# Get the best distribution
message(paste("  Best distribution:", best_dist))

# 1.2 Fit the best model
message(paste("  Fitting flexible parametric model with", best_dist, "distribution..."))
flex_best <- flexsurvreg(
  Surv(LoS, mort_hospital == 1) ~ age + sex + Ct + wave, 
  dist = best_dist, data = Covid_data
)

# Save model summary
sink("output/flexsurv_best_model.txt")
print(summary(flex_best))
sink()

# Save model object
saveRDS(flex_best, "models/flexsurv_best_model.rds")

# 1.3 Visualize flexible model
message("  Creating flexible parametric model plots...")
# Survival curves
png("figures/flexsurv_survival.png", width = 900, height = 700)
plot(flex_best, type = "survival", main = paste("Flexible Parametric Survival Model -", best_dist),
     xlab = "Time (days)", ylab = "Survival probability", las = 1)
dev.off()

# Residuals
png("figures/flexsurv_residuals.png", width = 900, height = 700)
#   'type arg' should be one of “survival”, “cumhaz”, “hazard”, “rmst”, “mean”, “median”, “quantile”, “link”

plot(flex_best, type = "survival", main = paste("Flexible Parametric Model Residuals -", best_dist),
     xlab = "Time (days)", ylab = "Martingale residuals", las = 1)
dev.off()

# QQ plot of residuals
residuals <- residuals(flex_best)
png("figures/flexsurv_qq_plot.png", width = 900, height = 700)
qqnorm(residuals, main = paste("Normal Q-Q Plot for", best_dist, "Model"))
qqline(residuals)
dev.off()

#---------------------------------#
# 2. Accelerated Failure Time Models #
#---------------------------------#
message("Fitting Accelerated Failure Time (AFT) models...")

# 2.1 Lognormal AFT model
message("  Fitting lognormal AFT model...")
aft_lnorm <- survreg(
  Surv(LoS, mort_hospital) ~ age + factor(gender) + wave + Ct, 
  data = Covid_data, 
  dist = "lognormal"
)

# Save model summary
sink("output/aft_lognormal.txt")
print(summary(aft_lnorm))
cat("\n\nExponentiated Coefficients (for interpretation):\n")
print(exp(aft_lnorm$coefficients))
cat("\n\nConfidence Intervals (exponentiated):\n")
print(exp(confint(aft_lnorm)))
sink()

# Save model object
saveRDS(aft_lnorm, "models/aft_lognormal.rds")

# 2.2 Weibull AFT model
message("  Fitting Weibull AFT model...")
aft_weibull <- survreg(
  Surv(LoS, mort_hospital) ~ age + factor(gender) + wave + Ct, 
  data = Covid_data, 
  dist = "weibull"
)

# Save model summary
sink("output/aft_weibull.txt")
print(summary(aft_weibull))
cat("\n\nExponentiated Coefficients (for interpretation):\n")
print(exp(aft_weibull$coefficients))
cat("\n\nConfidence Intervals (exponentiated):\n")
print(exp(confint(aft_weibull)))
sink()

# Save model object
saveRDS(aft_weibull, "models/aft_weibull.rds")

# 2.3 Compare AFT models
message("  Comparing AFT models...")
aft_comparison <- data.frame(
  Model = c("Lognormal AFT", "Weibull AFT"),
  AIC = c(AIC(aft_lnorm), AIC(aft_weibull)),
  BIC = c(BIC(aft_lnorm), BIC(aft_weibull)),
  LogLik = c(aft_lnorm$loglik[2], aft_weibull$loglik[2]),
  stringsAsFactors = FALSE
)

# Save model comparison
write.csv(aft_comparison, "output/aft_model_comparison.csv", row.names = FALSE)

# Determine best AFT model
best_aft <- ifelse(aft_comparison$AIC[1] < aft_comparison$AIC[2], "Lognormal", "Weibull")
message(paste("  Best AFT model:", best_aft))

# 2.4 Visualize best AFT model with flexsurv (for plotting capabilities)
message("  Creating AFT model plots...")
if (best_aft == "Lognormal") {
  flex_aft <- flexsurvreg(
    Surv(LoS, mort_hospital) ~ age + factor(gender) + wave + Ct, 
    data = Covid_data, 
    dist = "lognormal"
  )
} else {
  flex_aft <- flexsurvreg(
    Surv(LoS, mort_hospital) ~ age + factor(gender) + wave + Ct, 
    data = Covid_data, 
    dist = "weibull"
  )
}

# Survival curves
png("figures/aft_survival.png", width = 900, height = 700)
plot(flex_aft, type = "survival", main = paste(best_aft, "AFT Model"),
     xlab = "Time (days)", ylab = "Survival probability", las = 1)
dev.off()

# QQ plot of residuals (for best AFT model)
if (best_aft == "Lognormal") {
  aft_residuals <- residuals(aft_lnorm)
} else {
  aft_residuals <- residuals(aft_weibull)
}

png("figures/aft_qq_plot.png", width = 900, height = 700)
qqnorm(aft_residuals, main = paste("Normal Q-Q Plot for", best_aft, "AFT Model"))
qqline(aft_residuals)
dev.off()

#----------------------------------#
# 3. Overall Model Comparison      #
#----------------------------------#
message("Comparing all parametric models...")

# Load Cox model for comparison (if available)
cox_model_path <- "models/cox_full_model.rds"
if (file.exists(cox_model_path)) {
  cox_full <- readRDS(cox_model_path)
  
  # Create comprehensive model comparison
  overall_comparison <- data.frame(
    Model = c("Cox PH", "Flexible Parametric", "Lognormal AFT", "Weibull AFT"),
    AIC = c(AIC(cox_full), best_aic, AIC(aft_lnorm), AIC(aft_weibull)),
    LogLik = c(cox_full$loglik[2], flex_best$loglik, aft_lnorm$loglik[2], aft_weibull$loglik[2]),
    stringsAsFactors = FALSE
  )
} else {
  # Create comparison without Cox model
  overall_comparison <- data.frame(
    Model = c("Flexible Parametric", "Lognormal AFT", "Weibull AFT"),
    AIC = c(best_aic, AIC(aft_lnorm), AIC(aft_weibull)),
    LogLik = c(flex_best$loglik, aft_lnorm$loglik[2], aft_weibull$loglik[2]),
    stringsAsFactors = FALSE
  )
}

# Sort by AIC (lower is better)
overall_comparison <- overall_comparison[order(overall_comparison$AIC), ]

# Save overall comparison
write.csv(overall_comparison, "output/overall_model_comparison.csv", row.names = FALSE)

message("Parametric modeling complete.")

