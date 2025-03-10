# Helper functions for COVID-19 survival analysis
# Extract functions from SurvivalTutorial2023.Rmd

# Function to categorize age groups
categorize_age_groups <- function(age) {
  case_when(
    age <= 40 ~ "Young Adults",
    age <= 60 ~ "Middle-aged Adults",
    TRUE ~ "Elderly"
  )
}

# Function to categorize CT values
categorize_ct_values <- function(ct) {
  case_when(
    ct <= 24 ~ "Strongly positive",
    ct <= 30 ~ "Moderately Positive",
    TRUE ~ "Weakly Positive"
  )
}

# Function to categorize length of stay
categorize_los <- function(los) {
  case_when(
    los <= 7 ~ "One Week or Less",
    TRUE ~ "Over one weeks"
  )
}

# Function to create age brackets for population pyramid
create_age_brackets <- function(age) {
  ifelse(age >= 100, "100-105", paste0((age %/% 5) * 5, "-", ((age %/% 5) * 5 + 4)))
}

# Function to generate statistical summary for variables
get_variable_summary <- function(data, var) {
  summary_stats <- summary(data[[var]])
  iqr_value <- IQR(data[[var]])
  
  return(list(
    summary = summary_stats,
    iqr = iqr_value
  ))
}

# Function to create KM survival plot with consistent styling
create_km_plot <- function(surv_object, data, title, legend_labels = NULL) {
  ggsurvplot(
    surv_object, 
    data = data,
    palette = c("#0073C2FF", "#EFC000FF", "#FF0000"), 
    pval = TRUE,
    conf.int = TRUE,
    pval.method = TRUE,
    test.for.trend = FALSE,
    surv.median.line = "hv",
    risk.table = TRUE,
    censor.size = 0.8,
    font.main = 10,
    log.rank.weights = "sqrtN",
    pval.method.size = 3,
    pval.size = 3,
    legend.title = "",
    fontsize = 3,
    conf.int.style = "ribbon",
    risk.table.title = "",
    tables.height = 0.33,
    tables.font = 2,
    risk.table.fontsize = 3,
    risk.table.y.text.col = FALSE,
    pval.method.coord = c(20, 0.8),
    pval.coord = c(24, 0.8),
    legend.labs = legend_labels,
    xlab = "Survival time in days",
    ylab = "Survival Probability",
    title = title,
    ggtheme = theme_classic2(),
    risk.table.title.fontsize = 2
  )
}

# Model selection and evaluation function
compare_survival_models <- function(data, outcome_var, covariates) {
  # Create formula for Cox model
  cox_formula <- as.formula(paste("Surv(LoS, ", outcome_var, ") ~", paste(covariates, collapse = " + ")))
  
  # Fit Cox model
  cox_model <- coxph(cox_formula, data = data)
  
  # Create formula for AFT models
  aft_formula <- as.formula(paste("Surv(LoS, ", outcome_var, ") ~", paste(covariates, collapse = " + ")))
  
  # Fit lognormal AFT model
  lnorm_model <- survreg(aft_formula, data = data, dist = "lognormal")
  
  # Fit Weibull AFT model
  weibull_model <- survreg(aft_formula, data = data, dist = "weibull")
  
  # Calculate AIC for each model
  cox_aic <- AIC(cox_model)
  lnorm_aic <- AIC(lnorm_model)
  weibull_aic <- AIC(weibull_model)
  
  # Return model comparison
  return(list(
    cox_model = cox_model,
    lnorm_model = lnorm_model,
    weibull_model = weibull_model,
    aic_values = data.frame(
      Model = c("Cox PH", "Lognormal AFT", "Weibull AFT"),
      AIC = c(cox_aic, lnorm_aic, weibull_aic)
    )
  ))
}
