# 03_survival_analysis.R
# COVID-19 Survival Analysis - Non-parametric and Semi-parametric Analysis
# This script performs Kaplan-Meier survival analysis and Cox proportional hazards modeling

# Load required packages
if (!require("pacman")) install.packages("pacman")
pacman::p_load(survival, survminer, dplyr, ggplot2, tidyverse, car, gridExtra)


# Load preprocessed data
if (!file.exists("data/processed_covid_data.RData")) {
  stop("Processed data file not found. Run 01_data_preprocessing.R first.")
}

load(here("data/processed_covid_data.RData"))

# Create output directories
dir.create("figures", showWarnings = FALSE)
dir.create("output", showWarnings = FALSE)
dir.create("models", showWarnings = FALSE)

#--------------------------#
# 1. Kaplan-Meier Analysis #
#--------------------------#
message("Performing Kaplan-Meier survival analysis...")

# Function to create and save KM plots
#source(here("scripts/functions/create_km_plot.r"))
create_km_plot <- function(surv_object, data, title, filename, legend_labs = NULL,
                           palette = c("#0073C2FF", "#EFC000FF", "#FF0000"),
                           output_dir = "figures", width = 10, height = 10) {
  
  # Ensure the output directory exists
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  # Determine number of groups in survival object
  num_groups <- if (!is.null(surv_object$strata)) length(surv_object$strata) else 1
  
  # Auto-coordinate for p-value in single-group scenarios
  pval_coord <- if(num_groups == 1) {
    c(max(data$LoS, na.rm = TRUE) * 0.8, 0.8)
  } else {
    NULL
  }
  
  # Enable p-value display only if comparing groups
  pval_flag <- (num_groups > 1)
  
  # Use ggsurvplot's built-in capabilities for proper alignment
  # This is more reliable than manually arranging plots
  km_plot <- ggsurvplot(
    surv_object, data = data,
    palette = palette,
    pval = pval_flag,
    conf.int = TRUE,
    pval.method = pval_flag,
    surv.median.line = "hv",
    risk.table = TRUE,
    censor.size = 0.8,
    font.main = 10,
    log.rank.weights = "1",
    pval.method.size = 3,
    pval.size = 3,
    legend.title = "",
    fontsize = 3,
    conf.int.style = "ribbon",
    pval.coord = pval_coord,
    legend.labs = legend_labs,
    xlab = "Survival time in days",
    ylab = "Survival Probability",
    title = title,
    tables.theme = theme_cleantable(),
    risk.table.height = 0.15,  # Control risk table height (25% of total)
    risk.table.title = "Number at Risk",
    tables.font = 2,
    risk.table.fontsize = 3,
    risk.table.y.text.col = FALSE,
    # Critical parameter for alignment:
    tables.y.text = TRUE,  # Remove strata names from risk table to save space
    ggtheme = theme_bw()    # Clean theme for better visibility
  )
  
  # Save the plot without risk table
  plot_filepath <- file.path(output_dir, filename)
  #ggsave(plot_filepath, km_plot$plot, width = width, height = height)
  
  # Save the combined plot using survminer's built-in arrangement
  # This ensures proper alignment of axes
  combined_filename <- gsub("\\.png$", "_with_table.png", filename)
  combined_filepath <- file.path(output_dir, combined_filename)
  
  # Use ggsurvplot's arrange method which handles alignment properly
  km_plot_arranged <- arrange_ggsurvplots(
    list(km_plot),
    print = FALSE,
    ncol = 1, nrow = 1
  )
  
  # Save the arranged plot
  ggsave(combined_filepath, km_plot_arranged, width = width, height = height)
  
  return(km_plot)
}

# 1.1 Overall survival
message("  Creating overall survival curve...")
km_surv <- survfit(Surv(LoS, mort_hospital==1) ~ 1, data = Covid_data)
km_overall <- create_km_plot(km_surv, Covid_data, 
                            "Overall Kaplan-Meier Survival Curve",
                            "km_overall.png")
km_overall 

# Save summary to file
sink("output/km_overall_summary.txt")
print(summary(km_surv))
sink()
# 1.2 Survival by gender
message("  Creating survival curves by gender...")
km_surv_gender <- survfit(Surv(LoS, mort_hospital==1) ~ gender, data = Covid_data)
km_gender <- create_km_plot(
  surv_object = km_surv_gender, 
  data = Covid_data, 
  title = "Kaplan-Meier Survival Curves Stratified by Gender",
  filename = "km_gender.png", 
  legend_labs = c("Females", "Males")
)
km_gender
# Save summary to file
sink("output/km_gender_summary.txt")
print(summary(km_surv_gender))
sink()

# 1.3 Survival by wave
message("  Creating survival curves by wave...")
km_surv_wave <- survfit(Surv(LoS, mort_hospital==1) ~ wave, data = Covid_data)
km_wave <- create_km_plot(km_surv_wave, Covid_data, 
                         "Kaplan-Meier Survival Curves Stratified by Wave",
                         "km_wave.png", 
                         legend_labs = c("Wave 1", "Wave 2", "Wave 3"))
km_wave
# Save summary to file
sink("output/km_wave_summary.txt")
print(summary(km_surv_wave))
sink()

# 1.4 Survival by age group
message("  Creating survival curves by age group...")
km_surv_agegp <- survfit(Surv(LoS, mort_hospital) ~ age_gp, data = Covid_data)
km_age <- create_km_plot(km_surv_agegp, Covid_data, 
                        "Kaplan-Meier Survival Curves Stratified by Age Group",
                        "km_age.png", 
                        legend_labs = c("Elderly", "Middle Age", "Young adults"))
km_age
# Save summary to file
sink("output/km_age_summary.txt")
print(summary(km_surv_agegp))
sink()

# 1.5 Survival by CT group
message("  Creating survival curves by CT group...")
km_surv_ct_gp <- survfit(Surv(LoS, mort_hospital==1) ~ ct_gp, data = Covid_data)
km_ct <- create_km_plot(km_surv_ct_gp, Covid_data, 
                       "Kaplan-Meier Survival Curves Stratified by Cycle Threshold Level",
                       "km_ct.png", 
                       legend_labs = c("Moderately Positive", "Strongly positive", "Weakly Positive"))
km_ct 
# Save summary to file
sink("output/km_ct_summary.txt")
print(summary(km_surv_ct_gp))
sink()

# 1.6 Survival by Hospital (Optional)
message("  Creating survival curves by hospital...")

# Fit the survival model stratified by hospital
km_surv_hosp <- survfit(Surv(LoS, mort_hospital == 1) ~ hosp_id, data = Covid_data)

# Define a color palette for the main plot only (risk table will use uniform black text)
hospital_palette <- c(
  "#0073C2FF", "#EFC000FF", "#FF0000", "#00AFBBFF", "#F8766DFF", "#C77CFF",
  "#7CAE00FF", "#D2D5D6FF", "#BF55ECFF", "#00BFC4FF", "#F58231FF", "#999999FF"
)

# Generate the Kaplan-Meier plot with a risk table
km_hosp_plot <- ggsurvplot(
  km_surv_hosp, 
  data = Covid_data,
  palette = hospital_palette,
  pval = TRUE,
  conf.int = TRUE,
  pval.method = TRUE,
  risk.table = TRUE,
  risk.table.col = "black",  # Use uniform black text in risk table
  risk.table.height = 0.25,
  censor.size = 0.8,
  font.main = 10,
  log.rank.weights = "1",
  pval.method.size = 3,
  pval.size = 3,
  legend.title = "",
  legend.labs = paste0("Hospital ", LETTERS[1:12]),
  xlab = "Survival Time (days)",
  ylab = "Survival Probability",
  title = "Kaplan-Meier Survival Curves Stratified by Hospital",
  # Auto-adjust p-value position based on max LoS
  pval.coord = c(max(Covid_data$LoS, na.rm = TRUE) * 0.8, 0.8),
  ggtheme = theme_bw(base_size = 12) + theme(legend.position = "bottom")
)

# Display the plot in the R session
print(km_hosp_plot)

# Arrange the survival plot and risk table into a combined object
combined_plot <- arrange_ggsurvplots(
  list(km_hosp_plot),
  print = FALSE,
  ncol = 1, nrow = 2  # Adjust rows and columns as needed
)


# Save the combined plot (includes both survival curve and risk table)
ggsave(filename = here("figures/km_hosp_plot_combined.png"), 
       plot = combined_plot, width = 10, height = 8)

#-------------------------#
# 2. Log-Rank Test Analysis #
#-------------------------#
message("Performing log-rank tests...")

# Function to perform and save log-rank test results
perform_logrank <- function(formula, data, title, filename) {
  # Perform the test
  logrank_result <- survdiff(formula, data = data)
  
  # Save results to file
  sink(paste0("output/", filename))
  cat(title, "\n\n")
  print(logrank_result)
  sink()
  
  return(logrank_result)
}

# Perform log-rank tests
logrank_gender <- perform_logrank(
  Surv(LoS, mort_hospital) ~ gender, Covid_data,
  "Log-rank test for gender differences", "logrank_gender.txt"
)

logrank_wave <- perform_logrank(
  Surv(LoS, mort_hospital) ~ wave, Covid_data,
  "Log-rank test for wave differences", "logrank_wave.txt"
)

logrank_ct_level <- perform_logrank(
  Surv(LoS, mort_hospital) ~ ct_gp, Covid_data,
  "Log-rank test for CT level differences", "logrank_ct_level.txt"
)

logrank_age_group <- perform_logrank(
  Surv(LoS, mort_hospital) ~ age_gp, Covid_data,
  "Log-rank test for age group differences", "logrank_age_group.txt"
)

#---------------------------------#
# 3. Cox Proportional Hazards Model #
#---------------------------------#
message("Fitting Cox proportional hazards models...")

# 3.1 Full Cox model with all covariates
cox_full <- coxph(Surv(LoS, mort_hospital==1) ~ sex + age_gp + ct_gp + wave, 
                 data = na.omit(Covid_data))

# Save model summary
sink("output/cox_full_model.txt")
print(summary(cox_full))
sink()

# Save model object for later use
saveRDS(cox_full, "models/cox_full_model.rds")

# 3.2 Check proportional hazards assumption
ph_test <- cox.zph(cox_full)
sink("output/cox_proportional_hazards_test.txt")
print(ph_test)
sink()

# Create PH test plot
png("figures/cox_ph_test.png", width = 900, height = 800)
par(mfrow = c(2, 2))
plot(ph_test)
dev.off()

# 3.3 Model selection using drop1 method
drop1_test <- drop1(cox_full, test = "Chisq")
sink("output/cox_model_selection.txt")
print(drop1_test)
sink()

# 3.4 Visualize Cox model results
# Forest plot of hazard ratios
ggforest_plot <- ggforest(cox_full, data = Covid_data)
ggsave("figures/cox_forest_plot.png", ggforest_plot, width = 10, height = 8)

message("Cox proportional hazards modeling complete.")

