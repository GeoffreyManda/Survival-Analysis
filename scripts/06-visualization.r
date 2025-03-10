# 06_visualization.R
# COVID-19 Survival Analysis - Advanced Visualizations
# This script creates publication-quality visualizations from analysis results

# Load required packages
if (!require("pacman")) install.packages("pacman")
pacman::p_load(ggplot2, dplyr, tidyverse, patchwork, 
               lme4, viridis, gridExtra, RColorBrewer, 
               survival, survminer)

# Load preprocessed data
if (!file.exists("data/processed_covid_data.RData")) {
  stop("Processed data file not found. Run 01_data_preprocessing.R first.")
}

load("data/processed_covid_data.RData")

# Create output directories
dir.create("figures", showWarnings = FALSE)
dir.create("figures/publication", showWarnings = FALSE)

#-----------------------------------#
# 1. Combined Demographic Visualization #
#-----------------------------------#
message("Creating combined demographic visualization...")

# 1.1 Age distribution by gender
p1 <- ggplot(Covid_data, aes(x = age, fill = sex)) +
  geom_histogram(position = "identity", alpha = 0.6, bins = 30) +
  scale_fill_brewer(palette = "Set1") +
  labs(title = "Age Distribution by Gender",
       x = "Age (years)",
       y = "Count",
       fill = "Gender") +
  theme_minimal() +
  theme(legend.position = "bottom")
dev.new()
p1 
# 1.2 Age group distribution
p2 <- ggplot(Covid_data, aes(x = age_gp, fill = age_gp)) +
  geom_bar() +
  scale_fill_brewer(palette = "Blues") +
  labs(title = "Distribution by Age Group",
       x = "Age Group",
       y = "Count") +
  theme_minimal() +
  theme(legend.position = "none",
        axis.text.x = element_text(angle = 45, hjust = 1))
p2
# 1.3 Wave distribution
p3 <- ggplot(Covid_data, aes(x = wave, fill = wave)) +
  geom_bar() +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "Cases by COVID-19 Wave",
       x = "Wave",
       y = "Count") +
  theme_minimal() +
  theme(legend.position = "none")
p3
# 1.4 CT value distribution
p4 <- ggplot(Covid_data, aes(x = Ct, fill = ct_gp)) +
  geom_histogram(bins = 30) +
  scale_fill_brewer(palette = "YlOrRd") +
  labs(title = "CT Value Distribution",
       x = "CT Value",
       y = "Count",
       fill = "CT Group") +
  theme_minimal() +
  theme(legend.position = "bottom")
p4
# Combine the plots
combined_demo <- (p1 + p2) / (p3 + p4)
combined_demo <- combined_demo + 
  plot_annotation(
    title = "COVID-19 Patient Demographics",
    theme = theme(plot.title = element_text(size = 16, face = "bold", hjust = 0.5))
  )
combined_demo
# Save the combined plot
ggsave("figures/publication/demographics_combined.png", combined_demo, width = 12, height = 10, dpi = 300)

#-------------------------------#
# 2. Mortality Visualizations   #
#-------------------------------#
message("Creating mortality visualizations...")

# 2.1 Mortality by age group and gender
mortality_by_age_gender <- Covid_data %>%
  group_by(age_gp, sex) %>%
  summarize(
    total = n(),
    deaths = sum(mort_hospital == 0),  # Note: mort_hospital == 0 indicates death
    mortality_rate = deaths / total * 100,
    .groups = "drop"
  )

p_mort_age <- ggplot(mortality_by_age_gender, aes(x = age_gp, y = mortality_rate, fill = sex)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_brewer(palette = "Set1") +
  labs(title = "Mortality Rate by Age Group and Gender",
       x = "Age Group",
       y = "Mortality Rate (%)",
       fill = "Gender") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "bottom")
p_mort_age
# 2.2 Mortality by wave
mortality_by_wave <- Covid_data %>%
  group_by(wave) %>%
  summarize(
    total = n(),
    deaths = sum(mort_hospital == 0),
    mortality_rate = deaths / total * 100,
    .groups = "drop"
  )

p_mort_wave <- ggplot(mortality_by_wave, aes(x = wave, y = mortality_rate, fill = wave)) +
  geom_bar(stat = "identity") +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "Mortality Rate by COVID-19 Wave",
       x = "Wave",
       y = "Mortality Rate (%)") +
  theme_minimal() +
  theme(legend.position = "none")
p_mort_wave
# 2.3 Mortality by CT group
mortality_by_ct <- Covid_data %>%
  group_by(ct_gp) %>%
  summarize(
    total = n(),
    deaths = sum(mort_hospital == 0),
    mortality_rate = deaths / total * 100,
    .groups = "drop"
  )

p_mort_ct <- ggplot(mortality_by_ct, aes(x = ct_gp, y = mortality_rate, fill = ct_gp)) +
  geom_bar(stat = "identity") +
  scale_fill_brewer(palette = "YlOrRd") +
  labs(title = "Mortality Rate by CT Group",
       x = "CT Group",
       y = "Mortality Rate (%)") +
  theme_minimal() +
  theme(legend.position = "none",
        axis.text.x = element_text(angle = 45, hjust = 1))
p_mort_ct
# Combine the mortality plots
combined_mort <- p_mort_age + (p_mort_wave / p_mort_ct) +
  plot_layout(widths = c(2, 1)) +
  plot_annotation(
    title = "COVID-19 Mortality Patterns",
    theme = theme(plot.title = element_text(size = 16, face = "bold", hjust = 0.5))
  )

# Save the combined mortality plot
ggsave("figures/publication/mortality_combined.png", combined_mort, width = 12, height = 8, dpi = 300)

#-------------------------------#
# 3. Length of Stay Visualizations #
#-------------------------------#
message("Creating length of stay visualizations...")

# 3.1 Length of stay distribution and log-transformation
p_los_dist <- ggplot(Covid_data, aes(x = LoS)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "black", alpha = 0.7) +
  labs(title = "Distribution of Length of Stay",
       x = "Length of Stay (days)",
       y = "Count") +
  theme_minimal()
p_los_dist 
p_log_los_dist <- ggplot(Covid_data, aes(x = log(LoS))) +
  geom_histogram(bins = 30, fill = "forestgreen", color = "black", alpha = 0.7) +
  labs(title = "Distribution of Log-Transformed Length of Stay",
       x = "Log Length of Stay",
       y = "Count") +
  theme_minimal()
p_log_los_dist
# 3.2 LoS by age group and gender
p_los_age_gender <- ggplot(Covid_data, aes(x = age_gp, y = LoS, fill = sex)) +
  geom_boxplot(alpha = 0.7) +
  scale_fill_brewer(palette = "Set1") +
  labs(title = "Length of Stay by Age Group and Gender",
       x = "Age Group",
       y = "Length of Stay (days)",
       fill = "Gender") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "bottom")
p_los_age_gender
# 3.3 LoS by wave with violin plot
p_los_wave <- ggplot(Covid_data, aes(x = wave, y = LoS, fill = wave)) +
  geom_violin(trim = FALSE, alpha = 0.7) +
  geom_boxplot(width = 0.2, alpha = 0.9) +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "Length of Stay by COVID-19 Wave",
       x = "Wave",
       y = "Length of Stay (days)") +
  theme_minimal() +
  theme(legend.position = "none")
p_los_wave
# Combine the LoS plots
combined_los <- (p_los_dist + p_log_los_dist) / (p_los_age_gender + p_los_wave) +
  plot_annotation(
    title = "COVID-19 Length of Stay Analysis",
    theme = theme(plot.title = element_text(size = 16, face = "bold", hjust = 0.5))
  )
combined_los
# Save the combined LoS plot
ggsave("figures/publication/los_combined.png", combined_los, width = 12, height = 10, dpi = 300)

#----------------------------------#
# 4. Survival Analysis Visualizations #
#----------------------------------#
message("Creating enhanced survival analysis visualizations...")

# 4.1 Overall survival with custom theme
km_overall <- survfit(Surv(LoS, mort_hospital==1) ~ 1, data = Covid_data)
p_surv_overall <- ggsurvplot(
  km_overall,
  data = Covid_data,
  palette = "darkblue",
  conf.int = TRUE,
  conf.int.fill = "lightblue",
  conf.int.alpha = 0.3,
  surv.median.line = "hv",
  risk.table = TRUE,
  risk.table.height = 0.25,
  xlab = "Time (days)",
  ylab = "Survival Probability",
  title = "Overall Survival",
  ggtheme = theme_minimal() + theme(
    plot.title = element_text(face = "bold", size = 14),
    axis.title = element_text(face = "bold"),
    legend.position = "none"
  )
)

# 4.2 Survival by gender with enhanced appearance
km_gender <- survfit(Surv(LoS, mort_hospital==1) ~ sex, data = Covid_data)
p_surv_gender <- ggsurvplot(
  km_gender,
  data = Covid_data,
  palette = c("#E7298A", "#1B9E77"),
  conf.int = TRUE,
  conf.int.alpha = 0.2,
  pval = TRUE,
  pval.coord = c(20, 0.1),
  surv.median.line = "hv",
  risk.table = TRUE,
  risk.table.height = 0.25,
  risk.table.col = "strata",
  xlab = "Time (days)",
  ylab = "Survival Probability",
  title = "Survival by Gender",
  legend.labs = c("Male", "Female"),
  legend.title = "Gender",
  ggtheme = theme_minimal() + theme(
    plot.title = element_text(face = "bold", size = 14),
    axis.title = element_text(face = "bold"),
    legend.position = "bottom"
  )
)

# 4.3 Survival by age group with enhanced appearance
km_age <- survfit(Surv(LoS, mort_hospital==1) ~ age_gp, data = Covid_data)
p_surv_age <- ggsurvplot(
  km_age,
  data = Covid_data,
  palette = brewer.pal(3, "Blues"),
  conf.int = TRUE,
  conf.int.alpha = 0.1,
  pval = TRUE,
  pval.coord = c(20, 0.1),
  surv.median.line = "hv",
  risk.table = TRUE,
  risk.table.height = 0.25,
  risk.table.col = "strata",
  xlab = "Time (days)",
  ylab = "Survival Probability",
  title = "Survival by Age Group",
  legend.labs = c("Elderly", "Middle-aged Adults", "Young Adults"),
  legend.title = "Age Group",
  ggtheme = theme_minimal() + theme(
    plot.title = element_text(face = "bold", size = 14),
    axis.title = element_text(face = "bold"),
    legend.position = "bottom"
  )
)

# 4.4 Survival by wave with enhanced appearance
km_wave <- survfit(Surv(LoS, mort_hospital==1) ~ wave, data = Covid_data)
p_surv_wave <- ggsurvplot(
  km_wave,
  data = Covid_data,
  palette = brewer.pal(3, "Set2"),
  conf.int = TRUE,
  conf.int.alpha = 0.1,
  pval = TRUE,
  pval.coord = c(20, 0.1),
  surv.median.line = "hv",
  risk.table = TRUE,
  risk.table.height = 0.25,
  risk.table.col = "strata",
  xlab = "Time (days)",
  ylab = "Survival Probability",
  title = "Survival by COVID-19 Wave",
  legend.labs = c("Wave 1", "Wave 2", "Wave 3"),
  legend.title = "Wave",
  ggtheme = theme_minimal() + theme(
    plot.title = element_text(face = "bold", size = 14),
    axis.title = element_text(face = "bold"),
    legend.position = "bottom"
  )
)

# Save enhanced survival plots
ggsave("figures/publication/survival_overall.png", p_surv_overall$plot, width = 8, height = 6, dpi = 300)
ggsave("figures/publication/survival_gender.png", p_surv_gender$plot, width = 8, height = 6, dpi = 300)
ggsave("figures/publication/survival_age.png", p_surv_age$plot, width = 8, height = 6, dpi = 300)
ggsave("figures/publication/survival_wave.png", p_surv_wave$plot, width = 8, height = 6, dpi = 300)

# Save versions with risk tables
ggsave("figures/publication/survival_overall_with_table.png", p_surv_overall, width = 8, height = 8, dpi = 300)
ggsave("figures/publication/survival_gender_with_table.png", p_surv_gender, width = 8, height = 8, dpi = 300)
ggsave("figures/publication/survival_age_with_table.png", p_surv_age, width = 8, height = 8, dpi = 300)
ggsave("figures/publication/survival_wave_with_table.png", p_surv_wave, width = 8, height = 8, dpi = 300)

#----------------------------------#
# 5. Model Comparison Visualizations #
#----------------------------------#
message("Creating model comparison visualizations...")

# Check if model comparison data exists
if (file.exists("output/overall_model_comparison.csv")) {
  # Load model comparison data
  model_comparison <- read.csv("output/overall_model_comparison.csv")
  
  # Create bar plot for AIC comparison
  p_model_aic <- ggplot(model_comparison, aes(x = reorder(Model, -AIC), y = AIC, fill = Model)) +
    geom_bar(stat = "identity") +
    scale_fill_brewer(palette = "Set3") +
    labs(title = "Model Comparison by AIC",
         x = "Model",
         y = "AIC (lower is better)") +
    theme_minimal() +
    theme(legend.position = "none",
          axis.text.x = element_text(angle = 45, hjust = 1),
          plot.title = element_text(face = "bold", size = 14))
  
  # Save model comparison plot
  ggsave("figures/publication/model_comparison.png", p_model_aic, width = 8, height = 6, dpi = 300)
}
p_model_aic
#----------------------------------#
# 6. Hospital Effect Visualizations #
#----------------------------------#
message("Creating hospital effect visualizations...")

# Check if mixed-effects model results exist
if (file.exists("models/lmm_length_of_stay.rds")) {
  # Load mixed-effects model
  lmm_los <- readRDS("models/lmm_length_of_stay.rds")
  
  # Extract random effects (hospital effects)
  re <- ranef(lmm_los)$hosp_id
  re$hosp_id <- rownames(re)
  
  # Create caterpillar plot of hospital effects
  p_hosp_effect <- ggplot(re, aes(x = reorder(hosp_id, `(Intercept)`), y = `(Intercept)`)) +
    geom_point(size = 3, color = "steelblue") +
    geom_errorbar(aes(ymin = `(Intercept)` - 1.96*attr(re, "postVar")[1], 
                      ymax = `(Intercept)` + 1.96*attr(re, "postVar")[1]), 
                  width = 0.2) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
    labs(title = "Hospital Random Effects on Length of Stay",
         subtitle = "With 95% confidence intervals",
         x = "Hospital",
         y = "Random Effect Estimate") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
          plot.title = element_text(face = "bold", size = 14))
  
  # Save hospital effect plot
  ggsave("figures/publication/hospital_effects.png", p_hosp_effect, width = 10, height = 6, dpi = 300)
}

p_hosp_effect
#----------------------------------#
# 7. Prediction Visualizations #
#----------------------------------#
message("Creating prediction visualizations...")

# Check if prediction data exists
if (file.exists("output/los_predictions.csv")) {
  # Load prediction data
  predictions <- read.csv("output/los_predictions.csv")
  
  # Convert factors
  predictions$gender <- factor(predictions$gender, levels = c(0, 1), labels = c("Female", "Male"))
  predictions$wave <- factor(predictions$wave)
  
  # Create interaction plot
  p_pred_age_wave <- ggplot(predictions, aes(x = age, y = predicted_los, color = wave, linetype = gender)) +
    geom_line(size = 1) +
    facet_wrap(~ Ct, labeller = labeller(Ct = function(x) paste("CT Value:", x))) +
    scale_color_brewer(palette = "Set2") +
    labs(title = "Predicted Length of Stay by Age, Gender, CT Value, and Wave",
         subtitle = "Based on mixed-effects model",
         x = "Age (years)",
         y = "Predicted Length of Stay (days)",
         color = "Wave",
         linetype = "Gender") +
    theme_minimal() +
    theme(strip.background = element_rect(fill = "lightyellow"),
          strip.text = element_text(face = "bold"),
          plot.title = element_text(face = "bold", size = 14),
          legend.position = "bottom")
  
  # Save prediction plot
  ggsave("figures/publication/los_predictions_enhanced.png", p_pred_age_wave, width = 12, height = 8, dpi = 300)
}

message("Enhanced visualizations complete.")

