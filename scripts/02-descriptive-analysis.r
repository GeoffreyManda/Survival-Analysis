# 02_descriptive_analysis.R
# COVID-19 Survival Analysis - Descriptive Statistics and Visualizations
# This script generates descriptive statistics and basic visualizations

# Load required packages
if (!require("pacman")) install.packages("pacman")
pacman::p_load(dplyr, ggplot2, gmodels, tidyverse, here)

# Load preprocessed data
if (!file.exists("data/processed_covid_data.RData")) {
  stop("Processed data file not found. Run 01_data_preprocessing.R first.")
}

load(here("data/processed_covid_data.RData"))

# Create output directory for figures
dir.create("figures", showWarnings = FALSE)

# 1. Age Pyramid Visualization
message("Creating age pyramid visualization...")
age_pyramid_plot <- ggplot(age_counts, aes(x = age_group, y = n2, fill = sex)) +
  # Create the pyramid structure with bars
  geom_bar(stat = "identity", position = "identity") +
  
  # Add count labels
  geom_text(aes(label = abs(n2)), vjust = 0.5,
            hjust = case_when(
              age_counts$age_group %in% c("95-99", "100-105") & age_counts$sex == "Male" ~ 0.1,
              age_counts$age_group %in% c("95-99", "100-105") & age_counts$sex == "Female" ~ 1,
              
              age_counts$age_group %in% c("20-24") & age_counts$sex == "Male" ~ -1.2,
              age_counts$age_group %in% c("20-24") & age_counts$sex == "Female" ~ 1.6,
              
              age_counts$age_group %in% c( "25-29", "30-34", "35-39", "40-44") & age_counts$sex == "Male" ~ -1,
              age_counts$age_group %in% c( "25-29", "30-34", "35-39", "40-44") & age_counts$sex == "Female" ~ 2,
              
              age_counts$age_group %in% c( "70-74","85-89","50-54","60-64","75-79","65-69","45-49",
                                           "55-59", "90-94","80-84") & age_counts$sex == "Male" ~ -1.2,
              age_counts$age_group %in% c( "70-74","85-89","50-54","60-64","75-79","65-69","45-49",
                                           "55-59", "90-94","80-84") & age_counts$sex == "Female" ~ 1.2,
            ), size = 1.6) +
  
  # Add percentage labels
  geom_text(aes(label = paste0("(", sprintf("%.0f%%", abs(percentage2)), ")")), 
            hjust = case_when(
              age_counts$age_group %in% c("95-99", "100-105") & age_counts$sex == "Male" ~ 1.2,
              age_counts$age_group %in% c("95-99", "100-105") & age_counts$sex == "Female" ~ -0.1,
              
              age_counts$age_group %in% c("20-24") & age_counts$sex == "Male" ~ 1.2,
              age_counts$age_group %in% c("20-24") & age_counts$sex == "Female" ~ -0.3,
              
              age_counts$age_group %in% c("25-29", "30-34", "35-39", "40-44") & age_counts$sex == "Male" ~ 1.1,
              age_counts$age_group %in% c("25-29", "30-34", "35-39", "40-44") & age_counts$sex == "Female" ~ -0.2,
              
              age_counts$age_group %in% c( "70-74","85-89","50-54","60-64","75-79","65-69","45-49",
                                           "55-59", "90-94","80-84") & age_counts$sex == "Male" ~ 1.2,
              age_counts$age_group %in% c( "70-74","85-89","50-54","60-64","75-79","65-69","45-49",
                                           "55-59", "90-94","80-84") & age_counts$sex == "Female" ~ -0.2,), size = 1.6) +
  
  # Set colors and labels
  scale_fill_manual(values = c("#0073C2FF", "#EFC000FF"), labels = c("Male", "Female")) +
  
  # Add titles and theme
  labs(x = "Age Group", y = "") +
  ggtitle("Age pyramid of all Coronavirus disease 2019 cases in the analysis") +
  theme_classic() +
  theme(legend.position = "bottom",
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        panel.border = element_rect(color = "black", fill = NA, linewidth = 1)) +
  coord_flip()
# Visualise pyramid plot
age_pyramid_plot
# Save plot
ggsave("figures/age_pyramid.png", age_pyramid_plot, width = 12, height = 8, dpi = 300)

# 2. COVID Deaths Visualization
message("Creating COVID deaths visualization...")
covid_deaths_plot <- ggplot(covid_deaths, aes(x = age_group, y = n, fill = sex)) +
  geom_bar(stat = "identity", position = "fill") +
  scale_fill_manual(values = c("#EFC000FF", "#0073C2FF")) +
  labs(x = "Age Group", y = "Percentage of deaths", fill = "Gender") +
  ggtitle("Composition Plot of COVID-19 deaths by age group and by Gender") +
  theme_minimal() +
  theme(legend.position = "bottom", axis.text.x = element_text(angle = 45, hjust = 1))
# Visualise covid_deaths_plot
covid_deaths_plot
# Save plot
ggsave("figures/covid_deaths_composition.png", covid_deaths_plot, width = 10, height = 6, dpi = 300)

# 3. Length of Stay by Wave Visualization
message("Creating length of stay by wave visualization...")
# Define custom colors for the waves
my_colors <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442")

los_wave_plot <- ggplot(Covid_data, aes(x = wave, y = LoS, fill = wave)) +
  # Create violin plots showing the distribution
  geom_violin(trim = FALSE, scale = "width", alpha = 0.8) +
  # Add box plots inside the violins to show quartiles
  geom_boxplot(width = 0.15, fill = "white", color = "#757575") +
  # Set custom colors
  scale_fill_manual(values = my_colors) +
  # Add labels
  labs(x = "Wave", y = "Length of hospital Stay (Days)", fill = "Wave") +
  ggtitle("Violin plot with Box plot of Length of Stay by Wave") +
  # Set theme
  theme_classic() +
  theme(plot.title = element_text(size = 16, face = "bold"),
        axis.title = element_text(size = 14),
        axis.text = element_text(size = 12),
        legend.title = element_text(size = 12),
        legend.text = element_text(size = 10),
        panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
        legend.position = "bottom")
# Visualise los_wave_plot
los_wave_plot
# Save plot
ggsave("figures/los_by_wave.png", los_wave_plot, width = 10, height = 7, dpi = 300)

# 4. Distribution Histograms
message("Creating distribution histograms...")
# Function to create and save histograms
create_histogram <- function(data, title, xlab, filename, breaks = 30, color = "lightblue") {
  # Create directory if it doesn't exist
  if (!dir.exists("figures")) {
    dir.create("figures")
  }
  
  png(paste0("figures/", filename), width = 800, height = 500, res = 100)
  hist(data, 
       breaks = breaks, 
       main = title, 
       xlab = xlab,
       col = color,
       border = "white")
  dev.off()
}

# Create histograms for key variables
create_histogram(Covid_data$age, "Distribution of Patient Age", "Age in Years", 
                "hist_age.png", 30, "lightblue")
create_histogram(Covid_data$Ct, "Distribution of Cycle Threshold Values", "Ct Value in thermal cycles", 
                "hist_ct.png", 30, "lightgreen")
create_histogram(Covid_data$LoS, "Distribution of Hospital Length of Stay", "Length of stay in days", 
                "hist_los.png", 30, "lightsalmon")
create_histogram(log(Covid_data$LoS), "Distribution of Log-Transformed Length of Stay", "Log Length of stay in days", 
                "hist_log_los.png", 30, "plum")

# 5. Cross Tabulations
message("Conducting cross-tabulation analyses...")

# Function to create and save cross-tabulation results
perform_crosstab <- function(var1, var2, title, filename) {
  # Create sink to capture the output
  sink(paste0("output/", filename))
  cat(title, "\n\n")
  print(CrossTable(var1, var2, 
           prop.chisq = FALSE, 
           prop.t = TRUE, 
           prop.r = TRUE, 
           chisq = TRUE, 
           format = "SPSS"))
  sink()
}

# Create output directory
dir.create("output", showWarnings = FALSE)

# Perform cross-tabulations
perform_crosstab(Covid_data$sex, Covid_data$mort_hospital, 
                "Mortality by Gender", "crosstab_gender_mortality.txt")
perform_crosstab(Covid_data$ct_gp, Covid_data$mort_hospital, 
                "Mortality by CT Group", "crosstab_ct_mortality.txt")
perform_crosstab(Covid_data$age_gp, Covid_data$mort_hospital, 
                "Mortality by Age Group", "crosstab_age_mortality.txt")
perform_crosstab(Covid_data$wave, Covid_data$mort_hospital, 
                "Mortality by Wave", "crosstab_wave_mortality.txt")
perform_crosstab(Covid_data$ct_gp, Covid_data$sex, 
                "CT Group by Gender", "crosstab_ct_gender.txt")
perform_crosstab(Covid_data$los_gp, Covid_data$mort_hospital, 
                "Mortality by Length of Stay Group", "crosstab_los_mortality.txt")

# 6. Length of Stay by Gender and Wave for Recovered Patients
message("Analyzing length of stay for recovered patients...")
# Filter for recovered patients
recovered_data <- Covid_data %>%
  filter(mort_hospital == 1)

# Create a box plot
los_recovered_plot <- ggplot(recovered_data, aes(x = wave, y = LoS, fill = gender)) +
  geom_boxplot() +
  scale_fill_manual(values = c("#0073C2FF", "#EFC000FF"), labels = c("Female", "Male")) +  
  # Custom fill colors and labels
  xlab("Wave (Period 1, Period 2, Period 3)") +
  ylab("Length of Hospital Stay (Days)") +
  ggtitle("Length of Stay in Hospital for Recovered Patients by Wave and Gender") +
  theme_minimal() +
  theme(legend.position = "top",
        plot.title = element_text(size = 14),
        axis.title = element_text(size = 12))

# Save plot
ggsave("figures/los_recovered_by_wave_gender.png", los_recovered_plot, width = 10, height = 6, dpi = 300)

# 7. Summary Statistics 
message("Generating summary statistics tables...")
# Create summary statistics table
summary_stats <- data.frame(
  Variable = c("Age", "Length of Stay", "Cycle Threshold (Ct)"),
  Min = c(min(Covid_data$age, na.rm = TRUE), 
          min(Covid_data$LoS, na.rm = TRUE), 
          min(Covid_data$Ct, na.rm = TRUE)),
  Q1 = c(quantile(Covid_data$age, 0.25, na.rm = TRUE), 
         quantile(Covid_data$LoS, 0.25, na.rm = TRUE), 
         quantile(Covid_data$Ct, 0.25, na.rm = TRUE)),
  Median = c(median(Covid_data$age, na.rm = TRUE), 
             median(Covid_data$LoS, na.rm = TRUE), 
             median(Covid_data$Ct, na.rm = TRUE)),
  Mean = c(mean(Covid_data$age, na.rm = TRUE), 
           mean(Covid_data$LoS, na.rm = TRUE), 
           mean(Covid_data$Ct, na.rm = TRUE)),
  Q3 = c(quantile(Covid_data$age, 0.75, na.rm = TRUE), 
         quantile(Covid_data$LoS, 0.75, na.rm = TRUE), 
         quantile(Covid_data$Ct, 0.75, na.rm = TRUE)),
  Max = c(max(Covid_data$age, na.rm = TRUE), 
          max(Covid_data$LoS, na.rm = TRUE), 
          max(Covid_data$Ct, na.rm = TRUE)),
  IQR = c(IQR(Covid_data$age, na.rm = TRUE), 
          IQR(Covid_data$LoS, na.rm = TRUE), 
          IQR(Covid_data$Ct, na.rm = TRUE)),
  SD = c(sd(Covid_data$age, na.rm = TRUE), 
         sd(Covid_data$LoS, na.rm = TRUE), 
         sd(Covid_data$Ct, na.rm = TRUE))
)

# Round numeric columns to 2 decimal places
summary_stats[, 2:9] <- round(summary_stats[, 2:9], 2)

# Save summary statistics
write.csv(summary_stats, "output/summary_statistics.csv", row.names = FALSE)

# 8. Normality Tests
message("Performing normality tests...")
# Perform Shapiro-Wilk test for normality
grouped_data <- Covid_data %>%
  group_by(wave) %>%
  summarise(
    p_value_los = shapiro.test(LoS)$p.value,
    p_value_log_los = shapiro.test(log(LoS))$p.value
  )

# Save normality test results
write.csv(grouped_data, "output/normality_test_results.csv", row.names = FALSE)

message("Descriptive analysis complete. Results saved to 'figures/' and 'output/' directories.")

