# 01_data_preprocessing.R
# COVID-19 Survival Analysis - Data Preprocessing 
# This script loads and preprocesses the COVID-19 data

# Load required packages
if (!require("pacman")) install.packages("pacman")
pacman::p_load(dplyr, tidyverse, here)

# Create directories if they don't exist
dir.create("data", showWarnings = FALSE)
dir.create("output", showWarnings = FALSE)
dir.create("figures", showWarnings = FALSE)

# Set working directory using here package for better reproducibility
# Uncomment and modify as needed:
setwd(here())

# Read and preprocess COVID-19 data
read_and_preprocess_data <- function(file_path = here("data/covid_sample_data.txt")) {
  # Check if file exists
  if (!file.exists(file_path)) {
    stop(paste("Data file not found:", file_path))
  }
  
  # Read the data
  message("Reading data file...")
  Covid_data <- read.csv(file_path, sep = ";") %>% 
    dplyr::mutate(
      # Convert factors
      wave = as.factor(wave),
      gender = as.factor(gender),
      hosp_id = as.factor(hosp_id),
      
      # Create age groups
      age_gp = factor(
        case_when(
          age <= 40 ~ "Young Adults",
          age <= 60 ~ "Middle-aged Adults",
          TRUE ~ "Elderly"
        ),
        levels = c("Young Adults", "Middle-aged Adults", "Elderly")
      ),
      
      # Create CT value groups
      ct_gp = factor(
        case_when(
          Ct <= 24 ~ "Strongly positive",
          Ct <= 30 ~ "Moderately Positive",
          TRUE ~ "Weakly Positive"
        ),
        levels = c("Strongly positive", "Moderately Positive", "Weakly Positive")
      ),
      
      # Create length of stay groups
      los_gp = factor(
        case_when(
          LoS <= 7 ~ "One Week or Less",
          TRUE ~ "Over one weeks"
        )
      ),
      
      # Create sex variable from gender codes
      sex = factor(ifelse(gender == 1, "Male", "Female"), levels = c("Male", "Female")),
      patient_id = as.factor(patient_id),
      
      # Create age group bins for visualization (5-year bands)
      age_group = ifelse(age >= 100, "100-105", paste0((age %/% 5) * 5, "-", ((age %/% 5) * 5 + 4)))
    )
  
  message("Data preprocessing complete.")
  return(Covid_data)
}

# Calculate counts for age pyramid visualization
calculate_age_counts <- function(data) {
  age_counts <- data %>%
    count(age_group, sex) %>%
    group_by(age_group) %>%
    mutate(percentage = n / sum(n) * 100) %>% 
    ungroup() %>%
    mutate(age_group = factor(age_group, levels = c(paste0((0:19) * 5, "-", (0:19) * 5 + 4), "100-105"))) %>%
    mutate(n2 = ifelse(sex == "Male", -n, n)) %>%
    mutate(percentage2 = case_when(
      sex == "Male" ~ -percentage,
      TRUE ~ percentage
    ))
  
  return(age_counts)
}

# Calculate COVID deaths for mortality pyramid
calculate_covid_deaths <- function(data) {
  covid_deaths <- data %>%
    dplyr::filter(mort_hospital == 1) %>% 
    count(age_group, sex) %>%
    group_by(age_group) %>%
    mutate(percentage = n / sum(n) * 100) %>% 
    ungroup() %>%
    mutate(age_group = factor(age_group, levels = c(paste0((0:19) * 5, "-", (0:19) * 5 + 4), "100-105"))) %>%
    mutate(n2 = ifelse(sex == "Male", -n, n)) %>%
    mutate(percentage2 = case_when(
      sex == "Male" ~ -percentage,
      TRUE ~ percentage
    ))
  
  return(covid_deaths)
}

# Main processing execution
message("Starting data preprocessing...")

# Process data
Covid_data <- read_and_preprocess_data("data/covid_sample_data.txt")
age_counts <- calculate_age_counts(Covid_data)
covid_deaths <- calculate_covid_deaths(Covid_data)

# Create a log-transformed version of length of stay data
Covid_data$log_LoS <- log(Covid_data$LoS)

# Calculate summary statistics
age_summary <- summary(Covid_data$age)
los_summary <- summary(Covid_data$LoS)
iqr_age <- IQR(Covid_data$age, na.rm = T)
iqr_los <- IQR(Covid_data$LoS, na.rm = T)

# Save processed data
message("Saving processed data...")
save(Covid_data, age_counts, covid_deaths, 
     age_summary, los_summary, iqr_age, iqr_los,
     file = "data/processed_covid_data.RData")

# Print confirmation
message("Data preprocessing complete. Processed data saved to 'data/processed_covid_data.RData'")
message("Summary statistics:")
print(age_summary)
print(los_summary)

