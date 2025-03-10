# 00_run_analysis.R
# COVID-19 Survival Analysis - Main Runner Script
# This script runs all analysis steps in sequence

# Set working directory to project root using here package
setwd(here::here())

# Optional: Verify working directory
# print(getwd())

# Record the start time
start_time <- Sys.time()

# Create log directory if it doesn't exist
dir.create("logs", showWarnings = FALSE)

# Start log file
log_file <- file.path("logs", paste0("analysis_log_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".txt"))
sink(log_file, split = TRUE)

cat("COVID-19 Survival Analysis\n")
cat("=========================\n")
cat("Started at:", format(start_time), "\n\n")

# Check for required packages and install if missing
required_packages <- c(
  "dplyr", "survminer", "survival", "rio", "car", "gtsummary", "flexsurv", 
  "ggplot2", "tidyverse", "coxme", "lme4", "MASS", "rstanarm", "ranger", 
  "randomForestSRC", "gmodels", "patchwork", "viridis", "gridExtra", 
  "RColorBrewer", "sjPlot", "performance", "pacman", "here"
)

cat("Checking and installing required packages...\n")
missing_packages <- required_packages[!required_packages %in% installed.packages()[, "Package"]]
if (length(missing_packages) > 0) {
  cat("Installing missing packages:", paste(missing_packages, collapse = ", "), "\n")
  install.packages(missing_packages, repos = "https://cloud.r-project.org")
}

# Load pacman for package management
if (!require("pacman")) install.packages("pacman", repos = "https://cloud.r-project.org")
library(pacman)

# Verify that the scripts sub-folder exists
if (!dir.exists("scripts")) {
  stop("Scripts sub-folder 'scripts' not found! Please ensure your analysis scripts are in the 'scripts' folder.")
}

# Function to run a script and handle errors
run_script <- function(script_name, script_description) {
  cat("\n")
  cat("============================================\n")
  cat("Running:", script_name, "\n")
  cat(script_description, "\n")
  cat("============================================\n")
  
  script_start <- Sys.time()
  
  result <- tryCatch({
    source(script_name)
    return(TRUE)
  }, error = function(e) {
    cat("\nERROR in", script_name, ":\n")
    cat(e$message, "\n")
    return(FALSE)
  }, warning = function(w) {
    cat("\nWARNING in", script_name, ":\n")
    cat(w$message, "\n")
    return(TRUE)
  })
  
  script_end <- Sys.time()
  elapsed <- script_end - script_start
  
  if (result) {
    cat("\nCompleted:", script_name, "\n")
    cat("Time taken:", format(elapsed), "\n")
  } else {
    cat("\nFailed to complete:", script_name, "\n")
  }
  
  return(result)
}

# Execute all scripts in sequence (using the 'scripts' folder)
scripts <- list(
  list(
    name = "scripts/01-data-preprocessing.r",
    description = "Data loading and preprocessing"
  ),
  list(
    name = "scripts/02-descriptive-analysis.r",
    description = "Descriptive statistics and basic visualizations"
  ),
  list(
    name = "scripts/03-survival-analysis.r",
    description = "Kaplan-Meier survival analysis and Cox models"
  ),
  list(
    name = "scripts/04-parametric-models.r",
    description = "Parametric survival models"
  ),
  list(
    name = "scripts/05-mixed-effects-models.r",
    description = "Mixed-effects models for hospital clustering"
  ),
  list(
    name = "scripts/06-visualization.r",
    description = "Advanced visualizations for publication"
  )
)
scripts
# Run each script and collect results
results <- list()
for (script in scripts) {
  results[[script$name]] <- run_script(script$name, script$description)
}

# Check if all scripts completed successfully
all_success <- all(unlist(results))

# Record the end time
end_time <- Sys.time()
total_elapsed <- end_time - start_time

cat("\n")
cat("=========================\n")
cat("Analysis complete\n")
cat("Started at:", format(start_time), "\n")
cat("Ended at:", format(end_time), "\n")
cat("Total time:", format(total_elapsed), "\n")

if (all_success) {
  cat("All scripts completed successfully.\n")
} else {
  cat("Some scripts encountered errors. Check the log for details.\n")
  failed_scripts <- names(results)[!unlist(results)]
  cat("Failed scripts:", paste(failed_scripts, collapse = ", "), "\n")
}

cat("=========================\n")

# Close the log file
sink()

# Print completion message to console
cat("\nAnalysis complete. Log saved to", log_file, "\n")

# Save the final session info for reproducibility
session_info <- sessionInfo()
writeLines(capture.output(print(session_info)), "logs/session_info.txt")
