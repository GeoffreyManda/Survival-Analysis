# 09_causal_representation_learning.R
# R Interface for Python Causal Representation Learning
# Calls Python scripts using reticulate package

# Load required packages
if (!require("pacman")) install.packages("pacman")
pacman::p_load(reticulate, here, dplyr, ggplot2)

#==============================================================================#
#                    SETUP PYTHON ENVIRONMENT                                  #
#==============================================================================#

message("Setting up Python environment for causal representation learning...")

# Check if conda environment exists
conda_envs <- reticulate::conda_list()
causal_env_exists <- "causal_rep" %in% conda_envs$name

if (!causal_env_exists) {
  message("Conda environment 'causal_rep' not found.")
  message("Please create it by running:")
  message("  cd scripts/python")
  message("  conda create -n causal_rep python=3.9")
  message("  conda activate causal_rep")
  message("  pip install -r requirements.txt")
  stop("Please set up Python environment first.")
}

# Use the causal_rep conda environment
use_condaenv("causal_rep", required = TRUE)

# Verify Python is configured
py_config()

message("Python environment configured successfully!")

#==============================================================================#
#                    RUN PREPROCESSING                                         #
#==============================================================================#

run_preprocessing <- function() {
  message("\n========================================")
  message("Running data preprocessing...")
  message("========================================\n")

  # Change to Python scripts directory
  original_dir <- getwd()
  setwd(here("scripts/python"))

  # Run preprocessing script
  result <- system("python 00_data_preprocessing.py", intern = TRUE)
  print(result)

  # Return to original directory
  setwd(original_dir)

  message("\nPreprocessing complete!")
}

#==============================================================================#
#                    RUN CEVAE TRAINING                                        #
#==============================================================================#

run_cevae <- function(epochs = 100, latent_dim = 10, learning_rate = 1e-3) {
  message("\n========================================")
  message("Training CEVAE model...")
  message("========================================\n")

  original_dir <- getwd()
  setwd(here("scripts/python"))

  # Run CEVAE training
  result <- system("python 01_cevae_survival.py", intern = TRUE)
  print(result)

  setwd(original_dir)

  message("\nCEVAE training complete!")

  # Load and return results
  load_cevae_results()
}

#==============================================================================#
#                    LOAD RESULTS                                              #
#==============================================================================#

load_cevae_results <- function() {
  message("\nLoading CEVAE results...")

  results_path <- here("output/causal_rep/cevae_effects.json")

  if (!file.exists(results_path)) {
    warning("CEVAE results not found. Please run training first.")
    return(NULL)
  }

  # Read JSON results
  results <- jsonlite::fromJSON(results_path)

  message("\nCEVAE Causal Effect Estimates:")
  message(sprintf("  ATE (Average Treatment Effect): %.4f", results$ATE))
  message(sprintf("  ATT (Effect on Treated):        %.4f", results$ATT))
  message(sprintf("  ATC (Effect on Controls):       %.4f", results$ATC))
  message(sprintf("  ITE std:                        %.4f", results$ITE_std))

  return(results)
}

#==============================================================================#
#                    VISUALIZE RESULTS                                         #
#==============================================================================#

plot_cevae_results <- function() {
  message("\nGenerating plots from CEVAE results...")

  # Check if figures exist
  training_curves <- here("figures/causal_rep/cevae_training_curves.png")
  ite_dist <- here("figures/causal_rep/cevae_ite_distribution.png")

  if (file.exists(training_curves)) {
    message("Training curves: ", training_curves)
  }

  if (file.exists(ite_dist)) {
    message("ITE distribution: ", ite_dist)
  }

  # If running in RStudio, display images
  if (interactive() && file.exists(ite_dist)) {
    img <- png::readPNG(ite_dist)
    grid::grid.raster(img)
  }
}

#==============================================================================#
#                    COMPARE WITH TRADITIONAL METHODS                          #
#==============================================================================#

compare_methods <- function() {
  message("\n========================================")
  message("Comparing Causal Representation Learning with Traditional Methods")
  message("========================================\n")

  # Load CEVAE results
  cevae_results <- load_cevae_results()

  if (is.null(cevae_results)) {
    return(NULL)
  }

  # Load traditional IPW results (from previous R analysis)
  ipw_results_path <- here("output/causal/viral_load_effect_comparison.csv")

  if (file.exists(ipw_results_path)) {
    ipw_results <- read.csv(ipw_results_path)

    message("Method Comparison:")
    message(sprintf("  IPW (Traditional):  HR = %.2f (95%% CI: %.2f-%.2f)",
                    ipw_results$HR[ipw_results$Model == "IPW (Causal ATE)"],
                    ipw_results$Lower_CI[ipw_results$Model == "IPW (Causal ATE)"],
                    ipw_results$Upper_CI[ipw_results$Model == "IPW (Causal ATE)"]))
    message(sprintf("  CEVAE (Deep Learning): Risk Diff = %.4f", cevae_results$ATE))
    message("\nNote: IPW estimates hazard ratio; CEVAE estimates risk difference.")
  }

  # Create comparison table
  comparison <- data.frame(
    Method = c("IPW (Traditional)", "CEVAE (Deep Learning)"),
    Handles_Unmeasured_Confounding = c("No", "Yes (with proxies)"),
    Non_Linear_Relationships = c("Manual", "Automatic"),
    Individual_Effects = c("No (ATE only)", "Yes (ITE)"),
    Uncertainty_Quantification = c("Bootstrap", "Bayesian"),
    stringsAsFactors = FALSE
  )

  print(comparison)

  return(comparison)
}

#==============================================================================#
#                    MAIN WORKFLOW                                             #
#==============================================================================#

run_full_pipeline <- function(skip_preprocessing = FALSE, skip_training = FALSE) {
  message("="*70)
  message("CAUSAL REPRESENTATION LEARNING PIPELINE")
  message("="*70)

  # Step 1: Preprocessing
  if (!skip_preprocessing) {
    run_preprocessing()
  } else {
    message("Skipping preprocessing (using existing processed data)")
  }

  # Step 2: Train CEVAE
  if (!skip_training) {
    run_cevae()
  } else {
    message("Skipping training (using existing model)")
  }

  # Step 3: Load and display results
  results <- load_cevae_results()

  # Step 4: Visualize
  plot_cevae_results()

  # Step 5: Compare with traditional methods
  compare_methods()

  message("\n" + "="*70)
  message("PIPELINE COMPLETE!")
  message("="*70)
  message("\nNext steps:")
  message("  1. Review results in output/causal_rep/")
  message("  2. Examine figures in figures/causal_rep/")
  message("  3. Compare with traditional IPW/g-formula estimates")
  message("  4. Interpret individual treatment effects (ITE)")
  message("  5. Identify subgroups with heterogeneous effects")

  invisible(results)
}

#==============================================================================#
#                    HELPER FUNCTIONS                                          #
#==============================================================================#

get_python_status <- function() {
  message("\nPython Environment Status:")
  message("  Python version: ", py_config()$version)
  message("  Python executable: ", py_config()$python)
  message("  Conda environment: ", reticulate::conda_list()$name[1])

  # Check if required packages are installed
  message("\nChecking Python packages...")

  required_packages <- c("torch", "numpy", "pandas", "sklearn")

  for (pkg in required_packages) {
    installed <- py_module_available(pkg)
    status <- ifelse(installed, "\u2713", "\u2717")
    message(sprintf("  %s %s", status, pkg))
  }
}

install_python_dependencies <- function() {
  message("Installing Python dependencies...")
  message("This may take several minutes...")

  original_dir <- getwd()
  setwd(here("scripts/python"))

  # Install requirements
  result <- system("pip install -r requirements.txt", intern = TRUE)
  print(result)

  setwd(original_dir)

  message("Installation complete!")
}

#==============================================================================#
#                    USAGE EXAMPLES                                            #
#==============================================================================#

# Example 1: Check Python setup
# get_python_status()

# Example 2: Run full pipeline
# results <- run_full_pipeline()

# Example 3: Just load existing results
# results <- load_cevae_results()
# plot_cevae_results()

# Example 4: Compare methods
# compare_methods()

#==============================================================================#
#                    DOCUMENTATION                                             #
#==============================================================================#

message("\n========================================")
message("Causal Representation Learning - R Interface")
message("========================================\n")
message("Available functions:")
message("  - run_full_pipeline()      : Run complete analysis")
message("  - run_preprocessing()      : Prepare data")
message("  - run_cevae()             : Train CEVAE model")
message("  - load_cevae_results()    : Load results")
message("  - plot_cevae_results()    : Visualize results")
message("  - compare_methods()       : Compare with traditional methods")
message("  - get_python_status()     : Check Python environment")
message("\nFor detailed documentation, see:")
message("  - scripts/python/README.md")
message("  - CAUSAL_REPRESENTATION_LEARNING.md")
message("\n")
