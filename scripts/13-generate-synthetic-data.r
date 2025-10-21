# 13_generate_synthetic_data.R
# Generate realistic synthetic COVID-19 data with known causal structure
# Allows validation of causal inference methods

# Load required packages
if (!require("pacman")) install.packages("pacman")
pacman::p_load(dplyr, MASS, here)

set.seed(42)

message("========================================")
message("GENERATING SYNTHETIC COVID-19 DATA")
message("========================================\n")

#==============================================================================#
#                    PARAMETERS FOR DATA GENERATION                            #
#==============================================================================#

n <- 10000  # Sample size (matching real data)

# True causal parameters (KNOWN)
true_ATE <- 0.045  # Average treatment effect
true_age_effect <- 0.003  # Per year increase in age
true_male_effect <- 0.015  # Males have higher baseline risk
true_wave_effects <- c(0, -0.010, -0.020)  # Wave 2 and 3 improvements

# Unmeasured confounder effects
true_frailty_treatment <- 0.8  # Frailer patients more likely high viral load
true_frailty_outcome <- 1.2  # Frailer patients more likely to die

message("True causal parameters:")
message(sprintf("  ATE (high vs low viral load): %.4f", true_ATE))
message(sprintf("  Age effect (per year): %.4f", true_age_effect))
message(sprintf("  Male effect: %.4f", true_male_effect))
message(sprintf("  Wave 2 effect: %.4f", true_wave_effects[2]))
message(sprintf("  Wave 3 effect: %.4f\n", true_wave_effects[3]))

#==============================================================================#
#                    GENERATE BASELINE COVARIATES                              #
#==============================================================================#

message("Step 1: Generating baseline covariates...")

# Age: realistic distribution for hospitalized COVID patients
age <- rnorm(n, mean = 62, sd = 18)
age <- pmax(18, pmin(100, age))  # Truncate at 18-100

# Gender: slightly more males
gender <- rbinom(n, 1, prob = 0.55)  # 1 = male, 0 = female

# Pandemic wave: roughly equal distribution
wave <- sample(1:3, n, replace = TRUE, prob = c(0.35, 0.35, 0.30))

# Hospital: 12 hospitals with varying sizes
hospital_probs <- c(0.12, 0.10, 0.09, 0.08, 0.08, 0.08,
                   0.08, 0.07, 0.10, 0.09, 0.06, 0.05)
hospital <- sample(1:12, n, replace = TRUE, prob = hospital_probs)

# Hospital quality (latent variable affecting outcomes)
hospital_quality <- c(0.02, -0.01, -0.03, 0.01, 0.00, -0.02,
                     0.01, -0.01, 0.00, -0.02, 0.01, 0.03)
hosp_quality_effect <- hospital_quality[hospital]

message(sprintf("  Generated %d patients", n))
message(sprintf("  Age: mean=%.1f, sd=%.1f", mean(age), sd(age)))
message(sprintf("  Male: %.1f%%", 100 * mean(gender)))
message(sprintf("  Wave distribution: W1=%.1f%%, W2=%.1f%%, W3=%.1f%%",
                100 * mean(wave==1), 100 * mean(wave==2), 100 * mean(wave==3)))

#==============================================================================#
#                    GENERATE UNMEASURED CONFOUNDER (FRAILTY)                  #
#==============================================================================#

message("\nStep 2: Generating unmeasured confounder (frailty)...")

# Frailty index (latent variable) - UNMEASURED
# Correlated with age and gender
frailty_base <- 0.3 * scale(age)[,1] + 0.1 * gender + rnorm(n, 0, 0.5)
frailty <- pnorm(frailty_base)  # Map to [0, 1]

message(sprintf("  Frailty: mean=%.3f, sd=%.3f", mean(frailty), sd(frailty)))
message("  (This is the unmeasured confounder!)")

#==============================================================================#
#                    GENERATE TREATMENT (VIRAL LOAD)                           #
#==============================================================================#

message("\nStep 3: Generating treatment assignment (viral load)...")

# Treatment propensity (probability of high viral load)
# Depends on: age, gender, wave, hospital, AND frailty (unmeasured!)
logit_treatment <- -0.5 +  # Intercept
                   0.015 * (age - 60) +  # Age effect
                   0.2 * gender +  # Male effect
                   -0.3 * (wave == 2) +  # Wave 2 less likely
                   -0.5 * (wave == 3) +  # Wave 3 even less likely
                   0.1 * rnorm(n) +  # Hospital random effects
                   true_frailty_treatment * scale(frailty)[,1]  # FRAILTY (unmeasured)

ps_true <- plogis(logit_treatment)  # True propensity score

# Actual treatment assignment
high_viral_load <- rbinom(n, 1, ps_true)

message(sprintf("  High viral load: %d (%.1f%%)", sum(high_viral_load),
                100 * mean(high_viral_load)))
message(sprintf("  Propensity score: mean=%.3f, range=[%.3f, %.3f]",
                mean(ps_true), min(ps_true), max(ps_true)))

# Generate continuous Ct value (for realism)
# Low Ct = high viral load, high Ct = low viral load
Ct <- ifelse(high_viral_load == 1,
            rnorm(n, mean = 20, sd = 3),  # Strongly positive
            rnorm(n, mean = 33, sd = 3))  # Weakly positive
Ct <- pmax(10, pmin(40, Ct))  # Truncate at plausible range

#==============================================================================#
#                    GENERATE OUTCOMES (MORTALITY)                             #
#==============================================================================#

message("\nStep 4: Generating outcomes (mortality)...")

# Potential outcomes under control (low viral load)
logit_Y0 <- -2.0 +  # Baseline (low mortality ~10%)
           true_age_effect * age +  # Age effect
           true_male_effect * gender +  # Male effect
           true_wave_effects[wave] +  # Wave effects
           hosp_quality_effect +  # Hospital quality
           true_frailty_outcome * scale(frailty)[,1] +  # FRAILTY (unmeasured)
           rnorm(n, 0, 0.3)  # Individual variation

Y0_prob <- plogis(logit_Y0)

# Potential outcomes under treatment (high viral load)
# ATE = true_ATE, so logit(Y1) = logit(Y0) + log(OR)
# For simplicity, assume constant additive effect on probability scale
Y1_prob <- plogis(qlogis(Y0_prob) + log(1 + true_ATE * 10))  # Approx

# Observed outcome (based on actual treatment)
Y_prob <- ifelse(high_viral_load == 1, Y1_prob, Y0_prob)
mortality <- rbinom(n, 1, Y_prob)

message(sprintf("  Mortality rate: %.1f%%", 100 * mean(mortality)))
message(sprintf("  E[Y(0)] (under control): %.3f", mean(Y0_prob)))
message(sprintf("  E[Y(1)] (under treatment): %.3f", mean(Y1_prob)))
message(sprintf("  True ATE: %.4f", mean(Y1_prob - Y0_prob)))

#==============================================================================#
#                    GENERATE LENGTH OF STAY                                   #
#==============================================================================#

message("\nStep 5: Generating length of stay...")

# LoS depends on: age, mortality, viral load, hospital
los_mean <- 8 +
           0.05 * age +
           3 * mortality +  # Deaths have longer stays
           1.5 * high_viral_load +
           rnorm(n, 0, 2)

# Generate from Weibull distribution
los_shape <- 1.8
los_scale <- los_mean / gamma(1 + 1/los_shape)
LoS <- rweibull(n, shape = los_shape, scale = los_scale)
LoS <- pmax(0.5, pmin(60, LoS))  # Truncate at plausible range

# Censor some observations (discharged before event)
censoring_prob <- 0.05
censored <- rbinom(n, 1, censoring_prob)
# If censored, set mortality to 0 and adjust LoS
LoS[censored == 1] <- LoS[censored == 1] * 0.8
mortality[censored == 1] <- 0

message(sprintf("  LoS: mean=%.2f, median=%.2f", mean(LoS), median(LoS)))
message(sprintf("  Censored: %d (%.1f%%)", sum(censored), 100 * mean(censored)))

#==============================================================================#
#                    CREATE SYNTHETIC DATASET                                  #
#==============================================================================#

message("\nStep 6: Creating synthetic dataset...")

synthetic_data <- data.frame(
  patient_id = 1:n,
  age = age,
  gender = gender,
  wave = wave,
  hosp_id = hospital,
  Ct = Ct,
  high_viral_load = high_viral_load,
  LoS = LoS,
  mort_hospital = mortality,
  # Store true values for validation (would not be observed in real data)
  frailty = frailty,  # UNMEASURED
  Y0_prob = Y0_prob,  # Potential outcome under control
  Y1_prob = Y1_prob,  # Potential outcome under treatment
  ITE_true = Y1_prob - Y0_prob,  # True individual treatment effect
  ps_true = ps_true,  # True propensity score
  censored = censored
)

# Create categorical versions (like real data)
synthetic_data <- synthetic_data %>%
  mutate(
    ct_gp = case_when(
      Ct <= 24 ~ "Strongly positive",
      Ct <= 30 ~ "Moderately Positive",
      TRUE ~ "Weakly Positive"
    ),
    age_gp = case_when(
      age <= 40 ~ "Young Adults",
      age <= 60 ~ "Middle-aged Adults",
      TRUE ~ "Elderly"
    ),
    sex = factor(ifelse(gender == 1, "Male", "Female"))
  )

message(sprintf("  Created synthetic dataset with %d observations", nrow(synthetic_data)))

#==============================================================================#
#                    SAVE SYNTHETIC DATA                                       #
#==============================================================================#

message("\nStep 7: Saving synthetic data...")

# Save full dataset with truth
save(synthetic_data, true_ATE, true_age_effect, true_male_effect,
     true_wave_effects, true_frailty_treatment, true_frailty_outcome,
     file = here("data/synthetic_covid_data_with_truth.RData"))

# Save dataset WITHOUT truth values (for realistic testing)
synthetic_data_observed <- synthetic_data %>%
  select(-frailty, -Y0_prob, -Y1_prob, -ITE_true, -ps_true)

save(synthetic_data_observed,
     file = here("data/synthetic_covid_data.RData"))

# Also save as CSV (matching format of real data)
synthetic_csv <- synthetic_data_observed %>%
  select(patient_id, hosp_id, LoS, age, gender, Ct, wave, mort_hospital) %>%
  mutate(wave = as.character(wave))

write.table(synthetic_csv,
           file = here("data/synthetic_covid_sample_data.txt"),
           sep = ";",
           row.names = FALSE,
           quote = TRUE)

message("  Saved to:")
message("    - data/synthetic_covid_data_with_truth.RData (includes true values)")
message("    - data/synthetic_covid_data.RData (observed data only)")
message("    - data/synthetic_covid_sample_data.txt (CSV format)")

#==============================================================================#
#                    VALIDATION: CHECK TRUE ATE                                #
#==============================================================================#

message("\n========================================")
message("VALIDATION")
message("========================================\n")

message("True causal effects (from data generation):")
true_ate_empirical <- mean(synthetic_data$Y1_prob - synthetic_data$Y0_prob)
true_att_empirical <- mean(synthetic_data$ITE_true[synthetic_data$high_viral_load == 1])
true_atc_empirical <- mean(synthetic_data$ITE_true[synthetic_data$high_viral_load == 0])

message(sprintf("  ATE (population): %.4f", true_ate_empirical))
message(sprintf("  ATT (treated): %.4f", true_att_empirical))
message(sprintf("  ATC (controls): %.4f", true_atc_empirical))

# Naive estimate (ignoring confounding)
naive_ate <- mean(synthetic_data$mort_hospital[synthetic_data$high_viral_load == 1]) -
            mean(synthetic_data$mort_hospital[synthetic_data$high_viral_load == 0])

message(sprintf("\nNaive estimate (ignoring confounding): %.4f", naive_ate))
message(sprintf("Bias: %.4f (%.1f%% of true effect)",
                naive_ate - true_ate_empirical,
                100 * (naive_ate - true_ate_empirical) / true_ate_empirical))

message("\nThis demonstrates why causal inference methods are needed!")
message("Naive comparison is biased due to unmeasured frailty.")

#==============================================================================#
#                    CREATE SUMMARY REPORT                                     #
#==============================================================================#

sink(here("data/synthetic_data_codebook.txt"))
cat("SYNTHETIC COVID-19 DATA CODEBOOK\n")
cat("=================================\n\n")
cat("Generated:", as.character(Sys.time()), "\n\n")

cat("DATA GENERATION PARAMETERS:\n")
cat("---------------------------\n")
cat(sprintf("Sample size: %d\n", n))
cat(sprintf("True ATE: %.4f\n", true_ATE))
cat(sprintf("Age effect: %.4f per year\n", true_age_effect))
cat(sprintf("Male effect: %.4f\n", true_male_effect))
cat(sprintf("Wave 2 effect: %.4f\n", true_wave_effects[2]))
cat(sprintf("Wave 3 effect: %.4f\n", true_wave_effects[3]))
cat(sprintf("Frailty -> Treatment: %.2f (unmeasured confounding)\n", true_frailty_treatment))
cat(sprintf("Frailty -> Outcome: %.2f (unmeasured confounding)\n\n", true_frailty_outcome))

cat("VARIABLES:\n")
cat("----------\n")
cat("patient_id: Unique patient identifier (1 to N)\n")
cat("age: Age in years (continuous, 18-100)\n")
cat("gender: 0=Female, 1=Male\n")
cat("wave: Pandemic wave (1, 2, or 3)\n")
cat("hosp_id: Hospital identifier (1-12)\n")
cat("Ct: Cycle threshold value (continuous, 10-40)\n")
cat("  Lower Ct = higher viral load\n")
cat("high_viral_load: Treatment indicator (1=high VL, 0=low VL)\n")
cat("LoS: Length of hospital stay in days (continuous)\n")
cat("mort_hospital: Mortality indicator (1=death, 0=survival)\n")
cat("ct_gp: Categorical Ct (Strongly/Moderately/Weakly positive)\n")
cat("age_gp: Categorical age (Young/Middle-aged/Elderly)\n")
cat("sex: Categorical gender (Male/Female)\n\n")

cat("UNMEASURED VARIABLES (in with_truth.RData only):\n")
cat("-------------------------------------------------\n")
cat("frailty: Latent frailty index (0-1, unmeasured confounder)\n")
cat("Y0_prob: Potential outcome probability under control\n")
cat("Y1_prob: Potential outcome probability under treatment\n")
cat("ITE_true: True individual treatment effect (Y1 - Y0)\n")
cat("ps_true: True propensity score P(Treatment | Covariates, Frailty)\n\n")

cat("TRUE CAUSAL EFFECTS:\n")
cat("-------------------\n")
cat(sprintf("ATE: %.4f\n", true_ate_empirical))
cat(sprintf("ATT: %.4f\n", true_att_empirical))
cat(sprintf("ATC: %.4f\n\n", true_atc_empirical))

cat("USAGE:\n")
cat("------\n")
cat("1. Use synthetic_covid_data.RData for testing causal methods\n")
cat("   (mimics real-world scenario with unmeasured confounding)\n\n")
cat("2. Use synthetic_covid_data_with_truth.RData for validation\n")
cat("   (compare estimated effects to known truth)\n\n")
cat("3. Example validation:\n")
cat("   load('data/synthetic_covid_data_with_truth.RData')\n")
cat("   # Your causal method estimate\n")
cat("   estimated_ATE <- your_method(synthetic_data)\n")
cat("   # Compare to truth\n")
cat("   bias <- estimated_ATE - true_ATE\n")
cat("   relative_bias <- bias / true_ATE\n\n")

cat("EXPECTED PERFORMANCE OF METHODS:\n")
cat("--------------------------------\n")
cat("If unmeasured frailty is ignored:\n")
cat(sprintf("  Naive estimate bias: %.4f (%.0f%% of true effect)\n",
            naive_ate - true_ate_empirical,
            100 * abs(naive_ate - true_ate_empirical) / true_ate_empirical))
cat("  IPW will be biased (misses frailty)\n")
cat("  G-formula will be biased (misses frailty)\n\n")
cat("If unmeasured frailty is learned:\n")
cat("  CEVAE should recover true ATE (± sampling error)\n")
cat("  Deep learning methods that infer latent confounders should perform well\n\n")

cat("This synthetic data allows you to test whether causal methods\n")
cat("can handle unmeasured confounding!\n")

sink()

message("\nCodebook saved to: data/synthetic_data_codebook.txt")

#==============================================================================#
#                    DESCRIPTIVE STATISTICS                                    #
#==============================================================================#

message("\n========================================")
message("DESCRIPTIVE STATISTICS")
message("========================================\n")

summary_stats <- synthetic_data_observed %>%
  summarise(
    N = n(),
    Age_mean = mean(age),
    Age_sd = sd(age),
    Male_pct = 100 * mean(gender),
    Ct_mean = mean(Ct),
    Ct_sd = sd(Ct),
    LoS_mean = mean(LoS),
    LoS_median = median(LoS),
    Mortality_pct = 100 * mean(mort_hospital),
    HighVL_pct = 100 * mean(high_viral_load)
  )

message("Summary statistics:")
print(summary_stats)

message("\nBy treatment group:")
by_treatment <- synthetic_data_observed %>%
  group_by(high_viral_load) %>%
  summarise(
    N = n(),
    Age_mean = mean(age),
    Male_pct = 100 * mean(gender),
    Mortality_pct = 100 * mean(mort_hospital),
    LoS_mean = mean(LoS)
  )
print(by_treatment)

message("\n========================================")
message("SYNTHETIC DATA GENERATION COMPLETE!")
message("========================================\n")
message("Use this data to:")
message("  1. Test causal inference methods")
message("  2. Validate model performance (compare to known truth)")
message("  3. Demonstrate importance of handling unmeasured confounding")
message("  4. Benchmark different approaches")
