"""
Simplified data preprocessing using only numpy (for environments without full dependencies)
"""

import numpy as np
from pathlib import Path
import json

# Set random seed
np.random.seed(42)

print("="*70)
print("COVID-19 CAUSAL REPRESENTATION LEARNING - DATA PREPROCESSING")
print("="*70)

# Load data
print("\nLoading data from CSV...")
data_path = "../../data/covid_sample_data.txt"

# Read CSV manually (simple parser)
with open(data_path, 'r') as f:
    lines = f.readlines()

# Parse header
header = lines[0].strip().replace('"', '').split(';')
print(f"  Variables: {header}")

# Parse data
data_dict = {col: [] for col in header}
n_rows = 0

for line in lines[1:]:
    values = line.strip().split(';')
    if len(values) == len(header):
        for col, val in zip(header, values):
            # Convert to appropriate type
            if col in ['patient_id', 'hosp_id', 'wave']:
                data_dict[col].append(val.strip('"'))
            elif col == 'gender':
                data_dict[col].append(int(val) if val else np.nan)
            elif col == 'mort_hospital':
                data_dict[col].append(int(val) if val else np.nan)
            else:  # Numeric
                try:
                    data_dict[col].append(float(val) if val and val != 'NA' else np.nan)
                except:
                    data_dict[col].append(np.nan)
        n_rows += 1

print(f"  Loaded {n_rows} observations")

# Convert to numpy arrays
age = np.array(data_dict['age'])
gender = np.array(data_dict['gender'])
Ct = np.array(data_dict['Ct'])
LoS = np.array(data_dict['LoS'])
mort_hospital = np.array(data_dict['mort_hospital'])

# Encode wave and hospital
wave_unique = sorted(set([w for w in data_dict['wave']]))
hosp_unique = sorted(set([h for h in data_dict['hosp_id']]))

wave_encoded = np.array([wave_unique.index(w) for w in data_dict['wave']])
hosp_encoded = np.array([hosp_unique.index(h) for h in data_dict['hosp_id']])

print(f"  Wave levels: {wave_unique}")
print(f"  Hospital IDs: {hosp_unique}")

# Handle missing values
print("\nHandling missing values...")

# Remove rows with missing LoS or mortality
valid_idx = ~(np.isnan(LoS) | np.isnan(mort_hospital))
print(f"  Removed {(~valid_idx).sum()} rows with missing LoS/mortality")

age = age[valid_idx]
gender = gender[valid_idx]
Ct = Ct[valid_idx]
LoS = LoS[valid_idx]
mort_hospital = mort_hospital[valid_idx]
wave_encoded = wave_encoded[valid_idx]
hosp_encoded = hosp_encoded[valid_idx]

# Impute missing Ct with median
ct_missing = np.isnan(Ct)
if ct_missing.any():
    ct_median = np.nanmedian(Ct)
    Ct[ct_missing] = ct_median
    print(f"  Imputed {ct_missing.sum()} missing Ct values with median={ct_median:.2f}")

print(f"  Final sample size: {len(age)}")

# Create feature matrix
print("\nCreating feature matrices...")
X = np.column_stack([age, gender, wave_encoded, hosp_encoded])
feature_names = ['age', 'gender', 'wave', 'hosp_id']

# Create treatment: high viral load (Ct <= 24)
A = (Ct <= 24).astype(np.float32)

# Outcome: mortality
Y = mort_hospital.astype(np.float32)

# Time: length of stay
T = LoS.astype(np.float32)

print(f"  Feature matrix X: shape {X.shape}")
print(f"  Treatment A (high viral load): {A.sum():.0f} treated, {len(A) - A.sum():.0f} control")
print(f"  Outcome Y (mortality): {Y.sum():.0f} deaths ({100*Y.mean():.1f}%)")
print(f"  Time T: mean={T.mean():.2f} days, median={np.median(T):.2f}")

# Create train/val/test splits
print("\nCreating data splits (test=0.2, val=0.1)...")

# Create stratification variable (treatment × mortality)
stratify_var = (A.astype(int) * 2 + Y.astype(int))

# Stratified split
unique_strata = np.unique(stratify_var)
train_idx = []
val_idx = []
test_idx = []

for stratum in unique_strata:
    stratum_idx = np.where(stratify_var == stratum)[0]
    n_stratum = len(stratum_idx)

    # Shuffle
    np.random.shuffle(stratum_idx)

    # Split
    n_test = int(0.2 * n_stratum)
    n_val = int(0.1 * n_stratum)

    test_idx.extend(stratum_idx[:n_test])
    val_idx.extend(stratum_idx[n_test:n_test+n_val])
    train_idx.extend(stratum_idx[n_test+n_val:])

train_idx = np.array(train_idx)
val_idx = np.array(val_idx)
test_idx = np.array(test_idx)

X_train = X[train_idx]
X_val = X[val_idx]
X_test = X[test_idx]

A_train = A[train_idx]
A_val = A[val_idx]
A_test = A[test_idx]

Y_train = Y[train_idx]
Y_val = Y[val_idx]
Y_test = Y[test_idx]

T_train = T[train_idx]
T_val = T[val_idx]
T_test = T[test_idx]

print(f"  Train: n={len(train_idx)}, treated={A_train.sum():.0f}, deaths={Y_train.sum():.0f}")
print(f"  Val:   n={len(val_idx)}, treated={A_val.sum():.0f}, deaths={Y_val.sum():.0f}")
print(f"  Test:  n={len(test_idx)}, treated={A_test.sum():.0f}, deaths={Y_test.sum():.0f}")

# Standardize features
print("\nStandardizing features...")

# Compute mean and std on training set
train_mean = X_train.mean(axis=0)
train_std = X_train.std(axis=0)

# Avoid division by zero
train_std[train_std == 0] = 1.0

# Standardize all splits
X_train_scaled = (X_train - train_mean) / train_std
X_val_scaled = (X_val - train_mean) / train_std
X_test_scaled = (X_test - train_mean) / train_std

print(f"  Feature means (train): {X_train_scaled.mean(axis=0)}")
print(f"  Feature stds (train): {X_train_scaled.std(axis=0)}")

# Save processed data
print("\nSaving processed data...")
output_dir = Path("../../data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

np.save(output_dir / "X_train.npy", X_train_scaled)
np.save(output_dir / "X_val.npy", X_val_scaled)
np.save(output_dir / "X_test.npy", X_test_scaled)

np.save(output_dir / "A_train.npy", A_train)
np.save(output_dir / "A_val.npy", A_val)
np.save(output_dir / "A_test.npy", A_test)

np.save(output_dir / "Y_train.npy", Y_train)
np.save(output_dir / "Y_val.npy", Y_val)
np.save(output_dir / "Y_test.npy", Y_test)

np.save(output_dir / "T_train.npy", T_train)
np.save(output_dir / "T_val.npy", T_val)
np.save(output_dir / "T_test.npy", T_test)

# Save metadata
metadata = {
    'feature_names': feature_names,
    'n_features': X_train.shape[1],
    'n_train': len(X_train),
    'n_val': len(X_val),
    'n_test': len(X_test),
    'treatment_name': 'high_viral_load',
    'outcome_name': 'mort_hospital',
    'time_name': 'LoS',
    'train_mean': train_mean.tolist(),
    'train_std': train_std.tolist()
}

with open(output_dir / "metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

print("  Saved successfully!")

# Summary statistics
print("\n" + "="*70)
print("DATA SUMMARY STATISTICS")
print("="*70)

print(f"\nSample sizes:")
print(f"  Total: {len(X)}")
print(f"  Train: {len(X_train)} ({100*len(X_train)/len(X):.1f}%)")
print(f"  Val:   {len(X_val)} ({100*len(X_val)/len(X):.1f}%)")
print(f"  Test:  {len(X_test)} ({100*len(X_test)/len(X):.1f}%)")

print(f"\nTreatment (High Viral Load):")
print(f"  Train: {A_train.mean():.3f} ({100*A_train.mean():.1f}%)")
print(f"  Val:   {A_val.mean():.3f} ({100*A_val.mean():.1f}%)")
print(f"  Test:  {A_test.mean():.3f} ({100*A_test.mean():.1f}%)")

print(f"\nOutcome (Mortality):")
print(f"  Train: {Y_train.mean():.3f} ({100*Y_train.mean():.1f}%)")
print(f"  Val:   {Y_val.mean():.3f} ({100*Y_val.mean():.1f}%)")
print(f"  Test:  {Y_test.mean():.3f} ({100*Y_test.mean():.1f}%)")

print(f"\nTime to Event (Days):")
for split_name, T_split in [('Train', T_train), ('Val', T_val), ('Test', T_test)]:
    print(f"  {split_name}: mean={T_split.mean():.2f}, median={np.median(T_split):.2f}, "
          f"std={T_split.std():.2f}")

print(f"\nFeatures:")
for i, name in enumerate(feature_names):
    print(f"  {name}: mean={X_train[:, i].mean():.2f}, std={X_train[:, i].std():.2f}")

print("\n" + "="*70)
print("Preprocessing complete!")
print("Processed data saved to: data/processed/")
