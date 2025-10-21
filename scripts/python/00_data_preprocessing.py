"""
00_data_preprocessing.py

Data preprocessing pipeline for causal representation learning.
Loads COVID-19 data, handles missing values, creates treatment/outcome variables,
and prepares train/validation/test splits.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class COVID19DataPreprocessor:
    """
    Preprocessor for COVID-19 survival data for causal representation learning.
    """

    def __init__(self, data_path="../../data/covid_sample_data.txt"):
        """
        Initialize preprocessor.

        Args:
            data_path: Path to raw data file
        """
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def load_data(self):
        """Load raw data from file."""
        print(f"Loading data from {self.data_path}...")
        self.data = pd.read_csv(self.data_path, sep=";")
        print(f"  Loaded {len(self.data)} observations")
        print(f"  Variables: {list(self.data.columns)}")
        return self

    def handle_missing(self):
        """Handle missing values."""
        print("\nHandling missing values...")

        # Check missing patterns
        missing_counts = self.data.isnull().sum()
        print("  Missing values per variable:")
        print(missing_counts[missing_counts > 0])

        # Remove rows with missing LoS (cannot estimate survival without it)
        n_before = len(self.data)
        self.data = self.data.dropna(subset=['LoS', 'mort_hospital'])
        n_after = len(self.data)
        print(f"  Removed {n_before - n_after} rows with missing LoS/mortality")

        # For Ct values: impute with median or drop
        # Here we'll use median imputation
        if self.data['Ct'].isnull().any():
            ct_median = self.data['Ct'].median()
            n_imputed = self.data['Ct'].isnull().sum()
            self.data['Ct'].fillna(ct_median, inplace=True)
            print(f"  Imputed {n_imputed} missing Ct values with median={ct_median:.2f}")

        return self

    def create_features(self):
        """Create feature matrices and treatment/outcome variables."""
        print("\nCreating feature matrices...")

        # Encode categorical variables
        for col in ['wave', 'hosp_id']:
            le = LabelEncoder()
            self.data[f'{col}_encoded'] = le.fit_transform(self.data[col].astype(str))
            self.label_encoders[col] = le

        # Create treatment variables
        # Treatment 1: High viral load (strongly positive)
        self.data['high_viral_load'] = (self.data['Ct'] <= 24).astype(int)

        # Treatment 2: Late wave vs early wave
        self.data['late_wave'] = (self.data['wave'].astype(str) == '3').astype(int)

        # Feature matrix X (covariates)
        feature_cols = ['age', 'gender', 'wave_encoded', 'hosp_id_encoded']
        self.X = self.data[feature_cols].values.astype(np.float32)
        self.feature_names = feature_cols

        # Treatment A (primary: viral load)
        self.A = self.data['high_viral_load'].values.astype(np.float32)

        # Outcome Y (mortality)
        self.Y = self.data['mort_hospital'].values.astype(np.float32)

        # Time to event T (length of stay)
        self.T = self.data['LoS'].values.astype(np.float32)

        # Additional continuous treatment (for dose-response)
        self.Ct = self.data['Ct'].values.astype(np.float32)

        print(f"  Feature matrix X: shape {self.X.shape}")
        print(f"  Treatment A (high viral load): {self.A.sum():.0f} treated, "
              f"{len(self.A) - self.A.sum():.0f} control")
        print(f"  Outcome Y (mortality): {self.Y.sum():.0f} deaths "
              f"({100*self.Y.mean():.1f}%)")
        print(f"  Time T: mean={self.T.mean():.2f} days, median={np.median(self.T):.2f}")

        return self

    def create_splits(self, test_size=0.2, val_size=0.1, stratify=True):
        """
        Create train/validation/test splits.

        Args:
            test_size: Proportion for test set
            val_size: Proportion of train set for validation
            stratify: Whether to stratify by treatment and outcome
        """
        print(f"\nCreating data splits (test={test_size}, val={val_size})...")

        # Stratify by treatment × mortality for balanced splits
        if stratify:
            stratify_var = self.A.astype(int) * 2 + self.Y.astype(int)
        else:
            stratify_var = None

        # First split: train+val vs test
        indices = np.arange(len(self.X))
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=42,
            stratify=stratify_var if stratify else None
        )

        # Second split: train vs val
        if stratify:
            stratify_train_val = stratify_var[train_val_idx]
        else:
            stratify_train_val = None

        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_size / (1 - test_size),
            random_state=42,
            stratify=stratify_train_val
        )

        # Create splits
        self.X_train = self.X[train_idx]
        self.X_val = self.X[val_idx]
        self.X_test = self.X[test_idx]

        self.A_train = self.A[train_idx]
        self.A_val = self.A[val_idx]
        self.A_test = self.A[test_idx]

        self.Y_train = self.Y[train_idx]
        self.Y_val = self.Y[val_idx]
        self.Y_test = self.Y[test_idx]

        self.T_train = self.T[train_idx]
        self.T_val = self.T[val_idx]
        self.T_test = self.T[test_idx]

        # Also save Ct values
        self.Ct_train = self.Ct[train_idx]
        self.Ct_val = self.Ct[val_idx]
        self.Ct_test = self.Ct[test_idx]

        # Print split statistics
        print(f"  Train: n={len(train_idx)}, treated={self.A_train.sum():.0f}, "
              f"deaths={self.Y_train.sum():.0f}")
        print(f"  Val:   n={len(val_idx)}, treated={self.A_val.sum():.0f}, "
              f"deaths={self.Y_val.sum():.0f}")
        print(f"  Test:  n={len(test_idx)}, treated={self.A_test.sum():.0f}, "
              f"deaths={self.Y_test.sum():.0f}")

        return self

    def standardize(self):
        """Standardize features (fit on train, transform all)."""
        print("\nStandardizing features...")

        # Fit on training data
        self.scaler.fit(self.X_train)

        # Transform all splits
        self.X_train_scaled = self.scaler.transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print(f"  Feature means (train): {self.X_train_scaled.mean(axis=0)}")
        print(f"  Feature stds (train): {self.X_train_scaled.std(axis=0)}")

        return self

    def save_processed_data(self, output_dir="../../data/processed"):
        """Save processed data for use in modeling."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving processed data to {output_dir}...")

        # Save as numpy arrays
        np.save(output_path / "X_train.npy", self.X_train_scaled)
        np.save(output_path / "X_val.npy", self.X_val_scaled)
        np.save(output_path / "X_test.npy", self.X_test_scaled)

        np.save(output_path / "A_train.npy", self.A_train)
        np.save(output_path / "A_val.npy", self.A_val)
        np.save(output_path / "A_test.npy", self.A_test)

        np.save(output_path / "Y_train.npy", self.Y_train)
        np.save(output_path / "Y_val.npy", self.Y_val)
        np.save(output_path / "Y_test.npy", self.Y_test)

        np.save(output_path / "T_train.npy", self.T_train)
        np.save(output_path / "T_val.npy", self.T_val)
        np.save(output_path / "T_test.npy", self.T_test)

        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'n_features': self.X_train.shape[1],
            'n_train': len(self.X_train),
            'n_val': len(self.X_val),
            'n_test': len(self.X_test),
            'treatment_name': 'high_viral_load',
            'outcome_name': 'mort_hospital',
            'time_name': 'LoS'
        }

        import json
        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print("  Saved successfully!")

        return self

    def get_summary_statistics(self):
        """Print summary statistics of processed data."""
        print("\n" + "="*70)
        print("DATA SUMMARY STATISTICS")
        print("="*70)

        print(f"\nSample sizes:")
        print(f"  Total: {len(self.X)}")
        print(f"  Train: {len(self.X_train)} ({100*len(self.X_train)/len(self.X):.1f}%)")
        print(f"  Val:   {len(self.X_val)} ({100*len(self.X_val)/len(self.X):.1f}%)")
        print(f"  Test:  {len(self.X_test)} ({100*len(self.X_test)/len(self.X):.1f}%)")

        print(f"\nTreatment (High Viral Load):")
        print(f"  Train: {self.A_train.mean():.3f} ({100*self.A_train.mean():.1f}%)")
        print(f"  Val:   {self.A_val.mean():.3f} ({100*self.A_val.mean():.1f}%)")
        print(f"  Test:  {self.A_test.mean():.3f} ({100*self.A_test.mean():.1f}%)")

        print(f"\nOutcome (Mortality):")
        print(f"  Train: {self.Y_train.mean():.3f} ({100*self.Y_train.mean():.1f}%)")
        print(f"  Val:   {self.Y_val.mean():.3f} ({100*self.Y_val.mean():.1f}%)")
        print(f"  Test:  {self.Y_test.mean():.3f} ({100*self.Y_test.mean():.1f}%)")

        print(f"\nTime to Event (Days):")
        for split, T in [('Train', self.T_train), ('Val', self.T_val), ('Test', self.T_test)]:
            print(f"  {split}: mean={T.mean():.2f}, median={np.median(T):.2f}, "
                  f"std={T.std():.2f}")

        print(f"\nFeatures:")
        for i, name in enumerate(self.feature_names):
            print(f"  {name}: mean={self.X_train[:, i].mean():.2f}, "
                  f"std={self.X_train[:, i].std():.2f}")

        print("\n" + "="*70)


def main():
    """Main preprocessing pipeline."""
    print("="*70)
    print("COVID-19 CAUSAL REPRESENTATION LEARNING - DATA PREPROCESSING")
    print("="*70)

    # Initialize preprocessor
    preprocessor = COVID19DataPreprocessor(
        data_path="../../data/covid_sample_data.txt"
    )

    # Run preprocessing pipeline
    (preprocessor
     .load_data()
     .handle_missing()
     .create_features()
     .create_splits(test_size=0.2, val_size=0.1)
     .standardize()
     .save_processed_data()
     .get_summary_statistics())

    print("\nPreprocessing complete!")
    print("Processed data saved to: data/processed/")
    print("\nNext steps:")
    print("  1. Run CEVAE: python 01_cevae_survival.py")
    print("  2. Run Deep CFR: python 02_deep_cfr_survival.py")
    print("  3. Evaluate models: python 06_evaluation.py")


if __name__ == "__main__":
    main()
