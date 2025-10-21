#!/bin/bash
# Setup script for COVID-19 Causal Inference Framework
# This script installs all required dependencies for R and Python

set -e  # Exit on error

echo "=========================================="
echo "COVID-19 Causal Inference Framework Setup"
echo "=========================================="
echo ""

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="Linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macOS"
else
    OS="Other"
fi

echo "Detected OS: $OS"
echo ""

# Check R installation
echo "Checking R installation..."
if command -v R &> /dev/null; then
    R_VERSION=$(R --version | head -n 1)
    echo "✓ R found: $R_VERSION"
else
    echo "✗ R not found!"
    echo "Please install R from: https://www.r-project.org/"
    exit 1
fi
echo ""

# Check Python installation
echo "Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "✓ Python found: $PYTHON_VERSION"
    PYTHON_INSTALLED=true
else
    echo "⚠ Python3 not found. Deep learning features will be unavailable."
    echo "Install Python from: https://www.python.org/"
    PYTHON_INSTALLED=false
fi
echo ""

# Install R packages
echo "=========================================="
echo "Installing R packages..."
echo "=========================================="
echo ""
echo "This may take 10-20 minutes depending on your system."
echo ""

Rscript -e '
# Core packages
packages <- c(
  "survival", "survminer", "ggplot2", "dplyr", "tidyr",
  "randomForestSRC", "gbm", "glmnet",
  "dagitty", "ggdag", "ggridges",
  "pROC", "broom", "gridExtra", "cowplot"
)

# Advanced packages
advanced_packages <- c("BART", "tmle", "SuperLearner", "EValue")

# Function to install packages
install_if_missing <- function(pkg) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    cat(sprintf("Installing %s...\n", pkg))
    install.packages(pkg, dependencies = TRUE, repos = "https://cloud.r-project.org")
    return(TRUE)
  } else {
    cat(sprintf("✓ %s already installed\n", pkg))
    return(FALSE)
  }
}

# Install core packages
cat("\n=== Core Packages ===\n")
core_installed <- sapply(packages, install_if_missing)

# Install advanced packages (may fail on some systems)
cat("\n=== Advanced Packages ===\n")
for (pkg in advanced_packages) {
  tryCatch({
    install_if_missing(pkg)
  }, error = function(e) {
    cat(sprintf("⚠ Warning: Could not install %s\n", pkg))
    cat(sprintf("  Error: %s\n", e$message))
    cat(sprintf("  Analysis will still work without %s\n", pkg))
  })
}

cat("\n=== R Package Installation Complete ===\n")
'

echo ""
echo "R package installation complete!"
echo ""

# Install Python packages if Python is available
if [ "$PYTHON_INSTALLED" = true ]; then
    echo "=========================================="
    echo "Setting up Python environment..."
    echo "=========================================="
    echo ""

    # Check if virtual environment exists
    if [ ! -d "causal_env" ]; then
        echo "Creating virtual environment..."
        python3 -m venv causal_env
        echo "✓ Virtual environment created"
    else
        echo "✓ Virtual environment already exists"
    fi

    echo ""
    echo "Activating virtual environment..."
    source causal_env/bin/activate

    echo "Upgrading pip..."
    pip install --upgrade pip --quiet

    echo ""
    echo "Installing Python packages..."
    echo "This may take 5-10 minutes..."
    echo ""

    if [ -f "scripts/python/requirements.txt" ]; then
        pip install -r scripts/python/requirements.txt
        echo ""
        echo "✓ Python packages installed successfully"
    else
        echo "⚠ requirements.txt not found, installing core packages..."
        pip install numpy pandas torch scikit-learn lifelines matplotlib seaborn
        echo "✓ Core Python packages installed"
    fi

    deactivate
    echo ""
fi

# Create results directories
echo "=========================================="
echo "Creating output directories..."
echo "=========================================="
echo ""

mkdir -p results/{figures,tables,traditional_methods,ml_methods,deep_learning,advanced_methods,sensitivity,synthetic}
mkdir -p models
mkdir -p data/preprocessed

echo "✓ Directory structure created"
echo ""

# Verify installation
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="
echo ""

Rscript -e '
# Check critical packages
critical <- c("survival", "ggplot2", "dplyr", "randomForestSRC", "gbm")
all_ok <- TRUE

for (pkg in critical) {
  if (require(pkg, character.only = TRUE, quietly = TRUE)) {
    cat(sprintf("✓ %s loaded successfully\n", pkg))
  } else {
    cat(sprintf("✗ %s failed to load\n", pkg))
    all_ok <- FALSE
  }
}

if (all_ok) {
  cat("\n=== All critical R packages OK ===\n")
} else {
  cat("\n⚠ Some packages failed - see errors above\n")
}
'

echo ""

if [ "$PYTHON_INSTALLED" = true ]; then
    echo "Python environment:"
    source causal_env/bin/activate
    python3 -c "
import sys
try:
    import numpy
    print('✓ NumPy:', numpy.__version__)
except ImportError:
    print('✗ NumPy not available')

try:
    import pandas
    print('✓ Pandas:', pandas.__version__)
except ImportError:
    print('✗ Pandas not available')

try:
    import torch
    print('✓ PyTorch:', torch.__version__)
except ImportError:
    print('✗ PyTorch not available')

try:
    import sklearn
    print('✓ scikit-learn:', sklearn.__version__)
except ImportError:
    print('✗ scikit-learn not available')
"
    deactivate
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Review the documentation:"
echo "   - README_CAUSAL_FRAMEWORK.md (overview)"
echo "   - ANALYSIS_EXECUTION_GUIDE.md (detailed instructions)"
echo ""
echo "2. Run the analysis:"
echo "   - Option A: bash run_complete_analysis.sh"
echo "   - Option B: Individual scripts (see ANALYSIS_EXECUTION_GUIDE.md)"
echo ""
echo "3. For Python/deep learning features:"
echo "   source causal_env/bin/activate"
echo "   cd scripts/python"
echo "   python3 01_cevae_survival.py --epochs 200"
echo ""
echo "4. Check results in the results/ directory"
echo ""
echo "Questions? See documentation or check script comments."
echo ""
