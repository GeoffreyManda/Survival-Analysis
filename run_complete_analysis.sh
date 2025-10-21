#!/bin/bash
# Complete Analysis Pipeline for COVID-19 Causal Inference
# Runs all analysis scripts in the correct order

set -e  # Exit on error

echo "=============================================="
echo "COVID-19 Causal Inference Complete Analysis"
echo "=============================================="
echo ""
echo "Start time: $(date)"
echo ""

# Check if R is available
if ! command -v Rscript &> /dev/null; then
    echo "ERROR: Rscript not found. Please install R first."
    exit 1
fi

# Create output directories if they don't exist
mkdir -p results/{figures,tables,traditional_methods,ml_methods,deep_learning,advanced_methods,sensitivity,synthetic}

# Phase 1: Traditional Causal Inference
echo "=============================================="
echo "PHASE 1: Traditional Causal Inference"
echo "=============================================="
echo ""

echo "Step 1.1: Defining causal estimands and DAGs..."
echo "Running: 07-causal-inference-estimands.r"
Rscript scripts/07-causal-inference-estimands.r
if [ $? -eq 0 ]; then
    echo "✓ Causal estimands defined successfully"
else
    echo "✗ Error in script 07 - see error messages above"
    exit 1
fi
echo ""

echo "Step 1.2: Implementing IPW, g-formula, and doubly robust estimation..."
echo "Running: 08-causal-analysis-implementation.r"
Rscript scripts/08-causal-analysis-implementation.r
if [ $? -eq 0 ]; then
    echo "✓ Traditional methods completed successfully"
else
    echo "✗ Error in script 08 - see error messages above"
    exit 1
fi
echo ""

# Phase 2: Machine Learning Methods
echo "=============================================="
echo "PHASE 2: Causal Machine Learning"
echo "=============================================="
echo ""

echo "Step 2.1: Random Forests, Gradient Boosting, and Elastic Net..."
echo "Running: 10-causal-machine-learning.r"
Rscript scripts/10-causal-machine-learning.r
if [ $? -eq 0 ]; then
    echo "✓ Machine learning methods completed successfully"
else
    echo "✗ Error in script 10 - see error messages above"
    exit 1
fi
echo ""

# Phase 3: Advanced Methods
echo "=============================================="
echo "PHASE 3: Advanced Methods (BART & TMLE)"
echo "=============================================="
echo ""

echo "Step 3.1: BART and TMLE estimation..."
echo "Running: 12-bart-tmle-methods.r"
Rscript scripts/12-bart-tmle-methods.r
if [ $? -eq 0 ]; then
    echo "✓ Advanced methods completed successfully"
else
    echo "⚠ Warning: Advanced methods failed (BART/TMLE may not be installed)"
    echo "  Analysis can continue without these methods"
fi
echo ""

# Phase 4: Validation
echo "=============================================="
echo "PHASE 4: Validation with Synthetic Data"
echo "=============================================="
echo ""

echo "Step 4.1: Generating synthetic data with known ground truth..."
echo "Running: 13-generate-synthetic-data.r"
Rscript scripts/13-generate-synthetic-data.r
if [ $? -eq 0 ]; then
    echo "✓ Synthetic data generated successfully"
else
    echo "✗ Error in script 13 - see error messages above"
fi
echo ""

# Phase 5: Sensitivity Analyses
echo "=============================================="
echo "PHASE 5: Sensitivity Analyses"
echo "=============================================="
echo ""

echo "Step 5.1: E-values, negative controls, and placebo tests..."
echo "Running: 14-sensitivity-analyses.r"
Rscript scripts/14-sensitivity-analyses.r
if [ $? -eq 0 ]; then
    echo "✓ Sensitivity analyses completed successfully"
else
    echo "⚠ Warning: Sensitivity analyses failed"
    echo "  EValue package may not be installed"
fi
echo ""

# Phase 6: Visualizations
echo "=============================================="
echo "PHASE 6: Creating Visualizations"
echo "=============================================="
echo ""

echo "Step 6.1: Generating 10 publication-ready figures..."
echo "Running: 11-create-visualizations.r"
Rscript scripts/11-create-visualizations.r
if [ $? -eq 0 ]; then
    echo "✓ All visualizations created successfully"
else
    echo "✗ Error in script 11 - see error messages above"
fi
echo ""

# Optional: Deep Learning (if Python environment is set up)
echo "=============================================="
echo "PHASE 7: Deep Learning (Optional)"
echo "=============================================="
echo ""

if [ -d "causal_env" ] && [ -f "causal_env/bin/activate" ]; then
    echo "Python virtual environment found. Attempting deep learning methods..."
    source causal_env/bin/activate

    if python3 -c "import torch; import pandas; import numpy" 2>/dev/null; then
        echo "Step 7.1: Preprocessing data for deep learning..."
        cd scripts/python
        python3 00_data_preprocessing.py

        if [ $? -eq 0 ]; then
            echo "✓ Data preprocessing completed"

            echo "Step 7.2: Training CEVAE (this may take 10-30 minutes)..."
            python3 01_cevae_survival.py --epochs 200 --batch_size 128

            if [ $? -eq 0 ]; then
                echo "✓ CEVAE training completed"
            else
                echo "⚠ CEVAE training failed"
            fi

            echo "Step 7.3: Training Deep CFR (this may take 10-30 minutes)..."
            python3 02_deep_cfr.py --epochs 150 --batch_size 128

            if [ $? -eq 0 ]; then
                echo "✓ Deep CFR training completed"
            else
                echo "⚠ Deep CFR training failed"
            fi
        else
            echo "⚠ Data preprocessing failed - skipping deep learning"
        fi

        cd ../..
    else
        echo "⚠ Python dependencies not fully installed - skipping deep learning"
        echo "  Run: source causal_env/bin/activate && pip install -r scripts/python/requirements.txt"
    fi

    deactivate
else
    echo "⚠ Python virtual environment not found - skipping deep learning"
    echo "  To set up: bash setup_environment.sh"
fi
echo ""

# Summary
echo "=============================================="
echo "ANALYSIS COMPLETE!"
echo "=============================================="
echo ""
echo "End time: $(date)"
echo ""
echo "Results have been saved to:"
echo "  - results/figures/          (10 publication-ready figures)"
echo "  - results/tables/           (statistical tables)"
echo "  - results/traditional_methods/ (IPW, g-formula, DR estimates)"
echo "  - results/ml_methods/       (Random Forest, GBM, Elastic Net)"
echo "  - results/advanced_methods/ (BART, TMLE results)"
echo "  - results/sensitivity/      (E-values, negative controls)"
echo "  - results/synthetic/        (validation with known truth)"
echo ""
echo "Next steps:"
echo "  1. Review results in results/ directory"
echo "  2. Check results/figures/ for publication-ready plots"
echo "  3. Update MANUSCRIPT_TEMPLATE.md with actual results"
echo "  4. Review MODEL_FIT_ANALYSIS.md for interpretation guidance"
echo ""
echo "Key files to check:"
echo "  - results/traditional_methods/ate_estimates.csv"
echo "  - results/ml_methods/cate_predictions.csv"
echo "  - results/sensitivity/e_values.csv"
echo "  - results/figures/figure_02_forest_plot.pdf"
echo "  - results/figures/figure_04_heterogeneity_age.pdf"
echo ""
