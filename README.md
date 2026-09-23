# SellWise

This repository contains a modular, production-ready time-series forecasting pipeline built for the [M5 Forecasting - Accuracy](https://www.kaggle.com/c/m5-forecasting-accuracy) competition. It transitions top-tier Kaggle solutions (specifically based on Konstantin Yakovlev's recursive and non-recursive architectures) from exploratory Jupyter notebooks into a robust, Object-Oriented Python application.

## 📌 Project Overview

The goal of this project is to predict 28 days of daily unit sales for 30,490 items across 10 Walmart stores in 3 US states (California, Texas, and Wisconsin). 

This pipeline scales to train hundreds of highly optimized LightGBM models across different hierarchy levels (Store, Store-Category, and Store-Department) using both non-recursive and recursive forecasting strategies.

## 🏗️ Architecture & Features

*   **Modular OOP Design:** Built using a centralized `ConfigurationManager` to decouple hyperparameters, dataset masking, and file paths from the training logic.
*   **Dual Forecasting Strategies:**
    *   **Non-Recursive Pipeline:** Trains 110 distinct models directly predicting future targets using robust lag and rolling features.
    *   **Recursive Pipeline:** Trains models that predict step-by-step, feeding prior predictions back into lag features for sequential day forecasting.
*   **Hierarchical Training:** Models are trained at three specific granularities to balance data volume and specific item trends:
    *   **10 Store Models:** Heavy capacity (`num_leaves: 2047`, `min_data_in_leaf: 4095`) to capture macro-level seasonality and day-of-week trends.
    *   **30 Store-Category Models:** Medium capacity (`num_leaves: 255`) capturing category-specific effects.
    *   **70 Store-Department Models:** Medium capacity focusing on departmental micro-trends.
*   **Modern LightGBM Integration:** Uses native `callbacks` for early stopping and evaluation logging, optimized Tweedie objectives, and precise baseline initialization (`boost_from_average`).
*   **Memory Optimization:** Built-in memory reduction and automated feature dropping to prevent RAM exhaustion when handling tens of millions of rows.

## 📂 Project Structure

```text
├── artifacts/                # Generated datasets, models, and evaluation metrics
│   ├── model_trainer/        # Saved LightGBM binaries (.bin files)
├── config/
│   └── config.yaml           # Global file paths and directory settings
├── research/                 # Original Jupyter notebooks for EDA and testing
├── src/                      # Core OOP pipeline source code
│   ├── components/           # Data ingestion, processing, and trainer classes
│   ├── config/               # ConfigurationManager logic
│   └── pipeline/             # End-to-end execution wrappers
├── params.yaml               # Centralized hyperparameters and validation masking
└── main.py                   # Main execution entry point
```

## ⚙️ Configuration (`params.yaml`)

All training behavior is controlled via `params.yaml`. This avoids hardcoding logic in scripts.

### Validation Strategies (`cv` parameter)
The pipeline supports dynamic switching between local cross-validation and final private predictions:
*   **`cv: cv4` (Stage 1 - Tuning):** Validates on days `1886` to `1913`. Ground truth sales are known, allowing `early_stopping_rounds` to find the optimal number of trees without data leakage.
*   **`cv: private` (Stage 2 - Production):** Validates on days `1941` to `1969`. Used for final model generation using all historical data. *Note: Early stopping must be disabled or rely on fixed iterations here.*

### Target Ending Window (`end_train`)
Defines the boundary of the training data.
*   Set `end_train: 1941` for the final Private Leaderboard models.
*   Set `end_train: 1913` for Public Leaderboard evaluation.

## 🚀 Usage

**1. Environment Setup:**
Ensure Python 3.10+ is installed, then install dependencies:
```bash
pip install -r requirements.txt
```

**2. Configure Pipeline:**
Update `params.yaml` with your desired validation strategy (`cv`), hierarchical level parameters, and paths.

**3. Run Training:**
Execute the pipeline via the main entry point to trigger the non-recursive or recursive trainers:
```bash
python main.py
```
*(Alternatively, you can run specific stages using the modular pipeline scripts in `src/pipeline/`)*

## 🧠 Model Insights & Best Practices

*   **Tweedie Objective:** The models use the `tweedie` objective (`tweedie_variance_power: 1.1`) to naturally handle zero-inflated sales distributions (days where an item sells 0 units).
*   **Target Means & Convergence:** For subset models (like specific departments), `boost_from_average: true` is utilized to center the initial iteration at the subset's historical mean, accelerating convergence and saving boosting rounds.
*   **Feature Leakage Prevention:** The pipeline explicitly drops grouping keys at each level (e.g., dropping `store_id` for Store-level models) to prevent redundant data from skewing tree splits.

## 🙏 Acknowledgements

This architecture heavily references the top-performing M5 competition solutions, with special thanks to the Kaggle community and Konstantin Yakovlev ("A1") for the foundational data structures and recursive boosting strategies.