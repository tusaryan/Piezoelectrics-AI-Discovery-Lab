import numpy as np

# ==============================================================================
# MACHINE LEARNING MODEL CONFIGURATION
# ==============================================================================
# This file defines the "Search Space" for the Maximum Accuracy mode.
# You can edit these values to tune the models yourself!
#
# TIPS:
# - Log-Scale (np.logspace) is best for parameters where you don't know the order of magnitude.
#   Example: np.logspace(-4, 0, 5) checks [0.0001, 0.001, 0.01, 0.1, 1.0]
# - Larger lists = More accuracy but SLOWER training.
# ==============================================================================

PARAM_GRIDS = {
    # --------------------------------------------------------------------------
    # 1. Random Forest (Robust, Good Baseline)
    # --------------------------------------------------------------------------
    "Random Forest": {
        'n_estimators': [100, 300, 500],        # Number of trees (More is usually better)
        'max_depth': [10, 20, 30, None],        # Depth of each tree (None = full depth)
        'min_samples_split': [2, 5, 10],        # Minimum samples to split a node
        'max_features': ['sqrt', 'log2', None]  # Features to consider at each split
    },

    # --------------------------------------------------------------------------
    # 2. XGBoost (Gradient Boosting, High Performance)
    # --------------------------------------------------------------------------
    "XGBoost": {
        'n_estimators': [100, 300, 700],
        'learning_rate': [0.01, 0.05, 0.1, 0.2], # Lower rate + more trees = better accuracy
        'max_depth': [3, 5, 7, 9],               # Tree depth (avoid >10 on small data)
        'subsample': [0.6, 0.8, 1.0],            # Fraction of data used per tree
        'colsample_bytree': [0.6, 0.8, 1.0]      # Fraction of features used per tree
    },

    # --------------------------------------------------------------------------
    # 3. Support Vector Machine (SVR) (Excellent for small data)
    # --------------------------------------------------------------------------
    "SVM (SVR)": {
        'regressor__C': np.logspace(-1, 3, 5).tolist(),   # [0.1, 1, 10, 100, 1000] (Penalty Strength)
        'regressor__epsilon': [0.001, 0.01, 0.1, 0.2],    # Error tolerance
        'regressor__gamma': ['scale', 'auto'] + np.logspace(-3, 1, 5).tolist() # Kernel width [0.001 ... 10]
    },

    # --------------------------------------------------------------------------
    # 4. Gaussian Process (The "Gold Standard" for Materials Science)
    # --------------------------------------------------------------------------
    "Gaussian Process": {
        # How many times to restart the optimizer to avoid local minima
        'regressor__n_restarts_optimizer': [0, 2, 5], 
        # Value added to diagonal of kernel matrix (Noise level)
        'regressor__alpha': np.logspace(-10, -2, 5).tolist() # [1e-10 ... 0.01]
    },

    # --------------------------------------------------------------------------
    # 5. Kernel Ridge Regression (KRR) (Powerful non-linear regression)
    # --------------------------------------------------------------------------
    "Kernel Ridge": {
        # Regularization strength (Small = complex model, Large = simple model)
        'regressor__alpha': np.logspace(-4, 0, 5).tolist(), # [0.0001, 0.001 ... 1.0]
        # Kernel coefficient (Crucial for RBF kernel!)
        'regressor__gamma': np.logspace(-3, 2, 6).tolist()  # [0.001, 0.01, 0.1, 1, 10, 100]
    },

    # --------------------------------------------------------------------------
    # 6. LightGBM (Fast & Effective Gradient Boosting)
    # --------------------------------------------------------------------------
    "LightGBM": {
        'n_estimators': [100, 300, 500, 1000],
        'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.2],
        'num_leaves': [20, 31, 50, 70, 100],         # Controls complexity
        'max_depth': [-1, 10, 20, 30],               # -1 = no limit
        'subsample': [0.6, 0.8, 1.0]
    },

    # --------------------------------------------------------------------------
    # 7. Gradient Boosting (sklearn) (Reliable, high quality)
    # --------------------------------------------------------------------------
    "Gradient Boosting": {
        'n_estimators': [100, 300, 500, 800],
        'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'subsample': [0.6, 0.8, 1.0],
        'min_samples_split': [2, 5, 10]
    }
}
