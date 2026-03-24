from typing import Dict, Any

MODEL_REGISTRY = {
    "XGBoost": {
        "class_name": "xgboost.XGBRegressor",
        "description": "Gradient boosting framework that is highly efficient and flexible.",
        "params": [
            {
                "name": "n_estimators", "type": "int", "default": 200, "min": 50, "max": 1000, "optuna_range": [50, 500],
                "label": "Number of Trees", "help": "Number of boosting rounds.", "beginner_tip": "Higher is better but slower to train. 100-300 is a good starting point.", "impact": "high"
            },
            {
                "name": "max_depth", "type": "int", "default": 6, "min": 2, "max": 15, "optuna_range": [3, 10],
                "label": "Max Depth", "help": "Maximum depth of a tree.", "beginner_tip": "Controls model complexity. Low values prevent overfitting.", "impact": "high"
            },
            {
                "name": "learning_rate", "type": "float_log", "default": 0.1, "min": 0.001, "max": 1.0, "optuna_range": [0.01, 0.3],
                "label": "Learning Rate", "help": "Step size shrinkage used in update to prevents overfitting.", "beginner_tip": "Smaller learning rate needs more trees.", "impact": "high"
            }
        ]
    },
    "LightGBM": {
        "class_name": "lightgbm.LGBMRegressor",
        "description": "Fast, distributed, high performance gradient boosting framework based on decision tree algorithms.",
        "params": [
            {
                "name": "n_estimators", "type": "int", "default": 200, "min": 50, "max": 1000, "optuna_range": [50, 500],
                "label": "Number of Trees", "help": "Number of boosting iterations.", "beginner_tip": "Good starting point is 100.", "impact": "high"
            },
            {
                "name": "num_leaves", "type": "int", "default": 31, "min": 10, "max": 150, "optuna_range": [15, 63],
                "label": "Max Leaves per Tree", "help": "Main parameter to control complexity of the tree.", "beginner_tip": "Keep smaller than 2^(max_depth) to prevent overfitting.", "impact": "high"
            }
        ]
    },
    "RandomForest": {
        "class_name": "sklearn.ensemble.RandomForestRegressor",
        "description": "A meta estimator that fits a number of classifying decision trees on various sub-samples.",
        "params": [
            {
                "name": "n_estimators", "type": "int", "default": 100, "min": 10, "max": 1000, "optuna_range": [50, 300],
                "label": "Number of Trees", "help": "Number of trees in the forest.", "beginner_tip": "More trees means more robust predictions, but slower.", "impact": "high"
            },
            {
                "name": "max_depth", "type": "int", "default": 0, "min": 0, "max": 50, "optuna_range": [5, 30],
                "label": "Max Depth", "help": "Depth of each tree. 0 means unlimited.", "beginner_tip": "Set to 0 usually, unless overfitting heavily.", "impact": "medium"
            }
        ]
    },
    "GradientBoosting": {
        "class_name": "sklearn.ensemble.GradientBoostingRegressor",
        "description": "Builds an additive model in a forward stage-wise fashion.",
        "params": [
            {
                "name": "n_estimators", "type": "int", "default": 100, "min": 50, "max": 500, "optuna_range": [50, 200],
                "label": "Number of Estimators", "help": "Number of boosting stages to perform.", "beginner_tip": "Pairs tightly with learning rate.", "impact": "high"
            }
        ]
    },
    "SVR": {
        "class_name": "sklearn.svm.SVR",
        "description": "Epsilon-Support Vector Regression.",
        "params": [
            {
                "name": "C", "type": "float_log", "default": 1.0, "min": 0.01, "max": 1000.0, "optuna_range": [0.1, 100.0],
                "label": "C (Regularization)", "help": "Regularization parameter. The strength of the regularization is inversely proportional to C.", "beginner_tip": "Helps control overfitting.", "impact": "high"
            }
        ]
    },
    "DecisionTree": {
        "class_name": "sklearn.tree.DecisionTreeRegressor",
        "description": "Simple tree that predicts the value of a target variable by learning simple decision rules inferred from the data features.",
        "params": []
    },
    "GPR": {
        "class_name": "sklearn.gaussian_process.GaussianProcessRegressor",
        "description": "Gaussian process regression (GPR). Excellent for establishing uncertainty estimates.",
        "params": [
            {
                "name": "alpha", "type": "float_log", "default": 1e-10, "min": 1e-10, "max": 0.1, "optuna_range": [1e-10, 1e-2],
                "label": "Alpha (Noise)", "help": "Value added to the diagonal of the kernel matrix during fitting.", "beginner_tip": "Increase if experiencing numerical instability.", "impact": "medium"
            }
        ]
    },
    "ANN": {
        "class_name": "sklearn.neural_network.MLPRegressor",
        "description": "Multi-layer Perceptron regressor.",
        "params": [
            {
                "name": "hidden_layer_sizes", "type": "architecture_builder", "default": [100], "min": 1, "max": 5, "optuna_range": None,
                "label": "Hidden Layers", "help": "Architecture of the neural network.", "beginner_tip": "Start with a single layer, e.g. [100].", "impact": "high"
            }
        ]
    },
    "Stacking": {
        "class_name": "sklearn.ensemble.StackingRegressor",
        "description": "Stack of estimators with a final regressor.",
        "params": [] # Usually dynamically generated in auto mode
    }
}

def get_model_schema() -> Dict[str, Any]:
    return MODEL_REGISTRY
