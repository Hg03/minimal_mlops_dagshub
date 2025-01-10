from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from typing import Dict, Any

def evaluate(tuner: Any, y_train: np.array, train_predictions: np.array, y_test: np.array, test_predictions:np.array) -> Dict:    
    # Calculate and log metrics
    train_mse = mean_squared_error(y_train, train_predictions)
    test_mse = mean_squared_error(y_test, test_predictions)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    train_r2 = r2_score(y_train, train_predictions)
    test_r2 = r2_score(y_test, test_predictions)
    
    metrics = {
        "mean_squared_error": test_mse,
        "root_mean_squared_error": test_rmse,
        "r2": test_r2,
        "neg_mean_squared_error": tuner.best_score_
    }
    return metrics
    