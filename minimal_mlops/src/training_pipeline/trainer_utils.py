from typing import Dict, Tuple, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import mlflow
import pickle
from mlflow.models.signature import infer_signature
from sklearn.svm import SVR
from minimal_mlops.src.evaluate import evaluate
import polars as pl
import os

def get_model_with_hyperparams(config: Dict, model_name: str) -> Tuple[Any, Dict]:
    if config.get(model_name, None) == None:
        print("No model configs found in toml file")
        return
    if config[model_name].get("hyperparams", None) == None:
        print("No hyperparams for model existed")
        return
    elif model_name == "random-forest-regressor":
        random_forest_regressor = RandomForestRegressor()
        hyperparams = config[model_name]["hyperparams"]
        return (random_forest_regressor, hyperparams)
    elif model_name == "support-vector-regressor":
        support_vector_regressor  = SVR()
        hyperparams = config[model_name]["hyperparams"]
        return (support_vector_regressor, hyperparams)
        


def tune_and_predict(
    config: Dict,
    training_data: pl.DataFrame,
    testing_data: pl.DataFrame,
    model_name: str,
    model: Any,
    hyperparams: Dict
) -> None:


    # Convert Polars DataFrames to numpy arrays
    X_train = training_data.drop(config["columns"]["target"]).to_numpy()
    y_train = training_data[config["columns"]["target"]].to_numpy()
    X_test = testing_data.drop(config["columns"]["target"]).to_numpy()
    y_test = testing_data[config["columns"]["target"]].to_numpy()
    
    # Start MLflow run
    mlflow.set_experiment(config[model_name]["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name=f"{model.__class__.__name__}_training"):
        
        # Perform tuning
        tuner = GridSearchCV(
            estimator=model,
            param_grid=hyperparams,
            scoring='neg_mean_squared_error'
        )
        
        # Train model on full training set
        tuner.fit(X_train, y_train)
        
        # Make predictions
        train_predictions = tuner.predict(X_train)
        test_predictions = tuner.predict(X_test)
        
        # signature
        model_signature = infer_signature(X_train, test_predictions)

        # log metrics and hyperparams
        metrics = evaluate(tuner=tuner, y_train=y_train, train_predictions=train_predictions, y_test=y_test, test_predictions=test_predictions)
        mlflow.log_metrics(metrics)
        mlflow.log_params(tuner.best_params_)
        # Log the model
        mlflow.sklearn.log_model(sk_model=tuner, artifact_path="model", signature=model_signature)
        pickle.dump(tuner, open(os.path.join(config["path"]["root"], config["path"]["models"], config[model_name]["name"]), "wb"))