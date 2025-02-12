from typing import Dict, Tuple, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import mlflow
import pickle
from minimal_mlops.src.feature_pipeline.feature_engineer_utils import load_raw_from_local
from mlflow.models.signature import infer_signature
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline, make_pipeline
from minimal_mlops.src.evaluate import evaluate
import polars as pl
from pathlib import Path
from dotenv import load_dotenv
import dagshub
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
    preprocessor: Pipeline,
    model_name: str,
    model: Any,
    hyperparams: Dict
) -> None:
    
    # load_dotenv()
    # os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_TRACKING_USERNAME")
    # os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TRACKING_PASSWORD")
    # Get raw data again
    data = load_raw_from_local(config=config)
    training_data, testing_data = data["train"], data["test"]
    X_train = training_data.drop(config["columns"]["target"])
    y_train = training_data[config["columns"]["target"]]
    X_test = testing_data.drop(config["columns"]["target"])
    y_test = testing_data[config["columns"]["target"]]
    
    # Start MLflow run
    # mlflow.set_tracking_uri(Path(os.path.join(config["path"]["root"], config["path"]["models"], config["path"]["mlruns"])))
    dagshub.init(repo_owner='Hg03', repo_name='minimal_mlops_dagshub', mlflow=True)
    # dagshub.init(url="https://dagshub.com/Hg03/minimal_mlops_dagshub", mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/Hg03/minimal_mlops_dagshub.mlflow")
    mlflow.set_experiment(config[model_name]["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name=f"{model.__class__.__name__}_training"):
        
        # Perform tuning
        tuner = GridSearchCV(
            estimator=model,
            param_grid=hyperparams,
            scoring='neg_mean_squared_error'
        )
        
        # Append the preprocessor
        model = make_pipeline(preprocessor, tuner)
        # Train model on full training set and make predictions
        model.fit(X_train, y_train)
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)
        
        # signature
        model_signature = infer_signature(model.steps[0][1].transform(X_train), pl.DataFrame(data={"trip_price": test_predictions.tolist()}))

        # log metrics and hyperparams
        metrics = evaluate(tuner=tuner, y_train=y_train, train_predictions=train_predictions, y_test=y_test, test_predictions=test_predictions)
        mlflow.log_metrics(metrics)
        mlflow.log_params(model.steps[1][1].best_params_)
        # Log the model
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model", signature=model_signature)
        pickle.dump(model, open(os.path.join(config["path"]["root"], config["path"]["models"], config[model_name]["name"]), "wb"))