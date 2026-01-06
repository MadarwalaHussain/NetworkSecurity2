import dagshub
from urllib.parse import urlparse
import mlflow
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression
from network_security.utils.ml_utils.metric.classification_metric import get_classification_score
from network_security.utils.main_untils import load_numpy_array_data, evaluate_models
from network_security.utils.main_untils import save_object, load_object
from network_security.utils.ml_utils.model.estimator import NetworkModel
from network_security.entity.config_entity import ModelTrainerConfig
from network_security.entity.artificat_entity import DataTransformationArtifact, ModelTrainerArtifact
from network_security.logging.logger import logging
from network_security.exceptions.exception import NetworkSecurityException
import sys
import os
import fix_ssl
fix_ssl.apply_ssl_fix()

# Set ALL environment variables BEFORE importing any libraries
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/MadarwalaHussain/NetworkSecurity2.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "MadarwalaHussain"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "db926fa2f90af5a877aa3d843baf4a9555d30c6c"
os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"  # NEW: Critical for MLflow
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""


os.environ["HTTPX_VERIFY"] = "0"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/MadarwalaHussain/NetworkSecurity2.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "MadarwalaHussain"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "db926fa2f90af5a877aa3d843baf4a9555d30c6c"
# Create a custom httpx client with SSL verification disabled
# client = httpx.Client(verify=False)
dagshub.init(
    repo_owner='MadarwalaHussain',
    repo_name='NetworkSecurity2',
    mlflow=True
)


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def track_mlflow(self, best_model, classification_metric):
        import ssl
        import urllib3

        # Force disable SSL for this session
        ssl._create_default_https_context = ssl._create_unverified_context
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        mlflow.set_registry_uri("https://dagshub.com/MadarwalaHussain/NetworkSecurity2.mlflow")
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            f1_score = classification_metric.f1_score
            precision_score = classification_metric.precision_score
            recall_score = classification_metric.recall_score

            mlflow.log_metric("f1_score", f1_score)
            mlflow.log_metric("precision", precision_score)
            mlflow.log_metric("recall_score", recall_score)

            # First log the model without registering
            mlflow.sklearn.log_model(best_model, "model")

            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                # Register the model with a STRING name (not the model object)
                model_name = f"{best_model.__class__.__name__}_NetworkSecurity"
                mlflow.sklearn.log_model(
                    best_model,
                    "model",
                    registered_model_name=model_name  # ✓ Now it's a string!
                )
            else:
                mlflow.sklearn.log_model(best_model, "model")

    def train_model(self, X_train, y_train, X_test, y_test):
        models = {
            "Random Forest": RandomForestClassifier(verbose=1),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(verbose=1),
            "Logistic Regression": LogisticRegression(verbose=1),
            "AdaBoost": AdaBoostClassifier(),
        }
        params = {
            "Decision Tree": {
                'criterion': ['gini', 'entropy', 'log_loss'],
                # 'splitter':['best','random'],
                # 'max_features':['sqrt','log2'],
            },
            "Random Forest": {
                # 'criterion':['gini', 'entropy', 'log_loss'],

                # 'max_features':['sqrt','log2',None],
                'n_estimators': [8, 16, 32, 128, 256]
            },
            "Gradient Boosting": {
                # 'loss':['log_loss', 'exponential'],
                'learning_rate': [.1, .01, .05, .001],
                'subsample': [0.6, 0.7, 0.75, 0.85, 0.9],
                # 'criterion':['squared_error', 'friedman_mse'],
                # 'max_features':['auto','sqrt','log2'],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            "Logistic Regression": {},
            "AdaBoost": {
                'learning_rate': [.1, .01, .001],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            }

        }

        model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                             models=models, param=params)

        # To get best model score from dict
        best_model_score = max(sorted(model_report.values()))

        # To get best model name from dict

        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]
        best_model = models[best_model_name]
        y_train_pred = best_model.predict(X_train)

        classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
        logging.info(f"Classification Metrics: {classification_train_metric}")

        # Track the mlflow
        self.track_mlflow(best_model, classification_train_metric)

        y_test_pred = best_model.predict(X_test)
        classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)
        logging.info(f"Classification Metrics: {classification_test_metric}")

        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)
        network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path, obj=network_model)
        
        # TODO: using boto3 we can push the model to aws 
        save_object('final_model/model.pkl', best_model)
        # Model Trainer Artifact
        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric
        )
        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            # loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model_trainer_artifact = self.train_model(x_train, y_train, x_test, y_test)
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
