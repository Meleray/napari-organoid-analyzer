# Gaussian Process Classifier architecture
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import numpy as np
import pickle
import time


class GaussianProcessClassifierArchitecture:

    architecture_name = "Gaussian Process Classifier"
    architecture_description = "Gaussian Process Classifier based on organoid features with Laplace approximation. For further information, see https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html"
    train_data_type = "features"

    config_parameters = {
        "optimizer": ["fmin_l_bfgs_b", "None"],
        "n_restarts_optimizer": "int",
        "max_iter_predict": "int",
        "warm_start": "bool",
        "copy_X_train": "bool",
        "random_state": "int", # -1 for None
        "multi_class": ["one_vs_rest", "one_vs_one"],
        "n_jobs": "int", # -1 for all available cores
        "normalize_features": "bool",
        "cv_folds": "int",
    }

    default_config = {
        "optimizer": "fmin_l_bfgs_b",
        "n_restarts_optimizer": 0,
        "max_iter_predict": 100,
        "warm_start": False,
        "copy_X_train": True,
        "random_state": -1,
        "multi_class": "one_vs_rest",
        "n_jobs": -1,
        "normalize_features": True,
        "cv_folds": 0,
    }

    def __init__(self, config):

        for key, value in self.default_config.items():
            if key not in config:
                config[key] = value

        self.config = config
        self.model = None
        self.scaler = StandardScaler() if config.get('normalize_features', True) else None
        self.training_results = {}
        
        self.optimizer = config['optimizer'] if config['optimizer'] != "None" else None
        self.n_restarts_optimizer = config['n_restarts_optimizer']
        self.max_iter_predict = config['max_iter_predict']
        self.warm_start = config['warm_start']
        self.copy_X_train = config['copy_X_train']
        self.random_state = config['random_state'] if config['random_state'] != -1 else None
        self.multi_class = config['multi_class']
        self.n_jobs = config['n_jobs'] if config['n_jobs'] != -1 else None
        self.normalize_features = config['normalize_features']
        self.cv_folds = config['cv_folds']

    def train(self, training_data, training_labels):

        print(f"Training {self.architecture_name} with {len(training_data)} samples...")
        start_time = time.time()

        X = np.array(training_data)
        y = np.array(training_labels)

        if self.normalize_features and self.scaler:
            X = self.scaler.fit_transform(X)

        # For now kernel is fixed to default. TODO: allow kernel configuration
        kernel = 1.0 * RBF(1.0)

        self.model = GaussianProcessClassifier(
            kernel=kernel,
            optimizer=self.optimizer,
            n_restarts_optimizer=self.n_restarts_optimizer,
            max_iter_predict=self.max_iter_predict,
            warm_start=self.warm_start,
            copy_X_train=self.copy_X_train,
            random_state=self.random_state,
            multi_class=self.multi_class,
            n_jobs=self.n_jobs
        )
        
        if self.cv_folds > 0:
            cv_scores = cross_val_score(self.model, X, y, cv=self.cv_folds, scoring='accuracy')
            print(f"Cross-validation accuracy: {cv_scores.mean():.4f} +- {cv_scores.std():.4f}")
        
        self.model.fit(X, y)
        
        training_time = time.time() - start_time
        
        self.training_results = {
            'training_time': training_time,
            'n_samples': len(training_data),
            'n_features': X.shape[1] if X.ndim > 1 else 1,
            'feature_normalization': self.normalize_features,
            'multi_class_strategy': self.multi_class,
            'n_restarts_optimizer': self.n_restarts_optimizer,
            'log_marginal_likelihood': self.model.log_marginal_likelihood_value_ if hasattr(self.model, 'log_marginal_likelihood_value_') else None
        }
        
        print(f"Training completed in {training_time:.2f} seconds")
        return self.training_results

    def predict(self, data):
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        X = np.array(data)
        
        if self.normalize_features and self.scaler:
            X = self.scaler.transform(X)
        
        return self.model.predict(X)

    def save_model(self, dirpath):
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'config': self.config,
            'training_results': self.training_results,
            'architecture_name': self.architecture_name
        }

        with open(dirpath / "checkpoint.pkl", 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to: {dirpath}")

    def load_model(self, dirpath):
        
        try:
            with open(dirpath / "checkpoint.pkl", 'rb') as f:
                model_data = pickle.load(f)
        except FileNotFoundError:
            raise ValueError(f"No model found at {dirpath}. Please check the path.")
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.config = model_data['config']
        self.training_results = model_data.get('training_results', {})
        
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value

        self.optimizer = self.config['optimizer'] if self.config['optimizer'] != "None" else None
        self.n_restarts_optimizer = self.config['n_restarts_optimizer']
        self.max_iter_predict = self.config['max_iter_predict']
        self.warm_start = self.config['warm_start']
        self.copy_X_train = self.config['copy_X_train']
        self.random_state = self.config['random_state'] if self.config['random_state'] != -1 else None
        self.multi_class = self.config['multi_class']
        self.n_jobs = self.config['n_jobs'] if self.config['n_jobs'] != -1 else None
        self.normalize_features = self.config['normalize_features']
        self.cv_folds = self.config.get('cv_folds', 0)

        print(f"Model loaded from: {dirpath}")
