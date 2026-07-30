# K-Nearest Neighbors classifier architecture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import numpy as np
import pickle
import time


class KNNClassifierArchitecture:

    architecture_name = "KNN Classifier"
    architecture_description = "K-Nearest Neighbors classifier based on organoid features. For further information, see https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html"
    train_data_type = "features"

    config_parameters = {
        "n_neighbors": "int",
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "leaf_size": "int",
        "p": "float",
        "metric": "str",
        "n_jobs": "int", # -1 for None, otherwise positive integers
        "normalize_features": "bool",
        "cv_folds": "int",
    }

    default_config = {
        "n_neighbors": 5,
        "weights": "uniform",
        "algorithm": "auto",
        "leaf_size": 30,
        "p": 2,
        "metric": "minkowski",
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
        
        self.n_neighbors = config['n_neighbors']
        self.weights = config['weights']
        self.algorithm = config['algorithm']
        self.metric = config['metric']
        self.leaf_size = config['leaf_size']
        self.p = config['p']
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

        self.model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            p=self.p,
            metric=self.metric,
            n_jobs=self.n_jobs
        )
        
        if self.cv_folds > 0:
            cv_scores = cross_val_score(self.model, X, y, cv=self.cv_folds, scoring='accuracy')
            print(f"Cross-validation accuracy: {cv_scores.mean():.4f} +- {cv_scores.std():.4f}")
        
        print(X)
        print(y)
        self.model.fit(X, y)
        
        training_time = time.time() - start_time
        
        self.training_results = {
            'training_time': training_time,
            'n_samples': len(training_data),
            'n_features': X.shape[1] if X.ndim > 1 else 1,
            'n_neighbors_used': self.n_neighbors,
            'feature_normalization': self.normalize_features,
            'distance_metric': self.metric,
            'weight_function': self.weights
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

        self.n_neighbors = self.config['n_neighbors']
        self.weights = self.config['weights']
        self.algorithm = self.config['algorithm']
        self.metric = self.config['metric']
        self.leaf_size = self.config['leaf_size']
        self.p = self.config['p']
        self.n_jobs = self.config['n_jobs'] if self.config['n_jobs'] != -1 else None
        self.normalize_features = self.config['normalize_features']
        self.cv_folds = self.config.get('cv_folds', 0)

        print(f"Model loaded from: {dirpath}")
