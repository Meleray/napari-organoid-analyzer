# Support Vector Machine classifier architecture
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import numpy as np
import pickle
import time


class SVCClassifierArchitecture:

    architecture_name = "SVC Classifier"
    architecture_description = "Support Vector Machine classifier based on organoid features. For further information, see https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html"
    train_data_type = "features"

    config_parameters = {
        "C": "float",
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "degree": "int",
        "gamma": ["scale", "auto", "float"],
        'gamma-float': "float", # For explicit float gamma value
        "coef0": "float",
        "shrinking": "bool",
        "probability": "bool",
        "tol": "float",
        "cache_size": "float",
        "class_weight": ["balanced", "None"],
        "max_iter": "int",
        "decision_function_shape": ["ovo", "ovr"],
        "break_ties": "bool",
        "random_state": "int", # -1 means no random state
        "normalize_features": "bool",
        "cv_folds": "int",
    }

    default_config = {
        "C": 1.0,
        "kernel": "rbf",
        "degree": 3,
        "gamma": "scale",
        'gamma-float': 0.0,  # Doesnt matter in default config since 'scale' is used
        "coef0": 0.0,
        "shrinking": True,
        "probability": False,
        "tol": 0.001,
        "cache_size": 200,
        "class_weight": "None",
        "max_iter": -1,
        "decision_function_shape": "ovr",
        "break_ties": False,
        "random_state": -1,
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
        
        self.C = config['C']
        self.kernel = config['kernel']
        self.degree = config['degree']
        self.gamma = config['gamma'] if not config["gamma"] == "float" else config['gamma-float']
        self.coef0 = config['coef0']
        self.shrinking = config['shrinking']
        self.probability = config['probability']
        self.tol = config['tol']
        self.cache_size = config['cache_size']
        self.class_weight = config['class_weight'] if config['class_weight'] != "None" else None
        self.max_iter = config['max_iter']
        self.decision_function_shape = config['decision_function_shape']
        self.break_ties = config['break_ties']
        self.random_state = config['random_state'] if config['random_state'] != -1 else None
        self.normalize_features = config['normalize_features']
        self.cv_folds = config['cv_folds']

    def train(self, training_data, training_labels):

        print(f"Training {self.architecture_name} with {len(training_data)} samples...")
        start_time = time.time()

        X = np.array(training_data)
        y = np.array(training_labels)

        if self.normalize_features and self.scaler:
            X = self.scaler.fit_transform(X)

        self.model = SVC(
            C=self.C,
            kernel=self.kernel,
            degree=self.degree,
            gamma=self.gamma,
            coef0=self.coef0,
            shrinking=self.shrinking,
            probability=self.probability,
            tol=self.tol,
            cache_size=self.cache_size,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            decision_function_shape=self.decision_function_shape,
            break_ties=self.break_ties,
            random_state=self.random_state
        )
        print(X[0])
        print(y[0])
        
        if self.cv_folds > 0:
            cv_scores = cross_val_score(self.model, X, y, cv=self.cv_folds, scoring='accuracy')
            print(cv_scores)
            print(f"Cross-validation accuracy: {cv_scores.mean():.4f} +- {cv_scores.std():.4f}")
        
        self.model.fit(X, y)
        
        training_time = time.time() - start_time
        
        self.training_results = {
            'training_time': training_time,
            'n_samples': len(training_data),
            'n_features': X.shape[1] if X.ndim > 1 else 1,
            'C': self.C,
            'kernel': self.kernel,
            'gamma': self.gamma,
            'feature_normalization': self.normalize_features,
            'n_support_vectors': self.model.n_support_.sum() if hasattr(self.model, 'n_support_') else 0,
            'support_vector_ratio': self.model.n_support_.sum() / len(training_data) if hasattr(self.model, 'n_support_') else 0
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

        self.C = self.config['C']
        self.kernel = self.config['kernel']
        self.degree = self.config['degree']
        self.gamma = self.config['gamma'] if not self.config["gamma"] == "float" else self.config['gamma-float']
        self.coef0 = self.config['coef0']
        self.shrinking = self.config['shrinking']
        self.probability = self.config['probability']
        self.tol = self.config['tol']
        self.cache_size = self.config['cache_size']
        self.class_weight = self.config['class_weight'] if self.config['class_weight'] != "None" else None
        self.max_iter = self.config['max_iter']
        self.decision_function_shape = self.config['decision_function_shape']
        self.break_ties = self.config['break_ties']
        self.random_state = self.config['random_state'] if self.config['random_state'] != -1 else None  
        self.normalize_features = self.config['normalize_features']
        self.cv_folds = self.config.get('cv_folds', 0)

        print(f"Model loaded from: {dirpath}")