# AdaBoost Classifier architecture
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import numpy as np
import pickle
import time


class AdaBoostClassifierArchitecture:

    architecture_name = "AdaBoost Classifier"
    architecture_description = "AdaBoost Classifier based on organoid features using adaptive boosting with decision trees. For further information, see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html"
    train_data_type = "features"

    config_parameters = {
        "n_estimators": "int",
        "learning_rate": "float",
        "random_state": "int", # -1 for None
        "normalize_features": "bool",
        "cv_folds": "int",
    }

    default_config = {
        "n_estimators": 50,
        "learning_rate": 1.0,
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
        
        self.n_estimators = config['n_estimators']
        self.learning_rate = config['learning_rate']
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

        self.model = AdaBoostClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            random_state=self.random_state
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
            'n_estimators': self.n_estimators,
            'feature_normalization': self.normalize_features,
            'learning_rate': self.learning_rate,
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

        self.n_estimators = self.config['n_estimators']
        self.learning_rate = self.config['learning_rate']
        self.random_state = self.config['random_state'] if self.config['random_state'] != -1 else None
        self.normalize_features = self.config['normalize_features']
        self.cv_folds = self.config.get('cv_folds', 0)

        print(f"Model loaded from: {dirpath}")
