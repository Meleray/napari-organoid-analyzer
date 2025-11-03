# Random Forest Classifier architecture
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import numpy as np
import pickle
import time


class RandomForestClassifierArchitecture:

    architecture_name = "Random Forest Classifier"
    architecture_description = "Random Forest Classifier based on organoid features using ensemble of decision trees. For further information, see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html"
    train_data_type = "features"

    config_parameters = {
        "n_estimators": "int",
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": "int", # -1 for None, otherwise positive integers
        "min_samples_split": "float",
        "min_samples_leaf": "float",
        "min_weight_fraction_leaf": "float",
        "max_features": ["sqrt", "log2", "None", "float"],
        "max_features_float": "float",  # For explicit float max_features value
        "max_leaf_nodes": "int",  # -1 for None, otherwise positive integers
        "min_impurity_decrease": "float",
        "bootstrap": "bool",
        "oob_score": "bool",
        "n_jobs": "int", # -1 for None, otherwise positive integers
        "random_state": "int", # -1 for None, otherwise positive integers
        "verbose": "int",
        "warm_start": "bool",
        "class_weight": ["balanced", "balanced_subsample", "None"],
        "ccp_alpha": "float",
        "max_samples": "float", # -1 for None, otherwise positive integers
        "normalize_features": "bool",
        "cv_folds": "int",
    }

    default_config = {
        "n_estimators": 100,
        "criterion": "gini",
        "max_depth": -1,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "min_weight_fraction_leaf": 0,
        "max_features": "sqrt",
        "max_features_float": 1.0,
        "max_leaf_nodes": -1,
        "min_impurity_decrease": 0.0,
        "bootstrap": True,
        "oob_score": False,
        "n_jobs": -1,
        "random_state": -1,
        "verbose": 0,
        "warm_start": False,
        "class_weight": "None",
        "ccp_alpha": 0.0,
        "max_samples": -1,
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
        self.criterion = config['criterion']
        self.max_depth = config['max_depth'] if config['max_depth'] != -1 else None
        self.min_samples_split = self.convert_to_int(config['min_samples_split'])
        self.min_samples_leaf = self.convert_to_int(config['min_samples_leaf'])
        self.min_weight_fraction_leaf = config['min_weight_fraction_leaf']

        max_features_param = config['max_features']
        if max_features_param == "float":
            self.max_features = self.convert_to_int(config['max_features_float'])
        elif max_features_param == "None":
            self.max_features = None
        else:
            self.max_features = max_features_param

        self.max_leaf_nodes = config['max_leaf_nodes'] if config['max_leaf_nodes'] != -1 else None
        self.min_impurity_decrease = config['min_impurity_decrease']
        self.bootstrap = config['bootstrap']
        self.oob_score = config['oob_score']
        self.n_jobs = config['n_jobs'] if config['n_jobs'] != -1 else None
        self.random_state = config['random_state'] if config['random_state'] != -1 else None
        self.verbose = config['verbose']
        self.warm_start = config['warm_start']
        self.class_weight = config['class_weight'] if config['class_weight'] != "None" else None
        self.ccp_alpha = config['ccp_alpha']
        self.max_samples = self.convert_to_int(config['max_samples']) if config['max_samples'] != -1 else None
        self.normalize_features = config['normalize_features']
        self.cv_folds = config['cv_folds']

    @staticmethod
    def convert_to_int(value):
        """Convert a value to int if it is a float with no decimal part."""
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value
    
    def train(self, training_data, training_labels):

        print(f"Training {self.architecture_name} with {len(training_data)} samples...")
        start_time = time.time()

        X = np.array(training_data)
        y = np.array(training_labels)

        if self.normalize_features and self.scaler:
            X = self.scaler.fit_transform(X)

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
            warm_start=self.warm_start,
            class_weight=self.class_weight,
            ccp_alpha=self.ccp_alpha,
            max_samples=self.max_samples
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
            'criterion': self.criterion,
            'max_depth': self.max_depth,
            'bootstrap': self.bootstrap,
            'oob_score_value': self.model.oob_score_ if self.oob_score and hasattr(self.model, 'oob_score_') else None,
            'feature_importances': self.model.feature_importances_.tolist() if hasattr(self.model, 'feature_importances_') else None
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

        self.n_estimators = self.config['n_estimators']
        self.criterion = self.config['criterion']
        self.max_depth = self.config['max_depth'] if self.config['max_depth'] != -1 else None
        self.min_samples_split = self.convert_to_int(self.config['min_samples_split'])
        self.min_samples_leaf = self.convert_to_int(self.config['min_samples_leaf'])
        self.min_weight_fraction_leaf = self.config['min_weight_fraction_leaf']

        max_features_param = self.config['max_features']
        if max_features_param == "float":
            self.max_features = self.convert_to_int(self.config['max_features_float'])
        elif max_features_param == "None":
            self.max_features = None
        else:
            self.max_features = max_features_param

        self.max_leaf_nodes = self.config['max_leaf_nodes'] if self.config['max_leaf_nodes'] != -1 else None
        self.min_impurity_decrease = self.config['min_impurity_decrease']
        self.bootstrap = self.config['bootstrap']
        self.oob_score = self.config['oob_score']
        self.n_jobs = self.config['n_jobs'] if self.config['n_jobs'] != -1 else None
        self.random_state = self.config['random_state'] if self.config['random_state'] != -1 else None
        self.verbose = self.config['verbose']
        self.warm_start = self.config['warm_start']
        self.class_weight = self.config['class_weight'] if self.config['class_weight'] != "None" else None
        self.ccp_alpha = self.config['ccp_alpha']
        self.max_samples = self.convert_to_int(self.config['max_samples']) if self.config['max_samples'] != -1 else None
        self.normalize_features = self.config['normalize_features']
        self.cv_folds = self.config.get('cv_folds', 0)
        print(f"Model loaded from: {dirpath}")
