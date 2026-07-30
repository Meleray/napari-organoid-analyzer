# Multi-layer Perceptron Regressor architecture
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import numpy as np
import pickle
import time
import ast


class MLPRegressorArchitecture:

    architecture_name = "MLP Regressor"
    architecture_description = "Multi-layer Perceptron Regressor based on organoid features using neural networks. For further information, see https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html"
    train_data_type = "features"

    config_parameters = {
        "hidden_layer_sizes_str": "str",  # String representation of hidden layer sizes. 
        "activation": ["identity", "logistic", "tanh", "relu"],
        "solver": ["lbfgs", "sgd", "adam"],
        "alpha": "float",
        "batch_size": "int", # -1 for 'auto', otherwise positive integers
        "learning_rate": ["constant", "invscaling", "adaptive"],
        "learning_rate_init": "float",
        "power_t": "float",
        "max_iter": "int",
        "shuffle": "bool",
        "random_state": "int", # -1 means no random state
        "tol": "float",
        "verbose": "bool",
        "warm_start": "bool",
        "momentum": "float",
        "nesterovs_momentum": "bool",
        "early_stopping": "bool",
        "validation_fraction": "float",
        "beta_1": "float",
        "beta_2": "float",
        "epsilon": "float",
        "n_iter_no_change": "int",
        "max_fun": "int",
        "normalize_features": "bool",
        "cv_folds": "int",
    }

    default_config = {
        "hidden_layer_sizes_str": "(100,)",  
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.0001,
        "batch_size": -1, # -1 means 'auto'
        "learning_rate": "constant",
        "learning_rate_init": 0.001,
        "power_t": 0.5,
        "max_iter": 200,
        "shuffle": True,
        "random_state": -1, # -1 means no random state
        "tol": 0.0001,
        "verbose": False,
        "warm_start": False,
        "momentum": 0.9,
        "nesterovs_momentum": True,
        "early_stopping": False,
        "validation_fraction": 0.1,
        "beta_1": 0.9,
        "beta_2": 0.999,
        "epsilon": 1e-8,
        "n_iter_no_change": 10,
        "max_fun": 15000,
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
                
        self.hidden_layer_sizes = self.parse_hidden_layer_sizes(config['hidden_layer_sizes_str'])
        self.activation = config['activation']
        self.solver = config['solver']
        self.alpha = config['alpha']
        self.batch_size = config['batch_size'] if config['batch_size'] != -1 else 'auto'
        self.learning_rate = config['learning_rate']
        self.learning_rate_init = config['learning_rate_init']
        self.power_t = config['power_t']
        self.max_iter = config['max_iter']
        self.shuffle = config['shuffle']
        self.random_state = config['random_state'] if config['random_state'] != -1 else None
        self.tol = config['tol']
        self.verbose = config['verbose']
        self.warm_start = config['warm_start']
        self.momentum = config['momentum']
        self.nesterovs_momentum = config['nesterovs_momentum']
        self.early_stopping = config['early_stopping']
        self.validation_fraction = config['validation_fraction']
        self.beta_1 = config['beta_1']
        self.beta_2 = config['beta_2']
        self.epsilon = config['epsilon']
        self.n_iter_no_change = config['n_iter_no_change']
        self.max_fun = config['max_fun']
        self.normalize_features = config['normalize_features']
        self.cv_folds = config['cv_folds']

    @staticmethod
    def parse_hidden_layer_sizes(hidden_layer_sizes_str):
        parsed = ast.literal_eval(hidden_layer_sizes_str)
        if isinstance(parsed, (list, tuple)):
            return tuple(parsed)
        else:
            raise ValueError("hidden_layer_sizes_str must be a list or tuple")

    def train(self, training_data, training_labels):

        print(f"Training {self.architecture_name} with {len(training_data)} samples...")
        start_time = time.time()

        X = np.array(training_data)
        y = np.array(training_labels)

        if self.normalize_features and self.scaler:
            X = self.scaler.fit_transform(X)

        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            learning_rate_init=self.learning_rate_init,
            power_t=self.power_t,
            max_iter=self.max_iter,
            shuffle=self.shuffle,
            random_state=self.random_state,
            tol=self.tol,
            verbose=self.verbose,
            warm_start=self.warm_start,
            momentum=self.momentum,
            nesterovs_momentum=self.nesterovs_momentum,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            epsilon=self.epsilon,
            n_iter_no_change=self.n_iter_no_change,
            max_fun=self.max_fun
        )
        
        if self.cv_folds > 0:
            cv_scores = cross_val_score(self.model, X, y, cv=self.cv_folds, scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores)
            print(f"Cross-validation RMSE: {cv_rmse.mean():.4f} +- {cv_rmse.std():.4f}")
        
        self.model.fit(X, y)
        
        training_time = time.time() - start_time
        
        self.training_results = {
            'training_time': training_time,
            'n_samples': len(training_data),
            'n_features': X.shape[1] if X.ndim > 1 else 1,
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'feature_normalization': self.normalize_features,
            'activation': self.activation,
            'solver': self.solver,
            'n_iter': self.model.n_iter_ if hasattr(self.model, 'n_iter_') else None,
            'loss': self.model.loss_ if hasattr(self.model, 'loss_') else None,
            'best_loss': self.model.best_loss_ if hasattr(self.model, 'best_loss_') else None,
            'n_layers': self.model.n_layers_ if hasattr(self.model, 'n_layers_') else None
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

        self.hidden_layer_sizes = self.parse_hidden_layer_sizes(self.config['hidden_layer_sizes_str'])
        self.activation = self.config['activation']
        self.solver = self.config['solver']
        self.alpha = self.config['alpha']
        self.batch_size = self.config['batch_size'] if self.config['batch_size'] != -1 else 'auto'
        self.learning_rate = self.config['learning_rate']
        self.learning_rate_init = self.config['learning_rate_init']
        self.power_t = self.config['power_t']
        self.max_iter = self.config['max_iter']
        self.shuffle = self.config['shuffle']
        self.random_state = self.config['random_state'] if self.config['random_state'] != -1 else None
        self.tol = self.config['tol']
        self.verbose = self.config['verbose']
        self.warm_start = self.config['warm_start']
        self.momentum = self.config['momentum']
        self.nesterovs_momentum = self.config['nesterovs_momentum']
        self.early_stopping = self.config['early_stopping']
        self.validation_fraction = self.config['validation_fraction']
        self.beta_1 = self.config['beta_1']
        self.beta_2 = self.config['beta_2']
        self.epsilon = self.config['epsilon']
        self.n_iter_no_change = self.config['n_iter_no_change']
        self.max_fun = self.config['max_fun']
        self.normalize_features = self.config['normalize_features']
        self.cv_folds = self.config.get('cv_folds', 0)

        print(f"Model loaded from: {dirpath}")

