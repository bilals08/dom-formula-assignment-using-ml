import os
import joblib
from sklearn.neighbors import KNeighborsClassifier
from pipeline.logger import get_logger
from pipeline.utils.utils import ensure_dir


class ModelTrainer:
    
    def __init__(self, k_neighbors=1, metric='minkowski', p=2):
        self.k_neighbors = k_neighbors
        self.metric = metric
        self.p = p
        self.model = None
        self.models = []  # For ensemble models
        self.logger = get_logger("ModelTrainer")
    
    def train_model(self, training_data):
        metric_name = f"{self.metric}"
        if self.metric == 'minkowski':
            if self.p == 1:
                metric_name = "Manhattan (p=1)"
            elif self.p == 2:
                metric_name = "Euclidean (p=2)"
            else:
                metric_name = f"Minkowski (p={self.p})"
        
        self.logger.info(f"Training KNN model with k={self.k_neighbors}, metric={metric_name}...")
        
        # Prepare features and labels
        X = training_data['Mass_Daltons'].values.reshape(-1, 1).astype(float).round(5)
        y = training_data['Formula']
        
        # Create and train model with specified distance metric
        model_params = {
            'n_neighbors': self.k_neighbors,
            'weights': 'distance',
            'metric': self.metric
        }
        
        # Add p parameter for Minkowski metric
        if self.metric == 'minkowski':
            model_params['p'] = self.p
        
        self.model = KNeighborsClassifier(**model_params)
        self.model.fit(X, y)
        
        self.logger.info(f"Model trained on {len(training_data)} samples with {metric_name}")
        return self.model
    
    def save_model(self, model_path):
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        # Ensure the directory exists
        ensure_dir(os.path.dirname(model_path))
        
        # Save model
        joblib.dump(self.model, model_path)
        self.logger.info(f"Model saved to: {model_path}")
    
    def load_model(self, model_path):
        if not os.path.exists(model_path):
            self.logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = joblib.load(model_path)
        self.logger.info(f"Loaded model from: {model_path}")
        return self.model
    
    def train_and_save(self, training_data, model_path, force_retrain=False):

        
        self.train_model(training_data)
        self.save_model(model_path)
    
    def train_multiple_models(self, separate_training_data):
        metric_name = f"{self.metric}"
        if self.metric == 'minkowski':
            if self.p == 1:
                metric_name = "Manhattan (p=1)"
            elif self.p == 2:
                metric_name = "Euclidean (p=2)"
            else:
                metric_name = f"Minkowski (p={self.p})"
        
        self.logger.info(f"Training {len(separate_training_data)} separate KNN models with k={self.k_neighbors}, metric={metric_name}...")
        
        self.models = []
        
        for source_name, training_data in separate_training_data:
            # Prepare features and labels
            X = training_data['Mass_Daltons'].values.reshape(-1, 1).astype(float).round(5)
            y = training_data['Formula']
            
            # Create and train model with specified distance metric
            model_params = {
                'n_neighbors': self.k_neighbors,
                'weights': 'distance',
                'metric': self.metric
            }
            
            # Add p parameter for Minkowski metric
            if self.metric == 'minkowski':
                model_params['p'] = self.p
            
            model = KNeighborsClassifier(**model_params)
            model.fit(X, y)
            
            self.models.append((source_name, model))
            self.logger.info(f"Model trained on '{source_name}' dataset: {len(training_data)} samples with {metric_name}")
        
        return self.models
    
    def save_multiple_models(self, model_paths):
        """
        Save multiple trained models to disk.
        
        """
        if not self.models:
            raise ValueError("No models to save. Train models first.")
        
        if len(model_paths) != len(self.models):
            raise ValueError(f"Number of model paths ({len(model_paths)}) doesn't match number of models ({len(self.models)})")
        
        # Create a mapping of source names to model paths
        path_mapping = {source_name: path for source_name, path in model_paths}
        
        for source_name, model in self.models:
            if source_name not in path_mapping:
                raise ValueError(f"No model path found for source '{source_name}'")
            
            model_path = path_mapping[source_name]
            
            # Ensure the directory exists
            ensure_dir(os.path.dirname(model_path))
            
            # Save model
            joblib.dump(model, model_path)
            self.logger.info(f"Model for '{source_name}' saved to: {model_path}")
    
    def load_multiple_models(self, model_paths):
        """
        Load multiple trained models from disk.
        
        """
        self.models = []
        
        for source_name, model_path in model_paths:
            if not os.path.exists(model_path):
                self.logger.error(f"Model file not found: {model_path}")
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            model = joblib.load(model_path)
            self.models.append((source_name, model))
            self.logger.info(f"Loaded model for '{source_name}' from: {model_path}")
        
        return self.models
    
    def train_and_save_multiple(self, separate_training_data, model_paths, force_retrain=False):
        """
        Train and save multiple models, with option to skip if models already exist.

        """
        self.train_multiple_models(separate_training_data)
        self.save_multiple_models(model_paths)
    
    def get_model(self):
        """
        Get the current model instance.
        
        """
        return self.model
    
    def get_models(self):
        """
        Get all models (for ensemble predictions).
        """
        return self.models