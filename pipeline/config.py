import os
from dataclasses import dataclass
from typing import List, Union


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    k_neighbors: int = 1
    weights: str = 'distance'
    model_name: str = 'knn_model'
    metric: str = 'minkowski'  # Default distance metric
    p: int = 2  # Power parameter for Minkowski metric (p=1 for Manhattan, p=2 for Euclidean)


@dataclass 
class DataConfig:
    """Data configuration parameters."""
    base_dir: str = "data"
    output_dir: str = "output"
    training_folder_v2: str = None
    training_folder_v3: str = None
    training_folder_synthetic: str = None
    testing_folder: str = None
    peaklist_folder: str = None
    additional_training_sources: List[str] = None  # Can be folders or individual files
    custom_test_files: List[str] = None
    
    def __post_init__(self):
        """Initialize default paths after creation."""
        if self.training_folder_v2 is None:
            self.training_folder_v2 = os.path.join(self.base_dir, "DOM_training_set_ver2")
        if self.training_folder_v3 is None:
            self.training_folder_v3 = os.path.join(self.base_dir, "DOM_training_set_ver3")
        if self.training_folder_synthetic is None:
            self.training_folder_synthetic = os.path.join(self.base_dir, "synthetic_data")
        if self.testing_folder is None:
            self.testing_folder = os.path.join(self.base_dir, "DOM_testing_set")
        if self.peaklist_folder is None:
            self.peaklist_folder = os.path.join(self.base_dir, "DOM_testing_set_Peaklists")
        if self.additional_training_sources is None:
            self.additional_training_sources = []
        if self.custom_test_files is None:
            self.custom_test_files = []


@dataclass
class PipelineConfig:
    """Pipeline configuration parameters."""
    version_name: str
    training_folders: Union[str, List[str]]
    model_path: str
    result_dir: str
    k_neighbors: int = 1
    metric: str = 'minkowski'  # Distance metric
    p: int = 2  # Power parameter for Minkowski metric
    testing_folder: str = None
    peaklist_folder: str = None
    custom_test_files: List[str] = None
    additional_training_sources: List[str] = None
    use_ensemble: bool = True  # Whether to train separate models per dataset
    
    def __post_init__(self):
        """Handle post-initialization logic."""
        # Set default p value for minkowski metric if not specified
        if self.metric == 'minkowski' and not hasattr(self, 'p'):
            self.p = 2
        # For non-minkowski metrics, p parameter is not used
        elif self.metric != 'minkowski':
            self.p = None
    
    def get_training_folders_list(self):
        """Get training folders as a list."""
        if isinstance(self.training_folders, str):
            return [self.training_folders]
        return self.training_folders
    
    def get_all_training_sources(self, data_config):
        """Get all training sources (folders and files) including additional ones."""
        sources = self.get_training_folders_list()
        # Add global additional training sources
        if data_config.additional_training_sources:
            sources.extend(data_config.additional_training_sources)
        # Add per-config additional training sources
        if self.additional_training_sources:
            sources.extend(self.additional_training_sources)
        return sources
    
    def get_test_files_or_folder(self, data_config):
        """Get custom test files or default testing folder."""
        if self.custom_test_files:
            return self.custom_test_files
        elif self.testing_folder:
            return self.testing_folder
        else:
            return data_config.testing_folder
    
    def get_peaklist_folder(self, data_config):
        """Get peaklist folder (custom or default)."""
        return self.peaklist_folder or data_config.peaklist_folder
    
    def get_model_paths_for_ensemble(self, training_sources):
        if not self.use_ensemble or len(training_sources) <= 1:
            # If not using ensemble or only one source, return single model path
            source_name = os.path.basename(training_sources[0]) if isinstance(training_sources, list) and len(training_sources) > 0 else "single"
            return [(source_name, self.model_path)]
        
        # Generate separate model path for each training source
        model_dir = os.path.dirname(self.model_path)
        base_model_name = os.path.basename(self.model_path).replace('.joblib', '')
        
        model_paths = []
        for i, source in enumerate(training_sources):
            # Get a clean source name
            if os.path.isdir(source):
                source_name = os.path.basename(source)
            else:
                source_name = os.path.basename(source).replace('.xlsx', '').replace('.csv', '')
            
            # Create model path with source identifier
            model_path = os.path.join(model_dir, f"{base_model_name}_{source_name}.joblib")
            model_paths.append((source_name, model_path))
        
        return model_paths


class ConfigManager:
    """Manages pipeline configurations."""
    
    def __init__(self):
        self.data_config = DataConfig()
        self.model_config = ModelConfig()
    
    def get_standard_configs(self):
        configs = []
        
        # Define different distance metrics and K values to test
        distance_configs = [
            {"name": "Euclidean", "metric": "minkowski", "p": 2},
            {"name": "Manhattan", "metric": "minkowski", "p": 1},
            # {"name": "Cosine", "metric": "cosine", "p": None}
        ]
        
        k_values = [ 1, 3]
        
        # Generate configurations for each combination of dataset, distance metric, and K value
        base_configs = [
            {
                "name": "Model-L1",
                "training_folders": self.data_config.training_folder_v2,
                "base_path": "L1"
            },
            {
                "name": "Model-L3",
                "training_folders": self.data_config.training_folder_v3,
                "base_path": "L3"
            },
            {
                "name": "Model-L1-L3",
                "training_folders": [
                    self.data_config.training_folder_v2,
                    self.data_config.training_folder_v3
                ],
                "base_path": "combined"
            },
            ## Synthetic with combiend
            {
                "name": "Model-Synthetic",
                "training_folders": [
                    self.data_config.training_folder_v2,
                    self.data_config.training_folder_v3,
                    self.data_config.training_folder_synthetic
                ],
                "base_path": "synthetic_combined"
            }
        ]
        
        # Create configurations for each combination
        for base_config in base_configs:
            for k in k_values:
                for dist_config in distance_configs:
                    version_name = f"{base_config['name']}_K{k}_{dist_config['name']}"
                    model_filename = f"knn_model_{base_config['name']}_K{k}_{dist_config['name']}.joblib"
                    result_subdir = f"K{k}_{dist_config['name']}"
                    
                    config_params = {
                        "version_name": version_name,
                        "training_folders": base_config["training_folders"],
                        "model_path": os.path.join("models", model_filename),
                        "result_dir": os.path.join(self.data_config.output_dir, base_config["base_path"], result_subdir, "test_results"),
                        "k_neighbors": k,
                        "metric": dist_config["metric"],
                        "testing_folder": self.data_config.testing_folder,
                        "peaklist_folder": self.data_config.peaklist_folder
                    }
                    
                    # Add p parameter only for minkowski metric
                    if dist_config["p"] is not None:
                        config_params["p"] = dist_config["p"]
                    
                    configs.append(PipelineConfig(**config_params))
        
        return configs
    
    def create_custom_config(self, version_name, training_folders, model_path, result_dir, 
                           k_neighbors=None, metric=None, p=None, testing_folder=None, 
                           peaklist_folder=None, custom_test_files=None, additional_training_sources=None,
                           use_ensemble=True):
        """
        Create a custom pipeline configuration.
        
        """
        config_params = {
            'version_name': version_name,
            'training_folders': training_folders,
            'additional_training_sources': additional_training_sources,
            'model_path': model_path,
            'result_dir': result_dir,
            'k_neighbors': k_neighbors or self.model_config.k_neighbors,
            'metric': metric or self.model_config.metric,
            'testing_folder': testing_folder,
            'peaklist_folder': peaklist_folder,
            'custom_test_files': custom_test_files,
            'use_ensemble': use_ensemble
        }
        
        # Add p parameter only if specified or if using minkowski metric
        if p is not None:
            config_params['p'] = p
        elif (metric or self.model_config.metric) == 'minkowski':
            config_params['p'] = self.model_config.p
            
        return PipelineConfig(**config_params)
    
    def update_data_config(self, **kwargs):
        """Update data configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.data_config, key):
                setattr(self.data_config, key, value)
    
    def update_model_config(self, **kwargs):
        """Update model configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.model_config, key):
                setattr(self.model_config, key, value)
    
    def add_training_source(self, source_path):
        """Add an additional training source (folder or file) to the global configuration."""
        if source_path not in self.data_config.additional_training_sources:
            self.data_config.additional_training_sources.append(source_path)
    
    def add_training_sources(self, source_paths):
        """Add multiple additional training sources (folders or files)."""
        for source_path in source_paths:
            self.add_training_source(source_path)
    
    def add_training_folder(self, folder_path):
        """Add an additional training folder (backward compatibility)."""
        self.add_training_source(folder_path)
    
    def add_training_folders(self, folder_paths):
        """Add multiple additional training folders (backward compatibility)."""
        self.add_training_sources(folder_paths)
    
    def add_training_file(self, file_path):
        """Add an additional training file."""
        self.add_training_source(file_path)
    
    def add_training_files(self, file_paths):
        """Add multiple additional training files."""
        self.add_training_sources(file_paths)
    
    def set_custom_test_files(self, test_files):
        """Set custom test files to use instead of the default testing folder."""
        self.data_config.custom_test_files = test_files if isinstance(test_files, list) else [test_files]
    
    def add_custom_test_file(self, test_file):
        """Add a custom test file."""
        if test_file not in self.data_config.custom_test_files:
            self.data_config.custom_test_files.append(test_file)
    
    def clear_additional_training_sources(self):
        """Clear all additional training sources."""
        self.data_config.additional_training_sources = []
    
    def clear_additional_training_folders(self):
        """Clear all additional training folders (backward compatibility)."""
        self.clear_additional_training_sources()
    
    def clear_custom_test_files(self):
        """Clear all custom test files."""
        self.data_config.custom_test_files = []