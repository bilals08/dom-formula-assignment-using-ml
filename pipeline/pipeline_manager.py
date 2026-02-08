"""
Pipeline manager module for DOM formula assignment pipeline.
Orchestrates the entire pipeline workflow.
"""

import os
import pandas as pd
from pipeline.logger import get_logger
from pipeline.data_loader import DataLoader
from pipeline.model_trainer import ModelTrainer
from pipeline.predictor import Predictor, MultiPredictor
from pipeline.evaluator import Evaluator
from pipeline.config import ConfigManager, PipelineConfig
from pipeline.utils.utils import ensure_dir
from pipeline import plotting


class PipelineManager:
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.data_loader = DataLoader()
        self.evaluator = Evaluator()
        self.logger = get_logger("PipelineManager")
    
    def run_single_pipeline(self, config: PipelineConfig, force_retrain=False):
        self.logger.info(f"Running pipeline for {config.version_name}")
        
        ensure_dir(config.result_dir)

        ## if evaluation summary already exists and not force_retrain, skip
        summary_path = os.path.join(config.result_dir, "evaluation_summary_stats.csv")
        if os.path.exists(summary_path) and not force_retrain:
            self.logger.info(f"Existing evaluation summary found at {summary_path}, skipping pipeline run for {config.version_name}")
            return pd.read_csv(summary_path)
        

        try:
            all_training_sources = config.get_all_training_sources(self.config_manager.data_config)
            self.logger.info(f"Preparing training for {len(all_training_sources)} source(s)")

            if len(all_training_sources) > 1:
                model_labelled_list = []
                for idx, source in enumerate(all_training_sources):
                    label = self._derive_model_label(source, idx)
                    model_path = self._augment_model_path(config.model_path, label)
                    self.logger.info(f"Training individual model '{label}' from source: {source}")
                    trainer = ModelTrainer(k_neighbors=config.k_neighbors)
                    training_df = self.data_loader.load_training_data(source)
                    trainer.train_and_save(training_df, model_path, force_retrain=force_retrain)
                    model = trainer.load_model(model_path)
                    model_labelled_list.append((label, model))
                predictor = MultiPredictor(model_labelled_list)
            else:
                trainer = ModelTrainer(k_neighbors=config.k_neighbors)
                training_data = self.data_loader.load_training_data(all_training_sources)
                trainer.train_and_save(training_data, config.model_path, force_retrain=force_retrain)
                model = trainer.load_model(config.model_path)
                predictor = Predictor(model)

            self.logger.info("Loading test data and making predictions...")
            test_source = config.get_test_files_or_folder(self.config_manager.data_config)
            test_data = self.data_loader.load_testing_data(test_source)
            stat_summary = predictor.predict_testset(test_data, config.result_dir)

            self.logger.info("Processing peak lists...")
            peaklist_folder = config.get_peaklist_folder(self.config_manager.data_config)
            peaklist_files = self.data_loader.load_peaklist_data(peaklist_folder)
            predictor.predict_peaklist(peaklist_files, config.result_dir)

            stat_df = self.evaluator.save_evaluation_summary(
                stat_summary, config.result_dir, config.version_name
            )

            self.logger.info(f"Pipeline completed for {config.version_name}")
            return stat_df

        except Exception as e:
            self.logger.error(f"Pipeline failed for {config.version_name}: {e}")
            raise
    
    def run_all_pipelines(self, force_retrain=False):
        self.logger.info("Starting full pipeline run...")
        
        configs = self.config_manager.get_standard_configs()
        results = {}
        

        for config in configs:
            try:
                stat_df = self.run_single_pipeline(config, force_retrain=force_retrain)
                results[config.version_name] = stat_df
            except Exception as e:
                self.logger.error(f"Failed to run pipeline for {config.version_name}: {e}")
                results[config.version_name] = pd.DataFrame()
        

        self._generate_plots(results)
        
        self._save_combined_comparison(results)
        
        self.logger.info("All pipelines completed")
        return results

    def _derive_model_label(self, source, idx):
        """Create a short label for a training source for use in ensemble outputs."""
        base = os.path.basename(source.rstrip('/'))
        if base.lower().startswith('dom_training_set_ver2'):
            return '7T'
        if base.lower().startswith('dom_training_set_ver3'):
            return '21T'
        if base.lower().startswith('synthetic'):
            return 'SYN'
        if 'Pahokee' in base:
            return 'Pahokee'
        if 'Suwannee' in base:
            return 'Suw' + ''.join([c for c in base if c.isdigit()])
        return f"SRC{idx+1}"

    def _augment_model_path(self, base_model_path, label):
        root, ext = os.path.splitext(base_model_path)
        return f"{root}_{label}{ext}"
    
    def _generate_plots(self, results):
        self.logger.info("Generating plots...")
        
        try:
            # Prepare data for plotting
            result_dirs = []
            labels = []
            
            for version_name in results.keys():
                if not results[version_name].empty:
                    result_dir = self.config_manager.get_standard_configs()[
                        [c.version_name for c in self.config_manager.get_standard_configs()].index(version_name)
                    ].result_dir
                    result_dirs.append(result_dir)
                    labels.append(version_name)
            
            if result_dirs:
                # Generate testset plots
                plotting.plot_testset_main(
                    result_dirs, labels, self.config_manager.data_config.output_dir
                )
                
                # Generate peaklist plots
                plotting.plot_peaklist_main(
                    result_dirs, labels, self.config_manager.data_config.output_dir
                )
                
                self.logger.info("Plots generated successfully")
            else:
                self.logger.warning("No valid results found for plotting")
                
        except Exception as e:
            self.logger.error(f"Error generating plots: {e}")
    
    def _save_combined_comparison(self, results):

        try:
            # Combine all statistics
            all_stats = []
            for version_name, stat_df in results.items():
                if not stat_df.empty:
                    all_stats.append(stat_df)
            
            if all_stats:
                combined_stats = pd.concat(all_stats, ignore_index=True)
                comparison_path = os.path.join(
                    self.config_manager.data_config.base_dir, 
                    "knn_comparison_summary.csv"
                )
                combined_stats.to_csv(comparison_path, index=False)
                self.logger.info(f"Combined comparison saved to: {comparison_path}")
            else:
                self.logger.warning("No valid statistics found for comparison")
                
        except Exception as e:
            self.logger.error(f"Error saving combined comparison: {e}")
    
    def run_custom_pipeline(self, version_name, training_folders, model_path, 
                          result_dir, k_neighbors=1, testing_folder=None, 
                          peaklist_folder=None, custom_test_files=None, force_retrain=False):
        # Create custom configuration
        config = self.config_manager.create_custom_config(
            version_name=version_name,
            training_folders=training_folders,
            model_path=model_path,
            result_dir=result_dir,
            k_neighbors=k_neighbors,
            testing_folder=testing_folder,
            peaklist_folder=peaklist_folder,
            custom_test_files=custom_test_files
        )
        
        # Run pipeline
        return self.run_single_pipeline(config, force_retrain=force_retrain)


# Legacy function for backward compatibility
def run_pipeline_from_folder(version_name, training_folder, testing_folder, 
                           model_path, result_dir, k=1):
    manager = PipelineManager()
    
    # Update testing folder in config
    manager.config_manager.update_data_config(testing_folder=testing_folder)
    
    # Run custom pipeline
    return manager.run_custom_pipeline(
        version_name=version_name,
        training_folders=training_folder,
        model_path=model_path,
        result_dir=result_dir,
        k_neighbors=k
    )


def run_main():
    manager = PipelineManager()
    results = manager.run_all_pipelines()
    return results