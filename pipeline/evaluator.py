"""
Evaluator module for DOM formula assignment pipeline.
Handles model evaluation and metrics calculation.
"""

import os
import pandas as pd
from pipeline.logger import get_logger


class Evaluator:
    """Handles model evaluation and metrics calculation."""
    
    def __init__(self):
        self.logger = get_logger("Evaluator")
    
    def save_evaluation_summary(self, stat_summary, result_dir, model_version=None):
        """
        Save evaluation summary statistics to CSV.
        """
        if not stat_summary:
            self.logger.warning("No statistics to save")
            return pd.DataFrame()
        
        # Create dataframe from statistics
        stat_df = pd.DataFrame(stat_summary)
        
        # Add model version if provided
        if model_version:
            stat_df['Model Version'] = model_version
        
        # Save to file
        stat_path = os.path.join(result_dir, "evaluation_summary_stats.csv")
        stat_df.to_csv(stat_path, index=False)
        
        # Log summary
        total_samples = stat_df['Total Count'].sum()
        # Support both old and new column names; prefer new names if present
        if 'Predicted' in stat_df.columns:
            total_true = stat_df['Predicted'].sum()
        elif 'True Predictions' in stat_df.columns:
            total_true = stat_df['True Predictions'].sum()
        else:
            total_true = 0

        if 'Wrong Predictions' in stat_df.columns:
            total_false = stat_df['Wrong Predictions'].sum()
        elif 'False Predictions' in stat_df.columns:
            total_false = stat_df['False Predictions'].sum()
        else:
            total_false = 0

        # If 'No Predictions' isn't present, derive it from totals
        if 'No Predictions' in stat_df.columns:
            total_no = stat_df['No Predictions'].sum()
        else:
            total_no = total_samples - (total_true + total_false)
        if total_no < 0:
            total_no = 0
        
        self.logger.info(f"Evaluation summary saved to: {stat_path}")
        self.logger.info(f"Overall statistics - Total: {total_samples}, "
                        f"True: {total_true} ({total_true/total_samples*100:.1f}%), "
                        f"False: {total_false} ({total_false/total_samples*100:.1f}%), "
                        f"No: {total_no} ({total_no/total_samples*100:.1f}%)")
        
        return stat_df
    
    def calculate_accuracy_metrics(self, stat_df):
        """
        Calculate accuracy metrics from statistics dataframe.
        
        """
        total_samples = stat_df['Total Count'].sum()
        # Accept either new or legacy column names
        if 'Predicted' in stat_df.columns:
            total_true = stat_df['Predicted'].sum()
        elif 'True Predictions' in stat_df.columns:
            total_true = stat_df['True Predictions'].sum()
        else:
            total_true = 0

        if 'Wrong Predictions' in stat_df.columns:
            total_false = stat_df['Wrong Predictions'].sum()
        elif 'False Predictions' in stat_df.columns:
            total_false = stat_df['False Predictions'].sum()
        else:
            total_false = 0

        if 'No Predictions' in stat_df.columns:
            total_no = stat_df['No Predictions'].sum()
        else:
            total_no = total_samples - (total_true + total_false)
        if total_no < 0:
            total_no = 0
        
        if total_samples == 0:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'assignment_rate': 0.0
            }
        
        # Calculate metrics
        accuracy = total_true / total_samples
        assignment_rate = (total_true + total_false) / total_samples
        precision = total_true / (total_true + total_false) if (total_true + total_false) > 0 else 0.0
        recall = accuracy  # Same as accuracy in this context
        
        return {
            'accuracy': accuracy,
            'precision': precision, 
            'recall': recall,
            'assignment_rate': assignment_rate,
            'total_samples': total_samples,
            'true_predictions': total_true,
            'false_predictions': total_false,
            'no_predictions': total_no
        }
    
    def compare_models(self, stat_dfs, model_names):
        """
        Compare multiple models' performance.
        
        Args:
            stat_dfs: list - list of statistics dataframes
            model_names: list - list of model names
            
        Returns:
            pd.DataFrame: comparison dataframe
        """
        comparison_data = []
        
        for stat_df, model_name in zip(stat_dfs, model_names):
            metrics = self.calculate_accuracy_metrics(stat_df)
            metrics['Model'] = model_name
            comparison_data.append(metrics)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Log comparison
        self.logger.info("Model comparison:")
        for _, row in comparison_df.iterrows():
            self.logger.info(f"{row['Model']}: Accuracy={row['accuracy']:.3f}, "
                           f"Precision={row['precision']:.3f}, "
                           f"Assignment Rate={row['assignment_rate']:.3f}")
        
        return comparison_df
    
    def generate_evaluation_report(self, result_dir, model_version):
        """
        Generate a comprehensive evaluation report.
        """
        # Load evaluation summary if it exists
        summary_path = os.path.join(result_dir, "evaluation_summary_stats.csv")
        if not os.path.exists(summary_path):
            self.logger.warning(f"No evaluation summary found at {summary_path}")
            return {}
        
        stat_df = pd.read_csv(summary_path)
        metrics = self.calculate_accuracy_metrics(stat_df)
        
        # Create report
        report = {
            'model_version': model_version,
            'evaluation_date': pd.Timestamp.now().isoformat(),
            'metrics': metrics,
            'file_level_stats': stat_df.to_dict('records')
        }
        
        # Save report
        report_path = os.path.join(result_dir, f"evaluation_report_{model_version}.json")
        pd.Series(report).to_json(report_path, indent=2)
        
        self.logger.info(f"Evaluation report saved to: {report_path}")
        return report