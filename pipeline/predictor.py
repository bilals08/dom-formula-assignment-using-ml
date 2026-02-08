import os
import pandas as pd
from pipeline.logger import get_logger
from pipeline.utils.utils import calculate_theoretical_mass, ensure_dir



class Predictor:
    """Handles predictions using trained KNN model."""
    
    def __init__(self, model):
        self.model = model
        self.logger = get_logger("Predictor")
    
    def predict_peaklist(self, peaklist_files, result_dir):
        """
        Make predictions on peak list files.
        """
        peaklist_output_dir = os.path.join(result_dir, "peak_list")
        ensure_dir(peaklist_output_dir)
        
        self.logger.info(f"Predicting on {len(peaklist_files)} peak list files")
        
        for file_path, filename in peaklist_files:
            self.logger.info(f"Processing peak list: {filename}")
            
            try:
                # Load peak list data
                data = pd.read_csv(file_path)

                
                # Standardize column names
                if 'm/z Exp.' in data.columns:
                    data.rename(columns={'m/z Exp.': 'm/z exp.'}, inplace=True)
                
                #remove duplicates based on m/z exp.
                data = data.drop_duplicates(subset=['m/z exp.'])

                # Make predictions
                predictions = []
                for _, row in data.iterrows():
                    prediction = self._predict_single_peak(row)
                    predictions.append(prediction)
                
                # Save results
                df = pd.DataFrame(predictions)
                # df.drop_duplicates(subset=['predicted_formula'], inplace=True)
                output_path = os.path.join(peaklist_output_dir, filename)
                df.to_csv(output_path, index=False)
                
                valid_count = df['valid_prediction'].sum()
                self.logger.info(f"Peak list predictions saved to: {output_path} "
                               f"(Valid predictions: {valid_count}/{len(df)})")
                
            except Exception as e:
                self.logger.error(f"Error processing {filename}: {e}")
    
    def predict_testset(self, test_data, result_dir):
        print("Predicting on test set...")
        ensure_dir(result_dir)
        stat_summary = []
        
        self.logger.info(f"Evaluating model on {len(test_data)} test files")
        for filename, df in test_data:
            self.logger.info(f"Processing test file: {filename}")
            
            try:
                predictions = []
                for _, row in df.iterrows():
                    prediction = self._predict_test_sample(row, filename)
                    predictions.append(prediction)
                
                # Create results dataframe
                pred_df = pd.DataFrame(predictions)
                print(pred_df.shape)
                
                # Calculate statistics
                stats = self._calculate_prediction_stats(pred_df, filename)
                stat_summary.append(stats)
                
                # Save results
                results_file = os.path.join(result_dir, f"results_{filename.replace('.xlsx', '')}.csv")
                print(f"Saving results to: {results_file}")
                pred_df.to_csv(results_file, index=False)
                
                self.logger.info(f"Results saved to: {results_file} | "
                               f"Predicted: {stats['Predicted']}, "
                               f"New Assignments: {stats['New Assignments']}, "
                               f"Wrong Predictions: {stats['Wrong Predictions']}")
                
            except Exception as e:
                self.logger.error(f"Error processing {filename}: {e}")
        
        return stat_summary
    
    def _predict_single_peak(self, row):
        """
        Make prediction for a single peak in peak list.
        
        """
        mz = round(row['m/z exp.'], 5)
        intensity = row["Intensity"]

        # Single nearest neighbor prediction (reverted logic)
        pred_formula = self.model.predict([[mz]])[0]
        # Obtain nearest neighbor training mass (k=1 model)
        try:
            nn_idx = self.model.kneighbors([[mz]], n_neighbors=1, return_distance=False)[0][0]
            training_mz = self.model._fit_X[nn_idx][0]
        except Exception:
            training_mz = None
        pred_formula_mass = calculate_theoretical_mass(pred_formula)
        mass_error_ppm = (abs(pred_formula_mass - mz) / mz) * 1e6
        pred_formula_mass = round(pred_formula_mass, 5)
        pred_formula_mass_error = round(abs(pred_formula_mass - mz), 4)
        valid_prediction = 1 if mass_error_ppm <= 1 else 0

        return {
            'm/z exp.': mz,
            'Intensity': intensity,
            'predicted_formula': pred_formula,
            'pred_formula_mass': pred_formula_mass,
            'training_mz': training_mz,
            'pred_formula_mass_error': pred_formula_mass_error,
            'mass_error_in_ppm': mass_error_ppm,
            'valid_prediction': valid_prediction
        }
    
    def _predict_test_sample(self, row, filename=None):
        """
        Make prediction for a single test sample.
        
        Args:
            row: pandas Series - test data row
            
        Returns:
            dict: prediction results with evaluation metrics
        """
        mz = round(row['m/z exp.'], 5)
        true_formula = row['Chem. Formula']

        # Single nearest neighbor prediction (reverted logic)
        pred_formula = self.model.predict([[mz]])[0]
        pred_formula_mass = calculate_theoretical_mass(pred_formula)
        mass_error_ppm = (abs(pred_formula_mass - mz) / mz) * 1e6

        # Calculate masses and errors
        true_formula_mass = calculate_theoretical_mass(true_formula)
        pred_formula_mass = pred_formula_mass if isinstance(pred_formula_mass, (int, float)) else calculate_theoretical_mass(pred_formula)
        true_formula_mass_error = round(abs(true_formula_mass - mz), 4)
        pred_formula_mass_error = round(abs(pred_formula_mass - mz), 4)
        mass_error_ppm = mass_error_ppm if mass_error_ppm is not None else (abs(pred_formula_mass - mz) / mz) * 1e6
        
        predicted = 1 if pred_formula == true_formula else 0
        new_assignment = 1 if (pred_formula != true_formula and mass_error_ppm < 1) else 0
        wrong_prediction = 1 if (pred_formula != true_formula and mass_error_ppm >= 1) else 0

        # If the test file is from synthetic data, treat matching predictions as new assignments
        is_synthetic = False
        try:
            if filename and isinstance(filename, str):
                is_synthetic = 'synthetic' in filename.lower()
        except Exception:
            is_synthetic = False
        if is_synthetic and predicted == 1:
            predicted = 0
            new_assignment = 1

        return {
            'm/z exp.': mz,
            'proposed_formula': true_formula,
            'predicted_formula': pred_formula,
            'predicted': predicted,
            'new_assignment': new_assignment,
            'wrong_prediction': wrong_prediction,
            'proposed_formula_mass': true_formula_mass,
            'pred_formula_mass': pred_formula_mass,
            'proposed_formula_mass_error': true_formula_mass_error,
            'pred_formula_mass_error': pred_formula_mass_error,
            'mass_error_in_ppm': mass_error_ppm
        }
    
    def _calculate_prediction_stats(self, pred_df, filename):
        """
            
        Returns:
            dict: prediction statistics
        """
        # Use updated column names: 'predicted', 'new_assignment', 'wrong_prediction'
        return {
            'Filename': filename,
            'Total Count': len(pred_df),
            'Predicted': int(pred_df['predicted'].sum()) if 'predicted' in pred_df.columns else 0,
            'New Assignments': int(pred_df['new_assignment'].sum()) if 'new_assignment' in pred_df.columns else 0,
            'Wrong Predictions': int(pred_df['wrong_prediction'].sum()) if 'wrong_prediction' in pred_df.columns else 0
        }


class MultiPredictor:
    """Predictor that ensembles multiple trained models and selects best prediction.
    """

    def __init__(self, models):
        """Initialize with list of (model_label, model_instance)."""
        self.models = models  # list[tuple[str, KNeighborsClassifier]]
        self.logger = get_logger("MultiPredictor")

    def predict_peaklist(self, peaklist_files, result_dir):
        peaklist_output_dir = os.path.join(result_dir, "peak_list")
        ensure_dir(peaklist_output_dir)
        self.logger.info(f"[MultiPredictor] Predicting on {len(peaklist_files)} peak list files with {len(self.models)} models")

        for file_path, filename in peaklist_files:
            try:
                data = pd.read_csv(file_path)
                if 'm/z Exp.' in data.columns:
                    data.rename(columns={'m/z Exp.': 'm/z exp.'}, inplace=True)
                data = data.drop_duplicates(subset=['m/z exp.'])
                predictions = []
                for _, row in data.iterrows():
                    predictions.append(self._predict_ensemble_peak(row))
                df = pd.DataFrame(predictions)
                # df.drop_duplicates(subset=['predicted_formula'], inplace=True)
                output_path = os.path.join(peaklist_output_dir, filename)
                df.to_csv(output_path, index=False)
                valid_count = df['valid_prediction'].sum()
                self.logger.info(f"Saved multi-model peak predictions to {output_path} (Valid {valid_count}/{len(df)})")
            except Exception as e:
                self.logger.error(f"Error processing peak list {filename}: {e}")

    def predict_testset(self, test_data, result_dir):
        ensure_dir(result_dir)
        stat_summary = []
        self.logger.info(f"[MultiPredictor] Evaluating on {len(test_data)} test files with {len(self.models)} models")
        for filename, df in test_data:
            try:
                predictions = [self._predict_ensemble_test(row, filename) for _, row in df.iterrows()]
                pred_df = pd.DataFrame(predictions)
                stats = self._calculate_prediction_stats(pred_df, filename)
                stat_summary.append(stats)
                results_file = os.path.join(result_dir, f"results_{filename.replace('.xlsx', '')}.csv")
                pred_df.to_csv(results_file, index=False)
                self.logger.info(f"Saved multi-model results to {results_file} | Predicted {stats['Predicted']} NewAssignments {stats['New Assignments']} Wrong {stats['Wrong Predictions']}")
            except Exception as e:
                self.logger.error(f"Error evaluating test file {filename}: {e}")
        return stat_summary

    def _candidate_from_model(self, model, mz):
        """Generate candidate prediction from a single model for given m/z (single NN)."""
        pred_formula = model.predict([[mz]])[0]
        try:
            nn_idx = model.kneighbors([[mz]], n_neighbors=1, return_distance=False)[0][0]
            training_mz = model._fit_X[nn_idx][0]
        except Exception:
            training_mz = None
        pred_formula_mass = calculate_theoretical_mass(pred_formula)
        mass_error_ppm = (abs(pred_formula_mass - mz) / mz) * 1e6
        pred_formula_mass_round = round(pred_formula_mass, 5)
        pred_formula_mass_error = round(abs(pred_formula_mass_round - mz), 4)
        return {
            'predicted_formula': pred_formula,
            'pred_formula_mass': pred_formula_mass_round,
            'pred_formula_mass_error': pred_formula_mass_error,
            'mass_error_in_ppm': mass_error_ppm,
            'training_mz': training_mz,
        }

    def _predict_ensemble_peak(self, row):
        mz = round(row['m/z exp.'], 5)
        intensity = row.get('Intensity')
        candidates = {}
        for label, model in self.models:
            cand = self._candidate_from_model(model, mz)
            candidates[label] = cand
        # Select best candidate (lowest ppm)
        best_label, best_cand = min(candidates.items(), key=lambda kv: kv[1]['mass_error_in_ppm'])
        valid_prediction = 1 if best_cand['mass_error_in_ppm'] <= 1 else 0
        result = {
            'm/z exp.': mz,
            'Intensity': intensity,
            'predicted_formula': best_cand['predicted_formula'],
            'pred_formula_mass': best_cand['pred_formula_mass'],
            'training_mz': best_cand['training_mz'],
            'pred_formula_mass_error': best_cand['pred_formula_mass_error'],
            'mass_error_in_ppm': best_cand['mass_error_in_ppm'],
            'valid_prediction': valid_prediction,
            'selected_model': best_label
        }
        # Add per-model diagnostics
        for label, cand in candidates.items():
            result[f'{label}_formula'] = cand['predicted_formula']
            result[f'{label}_ppm_error'] = cand['mass_error_in_ppm']
        return result

    def _predict_ensemble_test(self, row, filename=None):
        mz = round(row['m/z exp.'], 5)
        true_formula = row['Chem. Formula']
        candidates = {}
        for label, model in self.models:
            cand = self._candidate_from_model(model, mz)
            candidates[label] = cand
  
        best_label, best_cand = min(candidates.items(), key=lambda kv: kv[1]['mass_error_in_ppm'])

        pred_formula = best_cand['predicted_formula']
        pred_formula_mass = best_cand['pred_formula_mass']
        mass_error_ppm = best_cand['mass_error_in_ppm']
        true_formula_mass = calculate_theoretical_mass(true_formula)
        true_formula_mass_error = round(abs(true_formula_mass - mz), 4)
        pred_formula_mass_error = best_cand['pred_formula_mass_error']
     
        predicted = 1 if pred_formula == true_formula else 0
        new_assignment = 1 if (pred_formula != true_formula and mass_error_ppm < 1) else 0
        wrong_prediction = 1 if (pred_formula != true_formula and mass_error_ppm >= 1) else 0

        # If the test file is from synthetic data, treat matching predictions as new assignments
        is_synthetic = False
        try:
            if filename and isinstance(filename, str):
                is_synthetic = 'synthetic' in filename.lower()
        except Exception:
            is_synthetic = False
        if is_synthetic and predicted == 1:
            predicted = 0
            new_assignment = 1
        result = {
            'm/z exp.': mz,
            'proposed_formula': true_formula,
            'predicted_formula': pred_formula,
            'predicted': predicted,
            'new_assignment': new_assignment,
            'wrong_prediction': wrong_prediction,
            'proposed_formula_mass': true_formula_mass,
            'pred_formula_mass': pred_formula_mass,
            'proposed_formula_mass_error': true_formula_mass_error,
            'pred_formula_mass_error': pred_formula_mass_error,
            'mass_error_in_ppm': mass_error_ppm,
            'selected_model': best_label
        }
        for label, cand in candidates.items():
            result[f'{label}_formula'] = cand['predicted_formula']
            result[f'{label}_ppm_error'] = cand['mass_error_in_ppm']
        return result

    def _calculate_prediction_stats(self, pred_df, filename):
        base_stats = {
            'Filename': filename,
            'Total Count': len(pred_df),
            'Predicted': int(pred_df['predicted'].sum()) if 'predicted' in pred_df.columns else 0,
            'New Assignments': int(pred_df['new_assignment'].sum()) if 'new_assignment' in pred_df.columns else 0,
            'Wrong Predictions': int(pred_df['wrong_prediction'].sum()) if 'wrong_prediction' in pred_df.columns else 0
        }
        # Append per-model selection usage if available
        if 'selected_model' in pred_df.columns:
            vc = pred_df['selected_model'].value_counts()
            for model_label, count in vc.items():
                base_stats[f'Selected_{model_label}'] = count
                base_stats[f'Selected_{model_label}_pct'] = round(count / len(pred_df) * 100, 2) if len(pred_df) else 0.0
        return base_stats