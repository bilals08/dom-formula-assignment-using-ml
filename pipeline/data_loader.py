"""
Data loader module for DOM formula assignment pipeline.
Handles loading and preprocessing of training and testing data.
"""

import os
import glob
import pandas as pd
from pipeline.logger import get_logger


class DataLoader:
    """Handles loading and preprocessing of training and testing data."""
    
    def __init__(self):
        self.logger = get_logger("DataLoader")
    
    def load_training_data(self, training_sources):
        """
        Load training data from Excel or CSV files.
        """
        files = []
        
        # Handle both single source and list of sources
        if isinstance(training_sources, str):
            training_sources = [training_sources]
        elif not isinstance(training_sources, list):
            raise ValueError("training_sources must be a string or list of strings")
        
        # Process each source (could be file or folder)
        for source in training_sources:
            if os.path.isfile(source):
                # It's a specific file
                if source.endswith(('.xlsx', '.csv')):
                    files.append(source)
                else:
                    self.logger.warning(f"Skipping unsupported file type: {source}")
            elif os.path.isdir(source):
                # It's a folder - find all Excel and CSV files
                files.extend(glob.glob(os.path.join(source, "*.xlsx")))
                files.extend(glob.glob(os.path.join(source, "*.csv")))
            else:
                self.logger.warning(f"Training source not found: {source}")
        
        all_data = []
        self.logger.info(f"Found {len(files)} training files from {len(training_sources)} source(s)")
        
        for file in files:
            try:
                # Load data based on file type
                if file.endswith('.csv'):
                    df = pd.read_csv(file, low_memory=False)
                else:
                    df = pd.read_excel(file)
                
                # Standardize column names
                df = self._standardize_columns(df)
                
                # Filter and clean data
                df = df[df['Mass_Daltons'].notna() & df['Formula'].notna()]
                df = df[['Mass_Daltons', 'Formula']]
                
                
                all_data.append(df)
                self.logger.info(f"Loaded {len(df)} rows from {os.path.basename(file)}")
                
            except Exception as e:
                self.logger.error(f"Error reading {file}: {e}")
        
        if not all_data:
            self.logger.error("No valid training data found.")
            raise ValueError("No valid training data found.")
        
        combined_data = pd.concat(all_data, ignore_index=True)

        self.logger.info(f"Total training rows: {len(combined_data)}")
        return combined_data
    
    def load_training_data_separate(self, training_sources):
        """
        Load training data from multiple sources, keeping each source separate.
        This is used for ensemble models where each dataset trains a separate model.
        
        Args:
            training_sources: str or list of strings - path(s) to training data folder(s) or file(s)
            
        Returns:
            list of tuples: [(source_name, pd.DataFrame), ...] for each training source
        """
        # Handle both single source and list of sources
        if isinstance(training_sources, str):
            training_sources = [training_sources]
        elif not isinstance(training_sources, list):
            raise ValueError("training_sources must be a string or list of strings")
        
        separate_datasets = []
        
        # Process each source separately
        for source in training_sources:
            files = []
            
            if os.path.isfile(source):
                # It's a specific file
                if source.endswith(('.xlsx', '.csv')):
                    files.append(source)
                else:
                    self.logger.warning(f"Skipping unsupported file type: {source}")
            elif os.path.isdir(source):
                # It's a folder - find all Excel and CSV files
                files.extend(glob.glob(os.path.join(source, "*.xlsx")))
                files.extend(glob.glob(os.path.join(source, "*.csv")))
            else:
                self.logger.warning(f"Training source not found: {source}")
                continue
            
            # Load data from all files in this source
            source_data = []
            for file in files:
                try:
                    # Load data based on file type
                    if file.endswith('.csv'):
                        df = pd.read_csv(file, low_memory=False)
                    else:
                        df = pd.read_excel(file)
                    
                    # Standardize column names
                    df = self._standardize_columns(df)
                    
                    # Filter and clean data
                    df = df[df['Mass_Daltons'].notna() & df['Formula'].notna()]
                    df = df[['Mass_Daltons', 'Formula']]
                    print(df.shape)
                    
                    source_data.append(df)
                    self.logger.info(f"Loaded {len(df)} rows from {os.path.basename(file)}")
                    
                except Exception as e:
                    self.logger.error(f"Error reading {file}: {e}")
            if source_data:
                combined_source_data = pd.concat(source_data, ignore_index=True)
                source_name = os.path.basename(source) if os.path.isdir(source) else os.path.basename(source).replace('.xlsx', '').replace('.csv', '')
                separate_datasets.append((source_name, combined_source_data))
                self.logger.info(f"Training dataset '{source_name}': {len(combined_source_data)} rows")
        
        if not separate_datasets:
            self.logger.error("No valid training data found.")
            raise ValueError("No valid training data found.")
        
        self.logger.info(f"Loaded {len(separate_datasets)} separate training datasets")
        return separate_datasets
    
    def load_testing_data(self, testing_source):
        """
        Load testing data.
        """
        # Handle both folder path and list of specific files
        if isinstance(testing_source, str):
            # It's a folder path
            files = glob.glob(os.path.join(testing_source, "*.xlsx")) or glob.glob(os.path.join(testing_source, "*.csv"))
            self.logger.info(f"Found {len(files)} testing files in {testing_source}")
        elif isinstance(testing_source, list):
            # It's a list of specific file paths
            files = testing_source
            self.logger.info(f"Loading {len(files)} specific test files")
        else:
            raise ValueError("testing_source must be a string (folder path) or list (file paths)")
        
        test_data = []
        
        for file in files:
            try:
                # File paths from glob.glob() are already correct, no need to join again
                if file.endswith('.xlsx'):
                    df = pd.read_excel(file)
                elif file.endswith('.csv'):
                    df = pd.read_csv(file, low_memory=False)
                # Group by m/z and formula to remove duplicates
                df = df.groupby(['m/z exp.', 'Chem. Formula'], as_index=False).first()
                
                if not df.empty:
                    test_data.append((os.path.basename(file), df))
                    self.logger.info(f"Loaded {len(df)} rows from {os.path.basename(file)}")
                else:
                    self.logger.warning(f"Skipped empty file: {os.path.basename(file)}")
                    
            except Exception as e:
                self.logger.error(f"Error reading {file}: {e}")
        
        return test_data
    
    def load_peaklist_data(self, peak_list_dir):
        """
        Load peak list data from CSV files.
        
        """
        pattern = os.path.join(peak_list_dir, '**', '*.csv')
        csv_paths = glob.glob(pattern, recursive=True)
        filenames = [os.path.basename(path) for path in csv_paths]
        
        self.logger.info(f"Found {len(csv_paths)} peak list files in {peak_list_dir}")
        return list(zip(csv_paths, filenames))
    
    def _standardize_columns(self, df):
        """
        """
        # Common column name mappings
        column_mappings = {
            'm/z exp.': 'Mass_Daltons',
            'Chem. Formula': 'Formula',
            'm/z Exp.': 'm/z exp.'  # For peak list files
        }
        
        for old_name, new_name in column_mappings.items():
            if old_name in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)
        
        return df