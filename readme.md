# DOM Formula Assignment

A machine learning pipeline for assigning molecular formulas to dissolved organic matter (DOM) mass spectrometry data using K-Nearest Neighbors (KNN) models.

##  Overview

This research implements an automated pipeline for formula assignment in DOM (Dissolved Organic Matter) mass spectrometry analysis. It uses KNN  models to predict molecular formulas based on mass-to-charge (m/z) and m/z error (for filtering and picking up the formula), supporting multiple training datasets, distance metrics, and K-neighbor configurations.


### 

- Python 3.8+
- Required packages: pandas, numpy, scikit-learn, matplotlib, seaborn, wordcloud, joblib

### Basic Usage

```python
from pipeline import run_main
# Run all standard pipelines (L1, L3, L1-L3, Synthetic)
results = run_main()
```

## Data
The data is organized into the following folders and files: The L1 is the data from the 7T with mass resolution of 1 PPM, L2 is the data from the 9.4T instrument with mass resolution of 0.2-0.4 PPM, and L3 is the data from 21T with mass resolution of 0.15 PPM. The synthetic_data folder contains synthetically generated formulas for training.

* readme.txt
* train.txt (L1 with Mobility Features)
* test.txt (L2-v2 with Mobility Features)
* DOM_testing_set (L2)
* DOM_testing_set_Peaklists (L2-Peaklists)
* DOM_training_set_ver2 (L1)
* DOM_training_set_ver3 (L3)
* synthetic_data (Synthetic)


## Model Configurations

The pipeline includes four standard model configurations:

### 1. Model-L1
- **Training Data**: DOM_training_set_ver2 (L1 data)
- **Description**: Model trained on L1 mass spectrometry data

### 2. Model-L3
- **Training Data**: DOM_training_set_ver3 (L3 data)
- **Description**: Model trained on L3 mass spectrometry data

### 3. Model-L1-L3 (Ensemble)
- **Training Data**: Both ver2 and ver3
- **Description**: Ensemble model combining L1 and L3 data

### 4. Model-Synthetic (Ensemble)
- **Training Data**: Combined L1, L3, and synthetic data
- **Description**: Enhanced with synthetically generated formulas

Each model is trained with multiple configurations:
- **K values**: 1, 3 (number of nearest neighbors)
- **Distance metrics**: Euclidean (p=2), Manhattan (p=1)

This results in 16 model variants (4 models × 2 K values × 2 metrics).


### Output Files

**Per-Test-File Results** (`results_*.csv`):

**Evaluation Summary** (`evaluation_summary_stats.csv`):

**Peak List Predictions** (`peak_list/*.csv`):


## Usage Examples

### Example 1: Basic Pipeline Execution
```python
from pipeline import run_main

# Run all standard pipelines (7T, 21T, Combined, Synthetic)
results = run_main()
```

### Example 2: Custom Pipeline
```python
from pipeline import PipelineManager

manager = PipelineManager()

# Train with custom data sources 
results = manager.run_custom_pipeline(
    version_name="Custom_Model",
    training_folders=[
        "data/custom_training/curated_formulas.csv"  # Additional file
    ],
    model_path="models/custom_model.joblib",
    result_dir="output/custom/test_results",
    k_neighbors=3,
    metric='minkowski',
    p=2  # Euclidean distance
)
```

# License and Usage Terms
This model and associated code are released under the CC-BY-NC-ND 4.0 license and may only be used for non-commercial, academic research purposes with proper attribution. Any commercial use, sale, or other monetization of this model and its derivatives, which include models trained on outputs from the model or datasets created from the model, is prohibited and requires prior approval. If you are a commercial entity, please contact the corresponding author.

# Contact
For any additional questions or comments, contact Fahad Saeed (fsaeed@fiu.edu).
