# FrogID Audio Classifier

This project is an end-to-end machine learning pipeline for multiclass frog species classification using audio recordings, following Databricks MLOps best practices. The pipeline uses BirdNET embeddings for feature extraction and supports 15 target frog species.

## 🎯 Target Species

The model classifies 15 Australian frog species:
- Rhinella marina (Cane Toad)
- Crinia signifera (Common Froglet)
- Limnodynastes peronii (Striped Marsh Frog)
- Litoria moorei (Motorbike Frog)
- Litoria fallax (Eastern Sedge Frog)
- Limnodynastes tasmaniensis (Spotted Grass Frog)
- Crinia parinsignifera (Beeping Froglet)
- Limnodynastes dumerilii (Eastern Banjo Frog)
- Litoria caerulea (Green Tree Frog)
- Litoria ewingii (Southern Brown Tree Frog)
- Litoria verreauxii (Whistling Tree Frog)
- Litoria rubella (Little Red Tree Frog)
- Crinia glauerti (Clicking Froglet)
- Litoria peronii (Peron's Tree Frog)
- Litoria ridibunda (Laughing Tree Frog)

## 📁 Project Structure

```
frogid-ml-15species-dbx/
├── mlops/                          # Main MLOps pipeline
│   ├── feature_engineering/        # Data processing and feature extraction
│   │   ├── data_selector.py        # Filter and select target species
│   │   ├── data_downloader.py      # Download audio files from URLs
│   │   ├── data_sampler.py         # Sample data for training
│   │   ├── preprocessor_*.py       # Audio preprocessing components
│   │   └── registry_*.py           # Strategy registries
│   ├── training/                   # Model training
│   │   ├── tf_model_trainer.py     # TensorFlow model training
│   │   ├── tf_model_registry.py    # Model registration
│   │   └── models/                 # Model architectures
│   │       └── birdnet_mlp_multiclass.py
│   ├── evaluation/                 # Model evaluation
│   │   └── tf_model_evaluator.py   # Comprehensive evaluation metrics
│   ├── utils/                      # Utilities and configuration
│   │   ├── config.py               # Configuration classes
│   │   ├── cache_utils.py          # Caching utilities
│   │   ├── data_utils.py           # Data processing utilities
│   │   ├── logging_utils.py        # Logging configuration
│   │   ├── mlflow_utils.py         # MLflow integration
│   │   └── gpu_config.py           # GPU configuration
│   ├── reporting/                  # Output generation
│   │   └── output_generator.py     # Report generation
│   ├── mlflow/                     # MLflow tracking
│   │   └── tracking.db             # SQLite database for experiments
│   ├── tests/                      # Unit tests
│   │   └── test_gpu.py             # GPU testing
│   ├── resources/                  # Databricks resources
│   │   └── ml-artifacts-resource.yml
│   └── databricks.yml              # Databricks bundle configuration
├── dashboard/                      # Streamlit visualization dashboard
│   ├── app.py                      # Main dashboard application
│   ├── ui.py                       # UI components
│   ├── data.py                     # Data loading for dashboard
│   ├── plots.py                    # Plotting functions
│   ├── constants.py                # Target species constants
│   ├── styles.css                  # Custom CSS styling
│   └── test.ipynb                  # Dashboard testing notebook
├── experiments/                    # Research notebooks
│   └── birdnet-average-3s-chunk-embeddings.ipynb
├── data/                          # Data directory (local)
├── docs/                          # Documentation
│   ├── FrogID ML - ML Project Repo Structure & Git Approach.pdf
│   ├── FrogID ML – Databricks MLOps Naming Conventions.pdf
│   └── repo structure quick summary.md
├── DEPLOYMENT_GUIDE.md            # Databricks deployment instructions
├── requirements.txt               # Python dependencies
├── pyproject.toml                 # Poetry configuration
└── README.md                      # This file
```

## 🚀 Quick Start

### Option 1: Run Pipeline
```bash
experiments/birdnet-average-3s-chunk-embeddings.ipynb
```

### Option 2: View Results Dashboard
```bash
streamlit run dashboard/app.py
```

### Option 3: View MLflow UI
```bash
mlflow ui --backend-store-uri sqlite:///mlops/mlflow/tracking.db --default-artifact-root ./mlruns --port 5001
```

## 🛠️ Local Setup

### Prerequisites

- **Python 3.11 or higher**
- **FFmpeg** (required for audio processing)
- **Poetry** (recommended for dependency management)

#### Install FFmpeg
```bash
# On macOS
brew install ffmpeg

# On Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# On Windows
# Download from https://ffmpeg.org/download.html
```

### Environment Setup

#### Using Poetry
```bash
# 1. Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# (Optional) check python version and tell Peotry to use a Python 
poetry env use $(which python3.11)

#Note: ^3.11 in pyproject.toml means: any version >=3.11.0 and <4.0.0. You can change the requirement if you want to allow older Python versions, but that's not recommended if the code depends on features from 3.11.

# 2. Install dependencies
poetry install

# 3. For Apple Silicon Macs, install platform-specific TensorFlow
poetry install --with apple-silicon

# 4. Activate environment
poetry shell

# 5. Create data directories
mkdir -p data/{raw,interim,processed,audio_files}
```

### Data Setup

```bash
# Place your FrogID dataset CSV file in data/raw/
cp /path/to/your/frogid_data.csv data/raw/20250317.csv
```

Under data/raw folder, please save the latest full dataset. 
The latest dataset: https://drive.google.com/file/d/1aR-USCjGQAld9Eb5BrjqSpqgsHI-2gQG/view

## 🎯 ML Pipeline Overview

The pipeline consists of several modular components:

### 1. Data Processing (`feature_engineering/`)
- **`data_selector.py`**: Filter and select target species from FrogID dataset
- **`data_downloader.py`**: Download audio files from URLs with retry logic
- **`data_sampler.py`**: Sample data for balanced training sets
- **`preprocessor_audio_loader.py`**: Load and resample audio files
- **`preprocessor_audio_filter.py`**: Apply noise reduction and audio filtering
- **`preprocessor_audio_chunker.py`**: Split audio into fixed-length chunks
- **`preprocessor_extractor_birdnet.py`**: Extract BirdNET embeddings
- **`preprocessor_extractor_logmel.py`**: Extract log-mel spectrograms
- **`preprocessor_extractor_mfcc.py`**: Extract MFCC features
- **`preprocessor_feature_normalizer.py`**: Normalize extracted features
- **`preprocessor_orchestrator.py`**: Coordinate the preprocessing pipeline

### 2. Model Training (`training/`)
- **`tf_model_trainer.py`**: Main training script with MLflow integration
- **`tf_model_registry.py`**: Model registration utilities
- **`models/birdnet_mlp_multiclass.py`**: Neural network architecture definitions

### 3. Model Evaluation (`evaluation/`)
- **`tf_model_evaluator.py`**: Comprehensive evaluation with metrics, confusion matrices, and performance analysis

### 4. Utilities (`utils/`)
- **`config.py`**: Configuration classes for all components
- **`mlflow_utils.py`**: MLflow integration helpers
- **`gpu_config.py`**: GPU configuration for TensorFlow
- **`data_utils.py`**: Data processing utilities
- **`logging_utils.py`**: Logging configuration

### 5. Visualization Dashboard (`dashboard/`)
- **`app.py`**: Streamlit dashboard for data exploration and results visualization
- **`plots.py`**: Interactive plotting functions
- **`data.py`**: Data loading and processing for dashboard

## 📊 Dashboard Features

The Streamlit dashboard provides:
- **Overview Tab**: Dataset statistics and distribution
- **Single Species Tab**: Individual species analysis
- **Multi Species Tab**: Comparative analysis across species
- **Experiment Tab**: MLflow experiment exploration

Launch the dashboard:
```bash
streamlit run dashboard/app.py
```

## 🔬 Running Experiments

To run the ML Pipeline, it is now concise and structured.
```bash
# Entry point 
/experiments/template_notebook.ipynb
```

### Research Notebooks
Explore the `experiments/` directory for research notebooks:
- `template_notebook.ipynb`: this is a template notabook for all experiments orinating from `birdnet-average-3s-chunk-embeddings.ipynb`: BirdNET embedding analysis.

Before starting any experiment, you should copy `template_notebook.ipynb` and create a new notebook to define the `EXPERIMENT_NAME` (ideally, matching the name of the notebook) and the baseline configuration for all the classes that will be used across the experiment.

This notebook will handle all the MLflow for you.

#### Configurable Options
```python
#-------------------------------------------------------------------------------
# CONFIGURABLE OPTIONS
#-------------------------------------------------------------------------------
# Experiment name (should match notebook name)
EXPERIMENT_NAME = "Experiment Name"                      
# Author email address on databricks
CURRENT_USER="current.user@matrgroup.com"                   
```

#### Start MLFlow
- for a new run, run_id = None
- for an existing run, run_id = "existing run id"
This an be a new run (run_id = None) or an existing run that you want to reload, by specifying the run_id

```python
run_name, run_id = start_mlflow_run(run_id="de0d86ce189d41548fe5b679ed33d63e"/None)
config = generate_pipeline_config(experiment, run_id)
pipeline = instantiate_pipeline(config)
```

#### Load Clean Data
The anchoring function determines how you select the class_label_single from a list of species in multi-species settings. Below we use the most-frequent-target strategy, which means that if there are multiple species the single class label will be the species that is most frequently represented among the list of class labels.
**ANCHORING_STRATEGY_REGISTRY**
- "first": "Use the first species in the list"
- "first-target": "Use the first target species in the list"
- "most-frequent": "Use globally most frequent species"
- "most-frequent-target": "Most frequent target species"
```python
# Load the cleaned data and their classes
df_data, class_labels_to_species_mapping = pipeline.data_selector.load_data(
    label_anchor_fn=ANCHORING_STRATEGY_REGISTRY["most-frequent-target"]
)
```

#### Modelling
1. Selecting the subset of data to use for modelling based on a strategy
2. Downloading & Preprocessing the selected subset to create a feature df
3. Training the model according to the initial experiment setup
**Modelling Sampling Strategy**
- species_data_to_use: 'single-species-only', 'multi-species-only'
- target_species_sampling_strategy: 'downsample', 'upsample'
- target_species_max_samples_per_class: 1300
- include_other_species: bool = True
- other_species_sampling_strategy: 'random', 'stratify'
- other_species_boost_factor: 1.0
- exclude_other_species_below_count: None

For more model, we can create a file under /models folder with model registration so that we can choose the model we want to test.
```python
model = pipeline.model_trainer.train(df_modelling_features, model_fn=MODEL_REGISTRY['birdnet_mlp_multiclass'])
```

#### Evaluaion
Evaluation of a model
- based on a Test Data
- based on a Hold Out data

#### Note
When we set up a new config and we want to update the new config in the codebase. We can add force_save = True parameter.

```python
config = generate_pipeline_config(experiment, run_id, force_save=True)
```

## ☁️ Databricks Deployment

For production deployment on Databricks, see the [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md).

### Quick Databricks Deployment
```bash
# From the mlops/ directory
databricks bundle validate
databricks bundle deploy -t dev
databricks bundle run -t dev frogid_training_job
```

## 📈 MLflow Integration

All experiments are automatically tracked with:
- Training parameters and hyperparameters
- Model metrics and performance
- Training curves and confusion matrices
- Model artifacts and metadata

### MLflow Configuration

**Local Development (Default):**
- Uses SQLite database: `mlops/mlflow/tracking.db`
- Artifacts stored in: `mlruns/`
- File-based tracking (no server required)

**Server-based (Optional):**
```bash
mlflow server --backend-store-uri sqlite:///mlops/mlflow/tracking.db --default-artifact-root ./mlruns --port 5001
```

**Databricks (Production):**
- Connects to hosted MLflow server
- Configuration handled via environment variables

## 🧪 Testing

```bash
# Run GPU tests
python mlops/tests/test_gpu.py

# Run all tests (when pytest is configured)
pytest mlops/tests/
```

## 📋 Configuration

The pipeline uses dataclass-based configuration in `mlops/utils/config.py`:

- **`FrogDataSelectorConfig`**: Data selection and filtering
- **`FrogAudioDownloaderConfig`**: Audio download settings
- **`AudioLoaderConfig`**: Audio loading parameters
- **`BirdNETExtractorConfig`**: BirdNET feature extraction
- **`TFModelTrainerConfig`**: Training parameters

## 🔧 Advanced Usage

### Managing Dependencies with Poetry

**Adding New Dependencies:**
```bash
# Add runtime dependency
poetry add new-package

# Add development dependency
poetry add --group dev pytest-cov

# Add optional dependency group
poetry add --group visualization seaborn plotly
```

**Lock File Management:**
- `poetry.lock` ensures reproducible builds
- Commit `poetry.lock` to version control
- Run `poetry install` to install exact versions from lock file

**Updating Dependencies:**
```bash
# Update all dependencies
poetry update

# Update specific package
poetry update tensorflow

# Update within version constraints
poetry update --dry-run  # Preview changes first
```

### Custom Species Selection
Modify `dashboard/constants.py` to change target species:
```python
TARGET_SPECIES = [
    "Your_Species_1",
    "Your_Species_2",
    # ... add more species
]
```

### Custom Model Architecture
Create new models in `mlops/training/models/` following the existing pattern.

### Custom Preprocessing
Add new preprocessors in `mlops/feature_engineering/` and register them in the orchestrator.

## 📖 Documentation

Additional documentation available in `docs/`:
- ML Project Repository Structure & Git Approach
- Databricks MLOps Naming Conventions
- Repository Structure Quick Summary

## 🤝 Contributing

1. Create a feature branch from `main`
2. Make your changes following the existing code structure
3. Add tests for new functionality
4. Update documentation as needed
5. Submit a pull request

## 🆘 Troubleshooting

### Common Issues

1. **Audio Processing Errors**: Ensure FFmpeg is installed and accessible
2. **GPU Issues**: Check `mlops/tests/test_gpu.py` for GPU availability
3. **Memory Issues**: Reduce batch size in training configuration
4. **MLflow Tracking**: Ensure the tracking database path is correct

### Support

- Check the [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for Databricks-specific issues
- Review experiment notebooks in `experiments/` for usage examples
- Use the dashboard for data exploration and debugging

## 🔄 Version History

- **v0.1.0**: Initial release with BirdNET embeddings and 15-species classification