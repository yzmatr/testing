# FrogID ML Pipeline - Databricks Deployment Guide

## Overview
Your FrogID ML pipeline is now configured as a proper Databricks Asset Bundle following the [official Databricks documentation](https://docs.databricks.com/aws/en/dev-tools/bundles/). The bundle will automatically build the Python wheel, deploy all dependencies, and run the training job on GPU clusters.

## Prerequisites ✅

- ✅ **Databricks CLI v0.256.0+** installed with OAuth authentication
- ✅ **Poetry** for Python package management  
- ✅ **Data uploaded to DBFS** at `/Volumes/aus_museum_dbx_dev/frogid_ml/frogid_files/input/20250317.csv`
- ✅ **Bundle configuration** updated for wheel deployment

## Standard Deployment Workflow

### 1. Validate Bundle Configuration
```bash
cd mlops
databricks bundle validate
```
This checks your `databricks.yml` configuration for any errors.

### 2. Deploy to Databricks
```bash
databricks bundle deploy -t dev
```
This will:
- Build the Python wheel using `poetry build`
- Upload the wheel to your Databricks workspace
- Create the MLflow experiment and model registry
- Set up the training job with GPU cluster configuration

### 3. Run the Training Job
```bash
databricks bundle run -t dev frogid_training_job
```
This starts your BirdNET training pipeline on the GPU cluster.

### 4. Monitor in Databricks Workspace
- **Jobs**: Check job progress in the Databricks Jobs UI
- **MLflow**: View experiment tracking and model metrics
- **Model Registry**: See registered models in Unity Catalog

## Key Benefits of This Approach

✅ **No custom scripts needed** - Uses standard Databricks tooling  
✅ **Automatic dependency management** - Poetry handles Python packages  
✅ **Integrated MLflow** - Experiments and models automatically tracked  
✅ **GPU optimization** - Configured for ML workloads with T4 GPUs  
✅ **Production ready** - Follows Databricks MLOps best practices  

## Bundle Structure
```
mlops/
├── databricks.yml           # Bundle configuration
├── resources/
│   └── ml-artifacts-resource.yml  # MLflow setup
├── training/
│   └── train_birdnet_element_wise_averaging.py  # Main script
├── feature_engineering/      # Pipeline modules
└── evaluation/              # Model evaluation
```

## Environment Detection
Your training script automatically detects the environment:
- **Databricks**: Uses DBFS volumes and Databricks MLflow
- **Local**: Uses local paths and file-based MLflow tracking

## Next Steps
1. Run the deployment workflow above
2. Monitor the training job in Databricks
3. Check MLflow for experiment results and model artifacts
4. View the final model in Unity Catalog model registry

## Troubleshooting
- **Validation errors**: Check `databricks.yml` syntax
- **Deploy failures**: Verify CLI authentication and workspace permissions
- **Job failures**: Check job logs in Databricks workspace
- **Missing data**: Ensure CSV is uploaded to the correct DBFS volume path 