# LSTM Sentiment Classifier - Clean Project Structure

## 📁 Core Application Files

### Main Entry Points
- **`app.py`** - Streamlit web UI (recommended for demos)
- **`run_api.py`** - FastAPI production server
- **`lstm_sentiment.py`** - Command-line interface (CLI)

### Core Scripts
- **`train.py`** - Model training pipeline
- **`evaluate.py`** - Model evaluation and metrics
- **`predict.py`** - Standalone prediction script
- **`download_glove.py`** - Download GloVe embeddings
- **`download_imdb_data.py`** - Download IMDB dataset
- **`config.py`** - Configuration utilities

## 📂 Directory Structure

```
lstm_model/
├── src/                          # Source code modules
│   ├── api/                     # API implementation
│   ├── data/                    # Data processing
│   ├── evaluation/              # Evaluation tools
│   ├── inference/               # Inference engine
│   ├── models/                  # Model architectures
│   ├── optimization/            # Model optimization
│   ├── training/                # Training pipeline
│   └── utils/                   # Utilities
├── tests/                        # Unit tests
├── configs/                      # YAML configurations
│   └── examples/                # Example configs
├── models/                       # Trained models
│   ├── improved_lstm_model_20251106_003134.pth
│   └── improved_lstm_model_20251106_003134_vocabulary.pth
├── checkpoints/                  # Training checkpoints
│   └── best_model.pth
├── data/                         # Datasets
│   ├── imdb/                    # IMDB movie reviews
│   └── glove/                   # GloVe embeddings
├── logs/                         # Training logs
├── examples/                     # Usage examples
└── evaluation_results/           # Evaluation outputs
```

## 🚀 Deployment Files

- **`Dockerfile`** - Container configuration
- **`.dockerignore`** - Docker build exclusions
- **`cloudbuild.yaml`** - GCP Cloud Build config
- **`deploy.ps1`** - Windows deployment script
- **`deploy.sh`** - Linux/Mac deployment script

## 📋 Configuration Files

- **`requirements.txt`** - Python dependencies
- **`requirements_api.txt`** - Additional API dependencies
- **`api_config.yaml`** - API server configuration
- **`.gitignore`** - Git exclusions

## 📚 Documentation

- **`README.md`** - Main project documentation
- **`DEPLOYMENT_README.md`** - Optimization & API deployment guide
- **`DEPLOYMENT_GCP.md`** - Google Cloud Platform deployment
- **`IMPROVEMENTS_README.md`** - Enhancement history
- **`PROJECT_STRUCTURE.md`** - This file

## 🎯 Quick Start Commands

### Run Web UI
```powershell
streamlit run app.py
```

### Run API Server
```powershell
python run_api.py
```

### Train Model
```powershell
python train.py --config configs/examples/quick_training.yaml
```

### Make Predictions
```powershell
python predict.py -m models/improved_lstm_model_20251106_003134.pth -v models/improved_lstm_model_20251106_003134_vocabulary.pth -t "Great movie!"
```

### Evaluate Model
```powershell
python evaluate.py -m models/improved_lstm_model_20251106_003134.pth -v models/improved_lstm_model_20251106_003134_vocabulary.pth -d data/imdb
```

## 🧹 Cleaned Up (Removed)

The following redundant files were removed to keep the project clean:

### Test Scripts (moved to tests/)
- `debug_model.py`
- `simple_test.py`
- `test_100_reviews.py`
- `test_api.py`
- `test_augmentation_integration.py`
- `test_embedding_integration.py`
- `test_improved_model.py`

### Redundant Training Scripts
- `retrain_improved_model.py`
- `retrain_model.py`
- `train_advanced_optimization.py`
- `train_production_model.py`

### Redundant Evaluation/Optimization
- `comprehensive_evaluation.py`
- `continuous_improvement.py`
- `hyperparameter_optimization.py`
- `run_hyperparameter_optimization.py`
- `benchmark_inference.py`

### Old Model Checkpoints
- `best_model_epoch_1.pth`
- `best_model_epoch_2.pth`
- Older model versions (kept only latest: 20251106_003134)

### Cache Files
- All `__pycache__/` directories
- All `.pyc` files

## 💡 Best Practices

1. **Virtual Environment**: Always activate venv before running
2. **Latest Model**: Use `improved_lstm_model_20251106_003134.pth`
3. **Configuration**: Use YAML configs in `configs/examples/`
4. **Testing**: Run tests from `tests/` directory
5. **Deployment**: Use Docker for production deployment

## 🔄 Git Status

Clean project structure ready for:
- Version control
- Deployment
- Collaboration
- Production use

Total reduction: **~117 files/directories removed**
