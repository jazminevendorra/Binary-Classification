# Lettuce Health Classifier Project Architecture

## Overview
This project is split into modular Python scripts for clarity and maintainability. Each script has a single responsibility, making the workflow easy to follow and debug.

## File Structure

```
Farming System v2/
├── Lettuce_disease_datasets/           # Original dataset (with subfolders per disease/class)
├── train/                              # Training data (created by data_preparation.py)
├── validation/                         # Validation data (created by data_preparation.py)
├── test/                               # Test data (created by data_preparation.py)
├── data_preparation.py                 # Script for dataset splitting and folder setup
├── model_training.py                   # Script for model definition, training, evaluation, and saving
├── inference.py                        # Script for loading the trained model and classifying a new image
├── healthy_vs_non_healthy_classifier.keras  # Saved trained model (output of model_training.py)
├── lettuce-health-app/                 # Streamlit web app for deployment
└── ARCHITECTURE.md                     # This architecture and documentation file
```

## How the Files Connect

- **data_preparation.py**
    - Reads the original dataset (`Lettuce_disease_datasets/`).
    - Splits images into `train/`, `validation/`, and `test/` folders with `healthy` and `non_healthy` subfolders.
    - Run this script first to set up the data folders.

- **model_training.py**
    - Uses the `train/`, `validation/`, and `test/` folders to create data generators.
    - Defines, trains, and evaluates the CNN model.
    - Saves the trained model as `healthy_vs_non_healthy_classifier.keras`.
    - Run this script after data preparation is complete.

- **inference.py**
    - Loads the trained `.keras` model.
    - Classifies a user-supplied image and displays the result.
    - Can be used for quick testing or as a backend utility for the Streamlit app.

- **lettuce-health-app/**
    - Contains the Streamlit web application for interactive deployment and user interface.
    - The app loads the `.keras` model and uses the same inference logic as in `inference.py`.

## Typical Workflow

1. **Install dependencies** (first time only):

   Open a terminal in the main project folder (`Farming System v2`) and run:
   ```sh
   pip install -r requirements.txt
   ```
   This will install all Python packages needed for data preparation, training, inference, and the Streamlit app.

2. **Run `data_preparation.py`**
   - Splits and organizes the dataset into `train/`, `validation/`, and `test/` folders with the correct subfolders.

3. **Run `model_training.py`**
   - Trains and evaluates the model, saving it as `healthy_vs_non_healthy_classifier.keras`.

4. **Use `inference.py`**
   - Tests the trained model on new images to verify predictions.

5. **(Optional) Deploy with Streamlit**
   - Use the app in `lettuce-health-app/` for an interactive interface. Make sure the `.keras` model file is accessible to the app.

---

**Following this order will help you avoid missing dependencies and ensure each step has the required files and folders.**
