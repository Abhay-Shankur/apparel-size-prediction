
# Apparel Size Prediction and Suggestion System

This project aims to predict and recommend the correct apparel size for users based on their measurements, reducing the likelihood of returns or exchanges due to incorrect sizing. By analyzing previously purchased sizes and their return/exchange data, the model helps in suggesting the optimal size to purchase.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Project Workflow](#project-workflow)
  - [Phase 1: Data Preprocessing](#phase-1-data-preprocessing)
  - [Phase 2: Model Comparison and Analysis](#phase-2-model-comparison-and-analysis)
  - [Phase 3: Building the Suggestion System](#phase-3-building-the-suggestion-system)
- [Usage](#usage)
- [License](#license)

## Project Overview
This system is designed to:
1. Analyze return/exchange data to understand patterns in size selection.
2. Use machine learning models to predict and recommend the most appropriate size based on user measurements.

The project consists of three main phases: data preprocessing, model comparison, and model selection for the final suggestion system.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Abhay-Shankur/apparel-size-prediction.git
   cd apparel-size-prediction
   ```

2. Install the required dependencies using `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

## Project Workflow

### Phase 1: Data Preprocessing (`preprocessing.ipynb`)
In this phase, we clean and preprocess the raw dataset to ensure it is ready for training. We handle missing values, remove outliers, and standardize the data.

- **Notebook**: `preprocessing.ipynb`
- **Input**: Raw dataset file (e.g., `synthetic_size_chart_data.csv`)
- **Output**: Cleaned dataset file (`cleaned_size_chart_data.csv`)

**Steps**:
1. Open the `preprocessing.ipynb` notebook.
2. Run each cell to clean the data and apply necessary transformations.
3. The notebook will output a cleaned dataset, saved as `cleaned_size_chart_data.csv`.

### Phase 2: Model Comparison and Analysis (`model_comparison.ipynb`)
In this phase, we compare different machine learning models to identify the best-performing one for our use case.

- **Notebook**: `model_comparison.ipynb`
- **Input**: Cleaned dataset (`cleaned_size_chart_data.csv`)
- **Output**: Performance metrics for various models

**Steps**:
1. Open the `model_comparison.ipynb` notebook.
2. Run the cells to load the cleaned dataset and train multiple machine learning models (e.g., Random Forest, Gradient Boosting, SVM, etc.).
3. The notebook will output accuracy and loss metrics for each model on both training and testing data.
4. Analyze the results to select the best model for the suggestion system.

### Phase 3: Building the Suggestion System (`model_prediction.ipynb`)
In this phase, we select the best-performing model from Phase 2 and use it to create a prediction and suggestion system. The model will take user measurements as input and recommend the appropriate apparel size.

- **Notebook**: `model_prediction.ipynb`
- **Input**: Cleaned dataset (`cleaned_size_chart_data.csv`)
- **Output**: Prediction of the recommended size for a new user input

**Steps**:
1. Open the `model_prediction.ipynb` notebook.
2. Load the trained model and preprocess new user input data as required.
3. Run the cells to predict the recommended apparel size based on the input measurements.

## Usage
To use this project, follow the phases as described. After completing the data preprocessing and model training steps, you can input custom measurements in the suggestion system to receive recommended sizes.

## License
This project is licensed under the MIT License. See `LICENSE` for more information.
