# Credit Score Prediction Model

A machine learning project that predicts credit scores using Random Forest classification. The model categorizes customers into three credit score classes: Poor, Standard, and Good.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Features](#features)
- [Project Structure](#project-structure)
- [Results](#results)

## 🎯 Overview

This project implements a credit score prediction system using machine learning techniques. The model analyzes customer financial data to classify credit scores into three categories:

- **Poor (0)**: Low creditworthiness
- **Standard (1)**: Average creditworthiness  
- **Good (2)**: High creditworthiness

The solution uses a Random Forest Classifier with feature engineering and proper data preprocessing to achieve reliable predictions.

## 📊 Dataset

The project uses two main datasets:
- `train.csv`: Training data with labeled credit scores
- `test.csv`: Test data for final predictions

### Data Preprocessing Steps:
1. **Target Encoding**: Credit scores mapped from categorical to numerical values
2. **ID Column Removal**: Unnecessary identifier columns dropped
3. **Categorical Encoding**: All categorical features converted using Label Encoding
4. **Data Consistency**: Train and test data processed together to ensure consistent encoding

## 🚀 Installation

### Prerequisites
- Python 3.7+
- Required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Setup
1. Clone or download the project
2. Ensure your data files are in the `./data/` directory:
   ```
   ./data/train.csv
   ./data/test.csv
   ```
3. Run the main script

## 💻 Usage

### Basic Usage
```python
python credit_scoring_model.ipynb
```

### Key Components

#### 1. Data Loading & Exploration
```python
train_df = pd.read_csv("./data/train.csv")
test_df = pd.read_csv("./data/test.csv")
```

#### 2. Target Variable Preparation
```python
score_mapping = {"Poor": 0, "Standard": 1, "Good": 2}
train_df["Credit_Score"] = train_df["Credit_Score"].map(score_mapping)
```

#### 3. Feature Engineering
- Automatic categorical column detection
- Label encoding for all categorical features
- Consistent preprocessing across train and test sets

#### 4. Model Training
```python
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
```

## 📈 Model Performance

The model performance is evaluated using:

### Metrics
- **Accuracy Score**: Overall prediction accuracy
- **Classification Report**: Precision, recall, and F1-score for each class
- **Confusion Matrix**: Visual representation of prediction accuracy

### Visualizations
1. **Confusion Matrix Heatmap**: Shows actual vs predicted classifications
2. **Feature Importance Plot**: Displays the top 15 most important features

## 🔧 Features

### Key Functionalities:
- ✅ Automated data preprocessing pipeline
- ✅ Robust categorical variable handling
- ✅ Stratified train-validation split
- ✅ Feature importance analysis
- ✅ Comprehensive model evaluation
- ✅ Visual performance metrics
- ✅ Test set predictions

### Model Specifications:
- **Algorithm**: Random Forest Classifier
- **Estimators**: 200 trees
- **Validation Split**: 80/20 train-validation
- **Random State**: 42 (for reproducibility)

## 📁 Project Structure

```
credit-score-prediction/
│
├── data/
│   ├── train.csv           # Training dataset
│   └── test.csv            # Test dataset
│
├── credit_scoring_model.ipynb  # Main script
├── README.md               # Project documentation
```

## 📊 Results

### Model Output Includes:
1. **Training/Test Dataset Shapes**: Data dimension verification
2. **Data Cleaning Verification**: Before/after target encoding
3. **Accuracy Metrics**: Validation set performance
4. **Detailed Classification Report**: Per-class performance metrics
5. **Visual Analytics**:
   - Confusion matrix heatmap
   - Top 15 feature importance plot
6. **Sample Predictions**: First 10 predictions on test data

### Sample Output Format:
```
Train Shape: (X, Y)
Test Shape: (A, B)
Accuracy: 0.XX
Sample Predictions on Test Data: [0 1 2 1 0 ...]
```

## 🛠️ Model Configuration

### Hyperparameters:
- `n_estimators=200`: Number of trees in the forest
- `random_state=42`: Ensures reproducible results
- `test_size=0.2`: 20% data for validation
- `stratify=y`: Maintains class distribution in splits

### Preprocessing Pipeline:
1. Target variable mapping
2. ID column removal
3. Categorical feature identification
4. Label encoding transformation
5. Data consistency verification

## 📝 Notes

- The model uses stratified sampling to maintain class balance
- All categorical variables are automatically detected and encoded
- Feature importance helps identify key predictors
- The pipeline ensures consistent preprocessing between train and test data

## ⚠️ Important Considerations

- Ensure data files are properly formatted CSV files
- Check for missing values in your dataset before running
- The model assumes categorical variables can be meaningfully label-encoded
- Results may vary with different random states or data splits

