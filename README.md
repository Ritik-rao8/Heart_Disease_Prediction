# Heart Disease Prediction using Machine Learning

A beginner-friendly project using Logistic Regression and Random Forest to predict heart disease labels from clinical data.

## Dataset

`heart_disease_cleveland.csv` contains 303 records, 13 input features, and a binary target. We assume `0` means disease absent and `1` means disease present.

## Steps

1. Data cleaning
2. Exploratory Data Analysis (EDA)
3. Data preprocessing
4. Model training
5. Model evaluation
6. Feature importance analysis

## Models and Results

| Model | Test Accuracy | ROC-AUC |
| --- | --- | --- |
| Logistic Regression | 86.89% | 0.9578 |
| Random Forest | 88.52% | 0.9389 |

Evaluation includes confusion matrices, ROC curves, precision, recall, and F1-score.

The leading Random Forest features are thallium scan result, chest pain category, maximum heart rate, vessel count, and ST depression.

## Run the Notebook

```bash
pip install -r requirements-notebook.txt
jupyter notebook Heart_Disease_Prediction.ipynb
```

Open the notebook and run all cells from top to bottom.

## Tools Used

Python, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, and Jupyter Notebook.

For educational use only. Predictions are not medical diagnoses.
