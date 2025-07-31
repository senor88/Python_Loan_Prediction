# Loan Prediction Model

## Project Overview
This project develops a machine learning model to predict loan approval status based on applicant data. Using a dataset with features like gender, marital status, education, income, loan amount, credit history, and property area, the model aims to classify whether a loan application will be approved (`Y`) or rejected (`N`).

## Dataset
- **Source**: Loan Prediction Dataset (`Loan Prediction Dataset.csv`)
- **Features**: Includes categorical (Gender, Married, Education, etc.) and numerical (ApplicantIncome, LoanAmount, etc.) attributes.
- **Target**: Loan_Status (Y/N)

## Key Steps
1. **Data Preprocessing**:
   - Handled missing values by filling numerical columns with mean values and categorical columns with mode.
   - Applied label encoding to categorical features (Gender, Married, Education, etc.).
   - Dropped irrelevant columns to improve model efficiency.
   - Applied logarithmic transformations to numerical features (e.g., ApplicantIncomeLog, LoanAmountLog) to normalize data.

2. **Exploratory Data Analysis (EDA)**:
   - Visualized correlations between numerical features using a heatmap to identify relationships.

3. **Model Training**:
   - Split data into training (75%) and testing (25%) sets.
   - Evaluated multiple models:
     - **Logistic Regression**: Achieved 77.27% accuracy, 80.95% cross-validation score.
     - **Decision Tree**: 73.38% accuracy, 70.53% cross-validation score.
     - **Random Forest**: 78.57% accuracy, 78.02% cross-validation score.
     - **Extra Trees**: 74.68% accuracy, 76.06% cross-validation score.
   - Performed hyperparameter tuning on Random Forest (n_estimators=100, min_samples_split=25, max_depth=7, max_features=1), resulting in 76.62% accuracy and 80.30% cross-validation score.

## Tools and Libraries
- **Python**: pandas, numpy, seaborn, matplotlib
- **Machine Learning**: scikit-learn (LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, ExtraTreesClassifier, train_test_split, cross_val_score)
- **Environment**: Jupyter Notebook

## Results
The Random Forest model with tuned hyperparameters showed robust performance, balancing accuracy and generalization. Logistic Regression also performed well, indicating linear relationships in the data. The project demonstrates effective data preprocessing, model selection, and evaluation for a classification task.

## How to Run
1. Clone the repository: `git clone <repo-link>`
2. Install dependencies: `pip install pandas numpy seaborn matplotlib scikit-learn`
3. Run the Jupyter Notebook: `jupyter notebook Python_Loan_Prediction.ipynb`

## Future Improvements
- Explore additional feature engineering (e.g., interaction terms).
- Test advanced models like XGBoost or LightGBM.
- Address class imbalance if present in the dataset.
- Deploy the model as a web application for real-time predictions.
