# Customer Churn Prediction

## Files
- `app.py`: complete machine-learning pipeline and Streamlit frontend
- `WA_Fn-UseC_-Telco-Customer-Churn.csv`: dataset used by the project
- `requirements.txt`: required Python libraries

## Run the project
1. Install Python.
2. Open a terminal in this folder.
3. Install packages:
   `pip install -r requirements.txt`
4. Start the frontend:
   `streamlit run app.py`

## What the project does
- Converts `TotalCharges` to numeric; blank values are handled by a median imputer.
- Removes `customerID`, which is an identifier rather than a useful behaviour feature.
- Splits data into 80% training and 20% testing using stratification.
- Uses preprocessing pipelines so imputation, scaling and one-hot encoding are learned from training folds only.
- Compares Logistic Regression, Decision Tree and Random Forest using 3-fold ROC-AUC cross-validation on the training set.
- Selects the model with the best mean cross-validation ROC-AUC.
- Reports final test accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix and classification report.
- Provides a form to predict churn for a new customer.

The test set is not used to choose the model. Accuracy is not guaranteed in advance; actual results depend on the dataset and model behaviour.
