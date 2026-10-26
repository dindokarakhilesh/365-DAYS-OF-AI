# Graduate Admission Prediction Using ANN

## 1. Introduction
Graduate Admission Prediction is a supervised machine-learning project that estimates an applicant's `Chance of Admit` from available academic and application-related features. An Artificial Neural Network (ANN) is used to learn relationships between these input features and the target value.

This project is intended for learning and experimentation. It should not be used as the sole basis for real admissions decisions.

## 2. Problem Statement
Given historical applicant records, train a model that predicts a continuous admission-chance value for a new record.

Because the target (`Chance of Admit`) is numeric and continuous, this is a **regression** problem rather than a classification problem.

## 3. Dataset
A commonly used educational dataset is named `Admission_Predict.csv`. It often contains these columns:

| Feature | Meaning |
|---|---|
| GRE Score | Graduate Record Examination score |
| TOEFL Score | TOEFL score |
| University Rating | Rating value for the university |
| SOP | Statement of Purpose rating |
| LOR | Letter of Recommendation rating |
| CGPA | Undergraduate academic performance measure |
| Research | Research experience indicator, often 0 or 1 |
| Chance of Admit | Target value to predict |

Some versions include `Serial No.`, which is an identifier and should generally be removed from the input features. Check the actual file's column names and definitions before training.

## 4. What Is an Artificial Neural Network?
An Artificial Neural Network is a machine-learning model inspired in a simplified way by connected neurons. It contains layers of units called neurons. Each neuron combines inputs using learned weights and a bias, then applies an activation function.

A typical dense layer calculates:

\[
z = Wx + b
\]

\[
a = f(z)
\]

where:
- `x` is the input vector,
- `W` is a matrix of learned weights,
- `b` is a bias,
- `f` is an activation function,
- `a` is the layer's output.

## 5. Architecture Used
The notebook uses a feed-forward ANN:

1. **Input layer:** receives the numeric applicant features.
2. **Hidden layer 1:** 64 neurons with ReLU activation.
3. **Dropout layer:** randomly disables a small fraction of units during training to help reduce overfitting.
4. **Hidden layer 2:** 32 neurons with ReLU activation.
5. **Output layer:** one neuron with a linear output for regression.

The architecture is a starting point, not necessarily the best architecture for every version of the dataset.

## 6. Data Preprocessing
### Removing identifiers
A serial number identifies a row rather than describing an applicant, so it is excluded when present.

### Handling missing values
The example notebook drops rows with missing values for simplicity. Larger projects may use imputation, but the choice should be made carefully.

### Feature scaling
Features such as GRE scores and CGPA use different numerical ranges. `StandardScaler` transforms each feature using the training set's mean and standard deviation:

\[
z = \frac{x - \mu}{\sigma}
\]

The scaler is fitted only on the training set and then applied to both training and test data. This prevents information from the test set leaking into preprocessing.

## 7. Training and Loss Function
The dataset is split into training and test sets. The ANN learns from the training set. A validation portion of the training data is used to monitor generalization during training.

The model uses:
- **Optimizer:** Adam, which updates weights to reduce the loss.
- **Loss:** Mean Squared Error (MSE).
- **Metric:** Mean Absolute Error (MAE).
- **Early stopping:** stops training when validation loss does not improve for a specified number of epochs and restores the best weights.

MSE is:

\[
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
\]

## 8. Evaluation Metrics
### Mean Absolute Error (MAE)
The average absolute difference between actual and predicted values:

\[
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|
\]

Lower MAE generally indicates predictions closer to the observed targets.

### Root Mean Squared Error (RMSE)
The square root of the mean squared prediction error:

\[
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
\]

RMSE gives relatively more weight to larger errors.

### R-squared (R²)
R² compares the model with a baseline that predicts the target mean. A value closer to 1 generally indicates a better fit, while a negative value means the model performs worse than that baseline on the evaluated data.

Metrics should be interpreted on the target scale and in context; a single train/test split is not a guarantee of future performance.

## 9. Prediction Workflow
1. Load the trained model.
2. Load the saved scaler.
3. Arrange new applicant features using the exact training column names and order.
4. Transform the features using the saved scaler.
5. Pass the transformed values to the model.
6. Interpret the output as a model estimate, not a guarantee or a calibrated probability unless calibration has been separately established.

## 10. Tools and Libraries
- **Python:** programming language.
- **Pandas:** loading and preparing tabular data.
- **NumPy:** numerical operations.
- **Matplotlib:** plotting training curves and prediction comparisons.
- **Scikit-learn:** train/test split, scaling, and evaluation metrics.
- **TensorFlow/Keras:** building and training the ANN.
- **Joblib:** saving the fitted scaler.

## 11. How to Run
1. Install Python and Jupyter support, or use VS Code with the Jupyter extension.
2. Put `code.ipynb` and `Admission_Predict.csv` in the same folder.
3. Install the required packages:

   ```bash
   python -m pip install pandas numpy matplotlib scikit-learn tensorflow joblib jupyter
   ```

4. Open `code.ipynb` in Jupyter or VS Code.
5. Run cells from top to bottom.
6. If the CSV uses a different filename or target-column name, update the corresponding values in the notebook.

TensorFlow installation compatibility depends on the operating system and Python version. If installation fails, use a Python version supported by the TensorFlow release you install.

## 12. Limitations and Responsible Use
- Historical data may contain bias, errors, or sampling limitations.
- A high score on one test split does not prove the model generalizes to all applicants or institutions.
- Admission outcomes depend on many factors not represented in the dataset.
- `Chance of Admit` is a dataset label; model output should not automatically be interpreted as a verified probability.
- Do not use this educational model as the sole basis for decisions affecting applicants. Validate data quality, fairness, calibration, and performance before any consequential use.

## 13. Conclusion
This project demonstrates a complete regression workflow: loading data, preprocessing features, creating an ANN, training with validation and early stopping, evaluating results, generating predictions, and saving the model and scaler. It provides a foundation for further experiments with baselines, cross-validation, feature analysis, and model tuning.
