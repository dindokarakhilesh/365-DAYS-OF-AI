# Day 104 — Customer Churn Prediction using ANN (Keras and TensorFlow)

A complete, end-to-end deep learning project on **tabular data**: from raw CSV to a trained, evaluated and saved neural network.

---

## 1. The problem

**Customer churn** = a customer stops using a company's service (here: a bank customer closes the account).

- Acquiring a new customer usually costs more than keeping an existing one, so predicting churn lets the business **act early** (offers, calls, support).
- Task type: **binary classification** (`Exited` = 1 churned, 0 stayed).
- Output we want: a **probability of churn** for each customer.

---

## 2. The dataset (Churn Modelling, 10,000 bank customers)

| Column | Type | Meaning |
|---|---|---|
| RowNumber, CustomerId, Surname | Identifier | No predictive meaning, drop them |
| CreditScore | Numeric | Credit score |
| Geography | Categorical | France / Germany / Spain |
| Gender | Categorical | Female / Male |
| Age | Numeric | Age in years |
| Tenure | Numeric | Years as a customer |
| Balance | Numeric | Account balance |
| NumOfProducts | Numeric | Number of bank products used |
| HasCrCard | Binary | Has a credit card |
| IsActiveMember | Binary | Active member |
| EstimatedSalary | Numeric | Estimated salary |
| **Exited** | **Target** | 1 = churned, 0 = stayed |

**Class imbalance:** only about **20%** of customers churn, so the classes are imbalanced.

---

## 3. Workflow overview

```
Load data → Explore → Clean/encode → Split → Scale → Build ANN → Compile → Train
      → Plot curves → Evaluate → Compare baselines → Improve → Predict → Save
```

---

## 4. Exploratory data analysis (what to check)

- **Shape, types, missing values, duplicates**
- **Class balance** (how many churn)
- **Churn rate by group** (geography, number of products, activity)
- **Distributions** (for example age of churned vs retained customers)

Why it matters: it tells you which preprocessing is needed and what a sensible baseline is.

### The accuracy paradox
If 80% of customers stay, a model that **always predicts "stays"** has **80% accuracy** while being useless. Any model must be judged against this **majority-class baseline**, and with metrics that look at the minority class.

---

## 5. Preprocessing

### 5.1 Drop identifier columns
`RowNumber`, `CustomerId` and `Surname` carry no useful signal. Keeping them can only add noise or cause overfitting.

### 5.2 Encode categorical variables
Neural networks need numbers.

| Column | Encoding |
|---|---|
| Geography (3 values) | **One-hot**, drop the first level, giving 2 columns |
| Gender (2 values) | One-hot with drop-first gives 1 column (`Gender_Male`) |

*Do not* encode Geography as 0, 1, 2: that would imply a fake order (France < Germany < Spain).

**Drop-first** avoids a redundant column (the dropped level is implied when all others are 0).

### 5.3 Train / validation / test split
- **Test set (20%)**: touched only at the very end, to estimate real-world performance.
- **Validation set** (20% of the remaining training data): used during training to watch for overfitting and to choose settings such as the threshold.
- Use **stratified** splitting so each part keeps the same churn rate.

### 5.4 Feature scaling
Neural networks train poorly when features have very different scales (salary around 100,000 vs tenure around 5). Use **standardization**: `x' = (x − mean) / std`.

> **Data leakage warning:** split **first**, then **fit the scaler on the training data only** and apply it to validation and test. Fitting on all the data lets information from the test set leak into training.

### Result
After preprocessing we have **11 input features**: 2 (Geography) + 1 (Gender) + 8 numeric/binary columns.

---

## 6. Building the ANN with Keras

```python
model = keras.Sequential([
    keras.Input(shape=(11,)),
    layers.Dense(11, activation="relu"),
    layers.Dense(11, activation="relu"),
    layers.Dense(1,  activation="sigmoid"),
])
```

| Part | Choice | Why |
|---|---|---|
| Input | 11 features | Number of columns after preprocessing |
| Hidden layers | `Dense(11, relu)` x 2 | ReLU is the default hidden activation; sizes are a starting point to tune |
| Output | `Dense(1, sigmoid)` | Probability of churn in (0, 1) |

### Parameter count (Day 101 formula)

| Layer | Weights | Biases | Total |
|---|---|---|---|
| 11 → 11 | 121 | 11 | 132 |
| 11 → 11 | 121 | 11 | 132 |
| 11 → 1 | 11 | 1 | 12 |
| **Total** | | | **276** |

`model.summary()` should show 276 trainable parameters.

---

## 7. Compile and train

```python
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    epochs=100, batch_size=32)
```

| Concept | Meaning |
|---|---|
| **Optimizer (Adam)** | Adaptive gradient-descent variant, a strong default |
| **Loss (binary cross entropy)** | The right loss for a sigmoid output (Day 99) |
| **Metric (accuracy)** | Reported for humans, not used for gradient updates |
| **Epoch** | One full pass over the training data |
| **Batch size** | Samples per weight update (32 means a weight update every 32 samples) |
| **Steps per epoch** | `ceil(n_train / batch_size)` |
| **validation_data** | Data used to monitor generalization, never to update weights |

### What happens inside `fit`
For every batch: **forward pass → loss → backpropagation → optimizer update** (the training loop from Day 93 onward; backpropagation itself comes in the next topics).

---

## 8. Reading the training curves

Plot loss and accuracy for **training vs validation** per epoch.

| Pattern | Meaning | What to do |
|---|---|---|
| Both losses decrease and stay close | Healthy learning | Continue |
| Training loss falls, **validation loss rises** | **Overfitting** | Early stopping, dropout, smaller model, more data |
| Both losses high and flat | **Underfitting** | Bigger model, more epochs, better features, tune learning rate |
| Very noisy curves | Learning rate or batch size issue | Lower the learning rate, bigger batches |

---

## 9. Evaluation

### 9.1 Prediction
The model outputs probabilities. Convert with a **threshold** (default 0.5):

```python
proba = model.predict(X_test).ravel()
pred = (proba >= 0.5).astype(int)
```

(`predict_classes` was removed from modern Keras. Use the line above.)

### 9.2 Confusion matrix

|  | Predicted: stays | Predicted: churns |
|---|---|---|
| **Actually stays** | True Negative (TN) | False Positive (FP) |
| **Actually churns** | False Negative (FN) | True Positive (TP) |

### 9.3 Metrics

| Metric | Formula | Question it answers |
|---|---|---|
| Accuracy | (TP + TN) / total | How often is the model right overall? (misleading when imbalanced) |
| **Precision** | TP / (TP + FP) | Of the customers flagged as churners, how many really churn? |
| **Recall** | TP / (TP + FN) | Of the real churners, how many did we catch? |
| **F1** | 2 · P · R / (P + R) | Balance of precision and recall |
| **ROC AUC** | Area under the ROC curve | How well does the model *rank* churners above non-churners, regardless of threshold? (0.5 = random, 1.0 = perfect) |

**For churn, recall on the churn class usually matters most**: missing a real churner (FN) means a lost customer, while a false alarm (FP) only costs a retention offer.

---

## 10. Baselines: always compare

| Baseline | Why |
|---|---|
| **Majority class** | The "do nothing" accuracy floor |
| **Logistic regression** | Simple, interpretable linear model |
| **Random forest / gradient boosting** | Often the strongest models on tabular data |

> On tabular data, tree ensembles often match or beat neural networks. An ANN is worth keeping only if it **clearly improves** the metric that matters, or if you need what neural networks offer (for example combining with other data types, or embeddings).

---

## 11. Improving the model

| Technique | What it does |
|---|---|
| **More/larger layers** | More capacity (watch for overfitting) |
| **Dropout** | Randomly switches off a fraction of neurons in each training step, so the network cannot depend on any single neuron (regularization) |
| **Early stopping** | Stops training when validation loss stops improving, and restores the best weights |
| **Class weights** | Gives the rare class a larger loss weight (`balanced` weight = `n / (2 · n_class)` for 2 classes) |
| **Resampling** | Oversample churners or undersample stayers (SMOTE etc.) |
| **L2 regularization** | Penalizes large weights |
| **Learning-rate tuning / schedules** | Faster, more stable convergence |
| **Hyperparameter search** | Systematically try layer sizes, dropout, learning rate (Keras Tuner, Optuna) |
| **Feature engineering** | For example balance-to-salary ratio, age groups |

### Threshold tuning
The best cutoff is rarely exactly 0.5 for imbalanced data.

- Pick the threshold on the **validation set** (not on the test set!), for example the one maximizing F1 or a business-cost objective.
- Then report the final result once on the test set.

**Cost-based thresholding:** if a retention offer costs `c_fp` and a lost customer costs `c_fn` (usually much larger), choose the threshold that **minimizes** `c_fp · FP + c_fn · FN` on validation data.

> Class weights and threshold tuning usually trade **accuracy for recall**. That is acceptable when missing churners is costly, but say so explicitly when reporting results.

---

## 12. Using the model on a new customer

New data must go through the **exact same preprocessing** (same encoder categories, same scaler mean and std from training).

```python
x_new = preprocess.transform(new_customer_df)      # fitted on the training data
p = model.predict(x_new)[0, 0]                     # churn probability
```

Keep the encoder and scaler inside **one preprocessing object** (for example a scikit-learn `ColumnTransformer`) so nothing is forgotten.

---

## 13. Saving and loading

```python
model.save("churn_model.keras")
joblib.dump(preprocess, "preprocess.joblib")

model = keras.models.load_model("churn_model.keras")
preprocess = joblib.load("preprocess.joblib")
```

**Save the model and the preprocessing together**, plus the chosen threshold. A model without its preprocessing cannot be used correctly.

---

## 14. Common mistakes

1. **Scaling before splitting** (data leakage).
2. **Judging by accuracy only** on imbalanced data.
3. **Tuning the threshold or hyperparameters on the test set**.
4. **Forgetting to scale or encode new data** at prediction time.
5. **Label-encoding nominal categories** (creating a fake order).
6. **Leaving ID columns in** the features.
7. **Using `softmax` with one output unit** or `sigmoid` with the wrong loss.
8. **Trusting one run**: neural network results vary with random seeds, so repeat or fix seeds, and compare against baselines.

---

## 15. Keras quick reference

| Call | Purpose |
|---|---|
| `keras.Sequential([...])` | Stack layers in order |
| `keras.Input(shape=(n,))` | Declare the input size |
| `layers.Dense(units, activation)` | Fully connected layer |
| `layers.Dropout(rate)` | Regularization |
| `model.summary()` | Layers, output shapes, parameter counts |
| `model.compile(optimizer, loss, metrics)` | Configure training |
| `model.fit(X, y, validation_data, epochs, batch_size, callbacks, class_weight)` | Train |
| `model.evaluate(X, y)` | Loss and metrics on a dataset |
| `model.predict(X)` | Output probabilities |
| `keras.callbacks.EarlyStopping(...)` | Stop at the best validation loss |
| `model.save(path)` / `keras.models.load_model(path)` | Persist and restore |

---

## 16. Key takeaways

- Churn prediction is **imbalanced binary classification**: judge it with **recall, precision, F1 and ROC AUC**, not accuracy alone.
- **Preprocess carefully**: drop IDs, one-hot encode, split first, then scale using training statistics only.
- A simple ANN: `Dense(ReLU) → Dense(ReLU) → Dense(sigmoid)` with Adam and binary cross entropy.
- Watch **training vs validation curves** to detect overfitting; use **dropout, early stopping, class weights**.
- **Tune the threshold on validation data**, and compare with **baselines** (logistic regression, random forest).
- **Save the model and the preprocessing together.**

## 17. Revision questions

1. Why is accuracy a poor metric for churn prediction? What is the majority-class baseline?
2. Why do we drop `RowNumber`, `CustomerId` and `Surname`?
3. Why one-hot encode Geography instead of mapping it to 0, 1, 2?
4. What is data leakage, and how does fitting the scaler on all the data cause it?
5. Count the parameters of an `11 → 11 → 11 → 1` network.
6. Why sigmoid in the output layer and binary cross entropy as the loss?
7. How do you recognize overfitting in the training curves, and name three ways to reduce it?
8. Define precision and recall, and explain which one matters more for churn.
9. How and where should the decision threshold be chosen?
10. Why must the preprocessing object be saved together with the model?

---
*Next topic: continue the Deep Learning course roadmap (Loss functions in deep learning).*
