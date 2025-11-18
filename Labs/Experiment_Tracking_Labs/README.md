# README

This README explains the changes **I made** to the original XGBoost-based dermatology classifier. I replaced the original machine learning pipeline with a Keras 3 neural network model and updated the entire workflow to ensure compatibility with TensorFlow/Keras 3 and Weights & Biases (W&B).

---

## ✅ Summary of What I Changed

### **1. I replaced the XGBoost model with a Keras deep learning model**

The original notebook used:

* `xgboost.DMatrix` for data
* `xgb.train()` for training
* `wandb.xgboost.WandbCallback()` for logging

I completely removed XGBoost and built a new **Keras Sequential model** trained with `model.fit()`.

### **2. I removed all XGBoost-related code**

I deleted:

* `import xgboost as xgb`
* DMatrix conversions
* XGBoost parameter configuration (`objective`, `eta`, `max_depth`, etc.)
* XGBoost callbacks
* `bst.predict()`

The pipeline is now fully deep-learning–based.

### **3. I updated the W&B callback for Keras 3 compatibility**

Keras 3 removed the `save_format` argument, which caused W&B to crash. To fix this, I disabled all model saving features inside the callback:

```python
wandb.keras.WandbCallback(
    log_model=False,
    save_model=False,
    save_graph=False
)
```

This prevents W&B from calling deprecated Keras APIs.

### **4. I replaced XGBoost evaluation with Keras predictions**

The original code computed predictions using `bst.predict()` and manually calculated error rate.

I updated this to:

* use `model.predict()`
* apply softmax + argmax to get class predictions
* compute accuracy/error using NumPy

### **5. I changed the confusion matrix logging**

The original version used:

```python
wandb.sklearn.plot_confusion_matrix(...)
```

Since I moved to Keras predictions, I switched to generating predictions manually and logging them in a W&B-friendly format.

### **6. I cleaned up and simplified data loading**

I kept the original dataset download but reorganized the processing:

* simplified NumPy slicing
* prepared data for neural network input instead of `DMatrix`

---

## 📌 Before vs After — High-Level Overview (From My Perspective)

| Area             | Original Code      | My Updated Code                  |
| ---------------- | ------------------ | -------------------------------- |
| Model Type       | XGBoost multiclass | Keras 3 neural network           |
| Data Pipeline    | DMatrix            | NumPy arrays / tensors           |
| Training         | `xgb.train()`      | `model.fit()`                    |
| W&B Callback     | XGBoost callback   | Keras callback (saving disabled) |
| Model Saving     | Enabled by default | Disabled to avoid Keras 3 errors |
| Evaluation       | `bst.predict()`    | `model.predict()`                |
| Confusion Matrix | sklearn helper     | Manual / custom logging          |

