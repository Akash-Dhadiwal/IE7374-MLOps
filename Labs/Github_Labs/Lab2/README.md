# Lab 2 — GitHub Actions for Model Training, Evaluation & Calibration

## Changes made for the lab submission
- **Dataset** changed to `Iris` (from synthetic / RCV1 in original).
- **Model** changed to `LogisticRegression` (instead of RandomForest/DecisionTree).
- **Metrics** now include accuracy, weighted F1, precision, and confusion matrix.
- `train_model.py` saves the test split in `data/` so evaluation uses the same held-out set.
- Model filename kept as `model_{timestamp}_dt_model.joblib` for workflow compatibility.


## Getting Started

1. **Fork this Repository**: Click the "Fork" button at the top right of this [repository](https://github.com/raminmohammadi/MLOps/) to create your own copy.
2. **Clone Your Repository**:
   ```bash
   git clone https://github.com/your-username/your-forked-repo.git
   cd your-forked-repo
   ````

3. Set up a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

---

## Running the Workflow

### Train and Evaluate Locally

1. Generate a timestamp and train the model:

   ```bash
   TS=$(date +%Y%m%d%H%M%S)
   python src/train_model.py --timestamp "$TS"
   ```

2. Evaluate the model:

   ```bash
   python src/evaluate_model.py --timestamp "$TS"
   ```

3. (Optional) Calibrate model:

   ```bash
   python src/calibrate_model.py --timestamp "$TS" --method isotonic
   ```

4. Check outputs:

   ```bash
   ls -1 models
   ls -1 metrics
   ```

**Expected outputs:**

* `models/model_{timestamp}_dt_model.joblib`
* `metrics/{timestamp}_metrics.json`
* (if calibrated) `models/model_{timestamp}_cal_isotonic.joblib` and `metrics/{timestamp}_calibration_isotonic.json`

---

### Push Your Changes

1. Commit your changes and push to your forked repository.
2. GitHub Actions will automatically trigger workflows for model training, evaluation, and calibration.

---

## GitHub Actions Workflow Details

The workflow consists of the following steps:

* **Generate and Store Timestamp**: A timestamp is generated and stored for versioning.
* **Model Training**: The `train_model.py` script trains a Logistic Regression model on the Iris dataset and stores the model in `models/`.
* **Model Evaluation**: The `evaluate_model.py` script evaluates the model on a held-out test split, calculating accuracy, weighted F1, precision, and confusion matrix, and stores results in `metrics/`.
* **Store and Version the Model**: The trained model is saved with a timestamp-based filename.
* **Commit and Push Changes**: Metrics and updated model are committed to the repository.

---

## Model Calibration Workflow

The `model_calibration_on_push.yml` workflow ensures that the model's predicted probabilities are well-calibrated. Calibration methods like Platt scaling or isotonic regression can be applied, and calibrated models are saved separately in `models/` for comparison.

---

## Integration with Model Training

The calibration workflow works seamlessly with the main training workflow. Once the model is trained, the calibration workflow can adjust predicted probabilities and generate updated metrics for downstream use.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Acknowledgments

* Uses GitHub Actions for continuous integration and deployment.
* Model training and evaluation powered by Python, scikit-learn, and MLflow.

---

