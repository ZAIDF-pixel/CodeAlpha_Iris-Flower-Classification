# CodeAlpha_Iris-Flower-Classification
# Iris Species Classification using Random Forest

This project trains a **Random Forest Classifier** on the classic Iris dataset to classify iris flowers into their respective species (*setosa*, *versicolor*, or *virginica*) based on physical measurements. It also provides an interactive CLI prompt to predict the species for new input measurements.

---

## 📌 Features

- **Data Loading & Prep:** Loads the Iris dataset and splits features (`SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, `PetalWidthCm`) and targets (`Species`).
- **Model Training:** Trains an ensemble `RandomForestClassifier` with deterministic train-test splits (`test_size=0.2`, `random_state=42`).
- **Model Evaluation:** Prints key classification metrics including overall accuracy, precision, recall, and F1-score.
- **Interactive Inference:** Prompts the user via terminal to enter continuous features for a custom sample and outputs the predicted species.

---

## 🛠️ Requirements & Installation

Make sure you have Python 3.7+ installed. Install the necessary dependencies using `pip`:

```bash
pip install pandas numpy scikit-learn
