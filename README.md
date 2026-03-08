
# 🌸 Iris Flower Multiclass Classification

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-yellow)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A machine learning project that classifies iris flowers into **three different species** based on flower measurements using **classical ML algorithms**.

The model is trained on the famous **Iris dataset** and demonstrates a complete **supervised learning pipeline** including data exploration, training, and prediction.

---

# 📌 Project Overview

The **Iris dataset** is one of the most widely used datasets for introducing machine learning classification problems.

In this project, machine learning models are trained to predict the species of an iris flower based on measurements of its **sepal and petal dimensions**.

The classification system predicts one of the following species:

* **Setosa**
* **Versicolor**
* **Virginica**

---

# 🏗 Project Workflow

```text
Dataset Loading
        │
        ▼
Data Exploration
        │
        ▼
Train-Test Split
        │
        ▼
Model Training
(KNN / SVM)
        │
        ▼
Model Evaluation
        │
        ▼
Prediction
```

---

# 📊 Dataset

The project uses the **Iris Flower Dataset** available in **Scikit-Learn**.

### Dataset Properties

| Property      | Value |
| ------------- | ----- |
| Total Samples | 150   |
| Features      | 4     |
| Classes       | 3     |

### Features

| Feature      | Description     |
| ------------ | --------------- |
| Sepal Length | Length of sepal |
| Sepal Width  | Width of sepal  |
| Petal Length | Length of petal |
| Petal Width  | Width of petal  |

### Target Classes

| Class | Species    |
| ----- | ---------- |
| 0     | Setosa     |
| 1     | Versicolor |
| 2     | Virginica  |

---

# ⚙️ Technologies Used

| Technology       | Purpose                     |
| ---------------- | --------------------------- |
| Python           | Programming language        |
| Scikit-Learn     | Machine learning algorithms |
| NumPy            | Numerical operations        |
| Pandas           | Data handling               |
| Matplotlib       | Data visualization          |
| Jupyter Notebook | Model development           |

---

# 📂 Project Structure

```
IRIS-MULTICLASSIFICATION
│
├── IRIS_MULTICLASSIFICATION.ipynb
├── README.md
```

---

# 🔬 Implementation

## 1️⃣ Import Libraries

The required libraries are imported for machine learning and data analysis.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
```

---

# 2️⃣ Dataset Loading

The dataset is loaded from the Scikit-Learn library.

```python
iris = load_iris()
```

The dataset contains:

* Feature matrix (`iris.data`)
* Target labels (`iris.target`)
* Feature names
* Target class names

---

# 3️⃣ Data Exploration

The notebook explores dataset properties such as:

* dataset shape
* feature names
* sample data
* class distribution

Example:

```python
iris.data.shape
```

Output:

```
(150, 4)
```

Meaning:

* **150 samples**
* **4 features**

---

# 4️⃣ Train-Test Split

The dataset is divided into training and testing sets.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2
)
```

| Dataset       | Portion |
| ------------- | ------- |
| Training Data | 80%     |
| Testing Data  | 20%     |

---

# 🧠 Machine Learning Models

The project implements classical ML classification algorithms.

---

# 🔹 K-Nearest Neighbors (KNN)

KNN classifies samples based on the **nearest neighbors in feature space**.

Idea:

```
New sample → find closest neighbors → majority class wins
```

Example implementation:

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
```

---

# 🔹 Support Vector Machine (SVM)

SVM finds an **optimal hyperplane that separates different classes**.

```python
from sklearn.svm import SVC

model = SVC()
model.fit(X_train, y_train)
```

SVM works well for **small datasets with clear boundaries**.

---

# 📊 Model Evaluation

The trained model is evaluated using the testing dataset.

Example:

```python
model.score(X_test, y_test)
```

The score represents **classification accuracy**.

---

# 🔮 Example Prediction

Example input:

```python
sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(sample)
```

Output:

```
Setosa
```

Meaning the flower belongs to the **Setosa species**.

---

# 🎯 Results

The models achieve **high accuracy** due to the well-separated nature of the iris dataset.

Typical performance:

| Model | Accuracy |
| ----- | -------- |
| KNN   | ~95–100% |
| SVM   | ~96–100% |

---

# 📚 Learning Outcomes

Through this project:

* Implemented a **multiclass classification model**
* Explored a **real-world ML dataset**
* Learned **data preprocessing and model training**
* Compared different machine learning algorithms
* Performed predictions using trained models

---

# 🔮 Future Improvements

Possible improvements:

* Hyperparameter tuning
* Cross-validation
* Visualizing decision boundaries
* Deploying the model as a web application
* Using additional classification algorithms

---

# 🤝 Contributing

Contributions are welcome.

Steps:

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Submit a pull request

---



If you want, I can also show you **how to organize them in your GitHub so your profile looks like a strong ML portfolio for internships and research labs.**
