# SCT_DS_TASK03

# Bank Marketing Prediction using Decision Tree

It uses a **Decision Tree Classifier** to predict whether a customer will subscribe to a term deposit, based on demographic and behavioral data from the popular **Bank Marketing dataset** (UCI Machine Learning Repository).

---

## 📊 **Project Overview**
- **Goal:** Predict customer subscription to a term deposit (`y` column: yes/no)
- **Dataset:** `bank-full.csv`  
- **Main steps:**
  - Data exploration
  - Preprocessing (encoding categorical variables)
  - Train-test split
  - Build & train Decision Tree Classifier
  - Evaluate model performance
  - Visualize feature importance & decision tree

---

## ⚙️ **Technologies & Libraries**
- Python 🐍
- pandas
- scikit-learn
- matplotlib
- seaborn

---

## 🔍 **Project Workflow**

### ✅ Step 1: Import libraries
Import necessary libraries for data handling, model building, evaluation, and visualization.

### 📂 Step 2: Load dataset
Read the `bank-full.csv` dataset, explore the first few rows, and check data types and target distribution.

### 🛠 Step 3: Preprocessing
Encode categorical variables into numeric values using `LabelEncoder` so they can be used by scikit-learn models.

### ✂ Step 4: Split data
Split into training and test sets (80%-20%).

### 🌳 Step 5: Build & train Decision Tree
Train a `DecisionTreeClassifier` with `max_depth=5` to avoid overfitting.

### 📈 Step 6: Evaluate model
- Calculate accuracy score
- Show confusion matrix & classification report

### 📊 Step 7: Visualization
- Plot feature importance (top features affecting predictions)
- Visualize the decision tree to understand model splits and logic

---

## ✨ **Results**
- Previous marketing contact outcome, duration of last contact, and customer age significantly influence purchase decisions.
- Decision tree visualization clearly shows how the model prioritizes these features.

---

## 📌 **How to run**
1. Clone the repository
2. Install required libraries:
   ```bash
   pip install pandas scikit-learn matplotlib seaborn

## Linkedin (Task link)
(https://www.linkedin.com/feed/update/urn:li:activity:7356540234517397504/)
   
