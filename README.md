# Machine Learning Projects  

This repository contains three machine learning projects  
Each project demonstrates different ML techniques: supervised learning, unsupervised learning, and neural networks applied to real-world datasets.  

---
## Project Info  

Duration: Nov 2024 – Jan 2025  
Type: Machine Learning Projects  
Language/Tools: Python, Scikit-learn, TensorFlow/Keras, PyCaret, Pandas, NumPy, Matplotlib, Seaborn  
Techniques: Supervised Learning, Unsupervised Learning, Neural Networks 

---
## 1. Credit Card Approval Prediction (Supervised Learning)  

**Dataset:** Credit Card Approval Prediction(Kaggle) consists of:
- **Table 1:** Client information (marital status, gender, family size, annual income, etc.)  
- **Table 2:** Loan payment history for each client  

**Objective:** Predict whether a client should be classified as **Good** (likely to repay) or **Bad** (risk of default) based on loan payment history and client information. 

**Workflow:**  
- Generated **Good/Bad labels** from loan payment records (Table 2) and merged them with client data (Table 1).  
- Applied data preprocessing: missing value imputation, categorical encoding, normalization, and outlier handling.  
- Trained supervised ML models: **Logistic Regression, Random Forest, and SVM**.  
- Performed **Grid Search** + **Cross Validation** for hyperparameter tuning.  
- Evaluated models using **Confusion Matrix** and **Classification Report**.    

---

## 2. Credit Card Customer Clustering (Unsupervised Learning)  
**Dataset:** Credit card usage behavior of ~9,000 customers over 6 months.  

**Objective:** Group customers with similar spending/usage behavior.  

**Workflow:**  
- Preprocessed data: missing values, normalization, outlier removal.  
- Reduced dimensionality using **PCA**.  
- Implemented clustering algorithms:  
  - KMeans (optimal k via Elbow Method)  
  - Hierarchical Clustering (dendrograms)  
  - Spectral Clustering  
- Evaluated clustering with **Dunn Index**.  
- Visualized clusters in PCA-reduced 2D space.  

---

## 3. Breast Cancer Detection using Microwave Sensor Data (Deep Learning)  
**Dataset:** Microwave sensor measurements (S11–S44 features: magnitude + phase), labeled as *20mm tumor* or *No tumor*.  

**Objective:** Build a neural network model to classify tumor presence.  

**Workflow:**  
- Preprocessing: normalization, correlation analysis, outlier handling.  
- Implemented a **Neural Network**:  
  - Optimizer: SGD (lr=0.001)  
  - Loss: Binary Cross Entropy  
  - Metrics: Accuracy, F1-score  
- Compared results with multiple ML models using **PyCaret**.  
- Interpreted results and highlighted trade-offs between NN and traditional ML.  

---
