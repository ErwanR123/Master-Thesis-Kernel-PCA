# Kernel PCA – Theory & Applications (Master Thesis)

This repository contains the work related to my Master’s thesis on **Kernel Principal Component Analysis (KPCA)**, a non-linear extension of PCA based on kernel methods.

📄 The full thesis is available here:  
**`Mémoire M1 Kernel PCA - Kevin Wardakhan - Ibrahim Youssouf Abdelatif - Erwan Ouabdesselam.pdf`**  
*(Note: the document is written in French)*

---

## Overview

The goal of this work is to study both the **theoretical foundations** and the **practical impact** of Kernel PCA.

The manuscript covers:
- the mathematical framework behind kernel methods (Hilbert spaces, kernel trick)
- the extension of PCA to non-linear settings
- several real-world applications where linear methods fail

Kernel PCA makes it possible to analyze complex structures by implicitly mapping data into high-dimensional feature spaces

---

## Structure of the Thesis

- **PCA fundamentals** (definitions, optimization, spectral decomposition)
- **Kernel PCA theory** (Aronszajn theorem, Gram matrix, algorithm)
- **Experiments on real datasets**:
  - Sentiment analysis (IMDb)
  - Anomaly detection (MNIST)
  - ECG signal denoising

---

## Experiments

### 1. Sentiment Analysis (IMDb)

- Text preprocessing + Bag-of-Words (CountVectorizer)
- Dimensionality reduction with Kernel PCA (cosine kernel)
- Models: Logistic Regression, SVM, KNN
- Comparison:
  - without reduction
  - PCA
  - Kernel PCA

Goal: evaluate whether KPCA improves classification performance.

---

### 2. Anomaly Detection (MNIST)

- Training on a single class (digit "0")
- Projection into feature space via Kernel PCA
- Detection based on **reconstruction error**

Kernel PCA allows measuring how far a sample is from the learned data manifold, which is useful for novelty detection

---

### 3. ECG Signal Denoising

- Noisy ECG signals reconstructed using Kernel PCA
- Projection + reconstruction in feature space
- Comparison with linear PCA

Goal: show how KPCA preserves non-linear structure while removing noise.

---

## Authors

- Erwan Ouabdesselam  
- Ibrahim Youssouf Abdelatif  
- Kevin Wardakhan  

Supervisors:
- Denis Pasquignon  
- Patrice Bertrand  

Université Paris Dauphine – PSL
