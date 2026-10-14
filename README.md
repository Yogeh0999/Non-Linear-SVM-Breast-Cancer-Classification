# Non-Linear SVM - Breast Cancer Classification

A Machine Learning project that uses a Non-Linear Support Vector Machine (SVM) with the RBF kernel to classify breast cancer tumors as Benign or Malignant.

The project uses the Breast Cancer Wisconsin dataset available directly through Scikit-learn and provides a Flask-based web interface for predictions.

## Algorithm

Support Vector Machine (SVM) with Radial Basis Function (RBF) Kernel

```python
SVC(kernel="rbf", C=1.0, gamma="scale")
