# Classification_Implementations

<ins>Iris Classification Models</ins>

This repository contains implementations of several machine learning models on the Iris dataset. The primary focus is on Decision Tree classifiers, which rely on finding threshold values for comparing attributes, and Logistic Regression classifiers, which identify linear separations between pairs of attributes. Additionally, a simple One Rule (1R) classifier is implemented to demonstrate a straightforward classification approach.

<ins>Decision Tree Classifier</ins>

The `DecisionTreeClassifier` from scikit-learn is used to build a model that predicts the species of Iris flowers by learning decision rules inferred from the data features.

```python
# Load the dataset, split into training and test sets, train the classifier

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
```

<ins>Model Evaluation</ins>

The accuracy, confusion matrix, and precision-recall-F1 scores are calculated to evaluate the model's performance.

```python
# Calculate and print the model's accuracy and error rates
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

<ins>Logistic Regression</ins>

A LogisticRegression classifier is also trained to delineate decision boundaries visually for each class using two attributes for simplicity.

```python
# Train the logistic regression model and plot the decision boundaries

from sklearn.linear_model import LogisticRegression
```

<ins>1R Classifier</ins>

A minimalistic 1R classifier is created to establish basic classification rules based on the most common class for combinations of sepal length and sepal width.

```python
# Generate classification rules and apply them to make predictions
```

<ins>Visualisation</ins>

The decision trees and logistic regression decision boundaries are visualized using Graphviz and Matplotlib, respectively.

```python
# Render and save the decision tree as a PDF file

import graphviz
from sklearn.tree import export_graphviz
```

<ins>Quick Start</ins>

```python
1. Clone the repository.
2. Ensure you have the required packages installed: scikit-learn, pandas, numpy, seaborn, matplotlib, and graphviz.
3. Run the Jupyter notebooks to train models and visualize the decision boundaries.
```

<ins>Contributing</ins>

Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change.

License Distributed under the MIT License. See LICENSE for more information.

Contact Your Name - @oscarmoxon - oscar@oscarmoxon.com

Project Link: https://github.com/mrmoxon/Classification_Implementations 