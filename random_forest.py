import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = np.load('./processed_data/linear_interpolation.npz').values()

X_train_reduced = X_train[:,:10]
X_test_reduced = X_test[:,:10]

reduced_rf = RandomForestClassifier()
reduced_rf.fit(X_train_reduced, y_train)
y_pred = reduced_rf.predict(X_test_reduced)
print("For reduced data:")
print(classification_report(y_test, y_pred))

print("==============================================")

rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)
print("For full data:")
print(classification_report(y_test, y_pred))