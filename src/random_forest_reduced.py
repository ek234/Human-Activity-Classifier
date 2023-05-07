import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from time import time

X_train, X_test, y_train, y_test = np.load('../processed_data/linear_interpolation.npz').values()

X_train_reduced = X_train[:,:10]
X_test_reduced = X_test[:,:10]

start_time = time()
reduced_rf = RandomForestClassifier()
reduced_rf.fit(X_train_reduced, y_train)
y_pred = reduced_rf.predict(X_test_reduced)
train_time = time() - start_time

acc_t = accuracy_score(y_test, y_pred)
acc_tr = accuracy_score(y_train, reduced_rf.predict(X_train_reduced))

class_rep = classification_report(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)

with open(f"../models/rf_metrics_red.txt", 'w') as file:
    file.write(f"train time: {train_time}\n")
    file.write(f"accuracy te: {acc_t*100:.3f}%\n")
    file.write(f"accuracy tr: {acc_tr*100:.3f}%\n")
    file.write(f"confusion matrix:\n{conf_mat}\n")
    file.write(f"classification report:\n{class_rep}\n")