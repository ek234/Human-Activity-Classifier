import sys
import pickle
import numpy as np
from time import time
import sklearn.metrics
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

def create_model (depth, isred):
    X_train, X_test, y_train, y_test = np.load('../processed_data/linear_interpolation.npz').values()
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=420)

    if isred:
        X_train = X_train[:,:10]
        X_val = X_val[:,:10]
        X_test = X_test[:,:10]

    # X_train, y_train = X_train[:100], y_train[:100]

    start_time = time()

    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    train_dur = time() - start_time
    # print(f"Time taken to run: {train_dur}")

    pickle.dump(dt, open(f'../models/dt_{depth}_{isred}.pkl', 'wb'))
    y_pred_val = dt.predict(X_val)
    acc_val = sklearn.metrics.accuracy_score(y_val, y_pred_val)
    # print(f"Accuracy wrt validation set for full data with C={c}: {acc_val}")

    y_pred_test = dt.predict(X_test)
    acc_test = sklearn.metrics.accuracy_score(y_test, y_pred_test)
    # print(f"Accuracy wrt test set for reduced data with C={c}: {acc_test}")
    confusion_matrix_test = sklearn.metrics.confusion_matrix(y_test, y_pred_test)
    class_rep = sklearn.metrics.classification_report(y_test, y_pred_test)

    y_pred_train = dt.predict(X_train)
    acc_train = sklearn.metrics.accuracy_score(y_train, y_pred_train)
    # print(f"Accuracy wrt test set for reduced data with C={c}: {acc_test}")
    
    return train_dur, acc_val, acc_test, acc_train, confusion_matrix_test, class_rep

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage error")
        exit(1)
    depth = int(sys.argv[1])
    use_reduced = sys.argv[2] == "red"
    print(f"starting dt exec->  depth: {depth}, using reduced set: {use_reduced}")

    train_time, acc_v, acc_t, acc_tr, confusion_matrix_test, class_rep = create_model(depth, use_reduced)
    with open(f"../models/dt_metrics_{sys.argv[2]}_{sys.argv[1]}.txt", 'w') as file:
        file.write(f"train time: {train_time}\n")
        file.write(f"accuracy v: {acc_v*100:.3f}%\n")
        file.write(f"accuracy te: {acc_t*100:.3f}%\n")
        file.write(f"accuracy tr: {acc_tr*100:.3f}%\n")
        file.write(f"confusion matrix:\n{confusion_matrix_test}\n")
        file.write(f"classification report:\n{class_rep}\n")