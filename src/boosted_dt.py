import sys
import pickle
import numpy as np
from time import time
import sklearn.metrics
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

def create_model (depth, num_trees, isred):
    X_train, X_test, y_train, y_test = np.load('../processed_data/linear_interpolation.npz').values()
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=420)

    # remove indices where label is 0 (as per the paper)
    train_idxs = np.where(y_train != 0)[0]
    val_idxs = np.where(y_val != 0)[0]
    test_idxs = np.where(y_test != 0)[0]

    X_train, y_train = X_train[train_idxs], y_train[train_idxs]
    X_val, y_val = X_val[val_idxs], y_val[val_idxs]
    X_test, y_test = X_test[test_idxs], y_test[test_idxs]

    if isred:
        X_train = X_train[:,:10]
        X_val = X_val[:,:10]
        X_test = X_test[:,:10]

    # X_train, y_train = X_train[:100], y_train[:100]

    start_time = time()

    dt = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=depth), n_estimators=num_trees,
    )
    dt.fit(X_train, y_train)
    train_dur = time() - start_time
    # print(f"Time taken to run: {train_dur}")

    pickle.dump(dt, open(f'../models/bdt_{num_trees}_{depth}_{isred}.pkl', 'wb'))
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
    num_trees = int(sys.argv[3])
    print(f"starting bdt exec->  depth: {depth}, num of tree: {num_trees}, using reduced set: {use_reduced}")

    train_time, acc_v, acc_t, acc_tr, confusion_matrix_test, class_rep = create_model(depth, num_trees, use_reduced)
    with open(f"../models/bdt_metrics_{num_trees}_{use_reduced}_{depth}.txt", 'w') as file:
        file.write(f"train time: {train_time}\n")
        file.write(f"accuracy v: {acc_v*100:.3f}%\n")
        file.write(f"accuracy te: {acc_t*100:.3f}%\n")
        file.write(f"accuracy tr: {acc_tr*100:.3f}%\n")
        file.write(f"confusion matrix:\n{confusion_matrix_test}\n")
        file.write(f"classification report:\n{class_rep}\n")
