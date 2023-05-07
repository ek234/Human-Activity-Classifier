import sys
import pickle
import numpy as np
from time import time
import sklearn.metrics
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def create_model (c, isred, kern):
    X_train, X_test, y_train, y_test = np.load('../processed_data/linear_interpolation.npz').values()
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=420)

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

    if len(y_train) > 17500:
        print("using reduced data", flush=True)
        X_train, y_train = X_train[:175000], y_train[:175000] # reducing size of data as per paper
        X_val, y_val = X_val[:3000], y_val[:3000]
        X_test, y_test = X_test[:7000], y_test[:7000]

    start_time = time()

    clf = make_pipeline(StandardScaler(), SVC(C=c, kernel=kern, gamma='auto'))
    clf.fit(X_train, y_train)
    train_dur = time() - start_time
    print(f"Time taken to run for C={c}, redu={isred}, kern={kern}: {train_dur}", flush=True)

    pickle.dump(clf, open(f'../models/svm_{kern}_{c}_{isred}.pkl', 'wb'))
    y_pred_val = clf.predict(X_val)
    acc_val = sklearn.metrics.accuracy_score(y_val, y_pred_val)
    print(f"Accuracy wrt validation set for full data with C={c}: {acc_val}", flush=True)

    y_pred_test = clf.predict(X_test)
    acc_test = sklearn.metrics.accuracy_score(y_test, y_pred_test)
    print(f"Accuracy wrt test set for reduced data with C={c}: {acc_test}", flush=True)
    confusion_matrix_test = sklearn.metrics.confusion_matrix(y_test, y_pred_test)
    class_rep = sklearn.metrics.classification_report(y_test, y_pred_test)

    y_pred_train = clf.predict(X_train)
    acc_train = sklearn.metrics.accuracy_score(y_train, y_pred_train)
    print(f"Accuracy wrt test set for reduced data with C={c}: {acc_test}", flush=True)

    return train_dur, acc_val, acc_test, acc_train, confusion_matrix_test, class_rep

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage error")
        exit(1)
    C = float(sys.argv[1])
    use_reduced = (sys.argv[2] == "red")
    kernel = sys.argv[3]
    print(f"starting svm exec->  C: {C}, using reduced set: {use_reduced}, kernel: {kernel}", flush=True)

    train_time, acc_v, acc_t, acc_tr, confusion_matrix_test, class_rep = create_model(C, use_reduced, kernel)
    with open(f"../models/svm_metrics_{kernel}_{C}_{use_reduced}.txt", 'w') as file:
        file.write(f"train time: {train_time}\n")
        file.write(f"accuracy v: {acc_v*100:.3f}%\n")
        file.write(f"accuracy te: {acc_t*100:.3f}%\n")
        file.write(f"accuracy tr: {acc_tr*100:.3f}%\n")
        file.write(f"confusion matrix:\n{confusion_matrix_test}\n")
        file.write(f"classification report:\n{class_rep}\n")
