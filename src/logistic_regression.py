import sys
import pickle
import numpy as np
from time import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def create_model (c, isred):
    X_train, X_test, y_train, y_test = np.load('../processed_data/linear_interpolation.npz').values()
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=420)

    if isred:
        X_train = X_train[:,:10]
        X_val = X_val[:,:10]
        X_test = X_test[:,:10]

    # X_train, y_train = X_train[:100], y_train[:100]

    start_time = time()

    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    lr = LogisticRegression(solver='sag', max_iter=5000, multi_class='multinomial', C=c)
    lr.fit(X_norm, y_train)
    train_dur = time() - start_time
    # print(f"Time taken to run: {train_dur}")

    pickle.dump(lr, open(f'../models/lr_{c}_{isred}.pkl', 'wb'))
    y_pred_val = lr.predict(X_val)
    acc_val = accuracy_score(y_val, y_pred_val)
    # print(f"Accuracy wrt validation set for full data with C={c}: {acc_val}")

    y_pred_test = lr.predict(X_test)
    acc_test = accuracy_score(y_test, y_pred_test)
    # print(f"Accuracy wrt test set for reduced data with C={c}: {acc_test}")
    confusion_matrix_test = confusion_matrix(y_test, y_pred_test)
    class_rep = classification_report(y_test, y_pred_test)

    y_pred_train = lr.predict(X_norm)
    acc_train = accuracy_score(y_train, y_pred_train)
    # print(f"Accuracy wrt test set for reduced data with C={c}: {acc_test}")
    
    return train_dur, acc_val, acc_test, acc_train, confusion_matrix_test, class_rep


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage error")
        exit(1)
    C = float(sys.argv[1])
    use_reduced = (sys.argv[2] == "red")
    print(f"starting LR exec->  C: {C}, using reduced set: {use_reduced}")

    train_time, acc_v, acc_t, acc_tr, confusion_matrix_test, class_rep = create_model(C, use_reduced)
    with open(f"../models/lr_metrics_{sys.argv[2]}_{sys.argv[1]}.txt", 'w') as file:
        file.write(f"train time: {train_time}\n")
        file.write(f"accuracy v: {acc_v*100:.3f}%\n")
        file.write(f"accuracy te: {acc_t*100:.3f}%\n")
        file.write(f"accuracy tr: {acc_tr*100:.3f}%\n")
        file.write(f"confusion matrix:\n{confusion_matrix_test}\n")
        file.write(f"classification report:\n{class_rep}\n")
