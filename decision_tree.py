import sys
import pickle
import numpy as np
from time import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

X_train, X_test, y_train, y_test = np.load('./processed_data/linear_interpolation.npz').values()
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=420)

X_train_reduced = X_train[:,:10]
X_val_reduced = X_val[:,:10]
X_test_reduced = X_test[:,:10]

def create_model (X_train, X_val, X_test, y_train, y_val, y_test, depth, isred):
    print("hiii")
    start_time = time()
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    train_dur = time() - start_time
    # print(f"Time taken to run: {train_dur}")
    pickle.dump(dt, open(f'./models/dt_{depth}_{isred}.pkl', 'wb'))
    y_pred_val = dt.predict(X_val)
    acc_val = accuracy_score(y_val, y_pred_val)
    # print(f"Accuracy wrt validation set for full data with C={c}: {acc_val}")
    y_pred_test = dt.predict(X_test)
    acc_test = accuracy_score(y_test, y_pred_test)
    # print(f"Accuracy wrt test set for reduced data with C={c}: {acc_test}")
    return train_dur, acc_val, acc_test

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage error")
        exit(1)
    depth = float(sys.argv[1])
    use_reduced = sys.argv[2] == "red"
    print(f"Depth: {depth}, using reduced set: {use_reduced}")

    if use_reduced:
        X_train, X_val, X_test = X_train_reduced, X_val_reduced, X_test_reduced

    print("heh")

    train_time, acc_v, acc_t = create_model(X_train, X_val, X_test, y_train, y_val, y_test, depth, use_reduced)
    with open(f"./models/dt_metrics_{sys.argv[2]}_{sys.argv[1]}.txt", 'w') as file:
        file.write(f"train time: {train_time}\n")
        file.write(f"accuracy v: {acc_v*100:.3f}%\n")
        file.write(f"accuracy t: {acc_t*100:.3f}%\n")
