from datasets import EqualityDataset, PremackDataset, PremackDatasetIntermediate
from itertools import product
import numpy as np
import os
import pandas as pd
from sklearn.metrics import accuracy_score
import warnings
from tf_tree import TfTree



def run():
    n_trials = 20
    data = []
    alphas = [0.001, 0.01,0.1,1]
    lr = [0.01, 0.001, 0.0001]
    embed_dims = [2, 10, 25, 50, 100]

    grid = (embed_dims, alphas)
    rez = []

    grid = list(product(*grid))

    print(f"Running {len(grid)} experiments")

    for embed_dim, alpha in grid:

        print(f"embed_dim={embed_dim}, alpha={alpha}; trial ", end=" ")

        best = (0,0)
        print(best)
        for j in range(1,10):
            mod = TfTree(embed_dim=embed_dim, alpha=alpha, lr=lr, hidden_dim=embed_dim, name="test3")

            X_eq_train, y_eq_train = PremackDatasetIntermediate(
                        embed_dim=embed_dim,
                        n_pos=10000,
                        n_neg=10000).create()

            X_eq_test, y_eq_test = PremackDatasetIntermediate(
                        embed_dim=embed_dim).create()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mod.pretrain(X_eq_train, y_eq_train)

            eq_predict = mod.predict_intermediate(X_eq_test)

            temp = []
            for i in y_eq_test:
                temp.append(i[0])
            for i in y_eq_test:
                temp.append(i[1])
            y_eq_test = temp

            equality_acc = accuracy_score(y_eq_test, eq_predict)
            X_premack_test, y_premack_test = PremackDatasetIntermediate(
                    embed_dim=embed_dim).create()

            premack_predict = mod.predict(X_premack_test)
            premack_acc = accuracy_score([i[2] for i in y_premack_test], premack_predict)
            print("\n\n", equality_acc,premack_acc,"\n\n")
            if premack_acc > best[0]:
                best = (premack_acc, j)
            print(best)
        for trial in range(1, n_trials+1):

            print(trial, end=" ")

            mod = TfTree(embed_dim=embed_dim, alpha=alpha, lr=lr, hidden_dim=embed_dim, name=str(trial))


            X_eq_train, y_eq_train = PremackDatasetIntermediate(
                embed_dim=embed_dim,
                n_pos=10000*best[1],
                n_neg=10000*best[1]).create()

            X_eq_test, y_eq_test = PremackDatasetIntermediate(
                embed_dim=embed_dim).create()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mod.pretrain(X_eq_train, y_eq_train)

            eq_predict = mod.predict_intermediate(X_eq_test)

            temp = []
            for i in y_eq_test:
                temp.append(i[0])
            for i in y_eq_test:
                temp.append(i[1])
            y_eq_test = temp

            equality_acc = accuracy_score(y_eq_test, eq_predict)

            X_premack_test, y_premack_test = PremackDatasetIntermediate(
                embed_dim=embed_dim).create()

            premack_predict = mod.predict(X_premack_test)
            premack_acc = accuracy_score([i[2] for i in y_premack_test], premack_predict)
            print("\n\n", equality_acc,premack_acc,"\n\n")
            rez.append((equality_acc,premack_acc))

            d = {
                'trial': trial,
                'embed_dim': embed_dim,
                'hidden_dim': embed_dim,
                'alpha': alpha,
                'learning_rate': lr,
                'premack_training': 0,
                'equality_accuracy': equality_acc,
                'premack_accuracy': premack_acc}

            data.append(d)

            for n_premack_train in range(20, 1000, 20):

                X_premack_train, y_premack_train = PremackDatasetIntermediate(
                    n_pos=20,
                    n_neg=20,
                    embed_dim=embed_dim).create()

                mod.fit(X_premack_train, y_premack_train)

                premack_predict = mod.predict(X_premack_test)

                premack_acc = accuracy_score([i[2] for i in y_premack_test], premack_predict)

                d = {
                    'trial': trial,
                    'embed_dim': embed_dim,
                    'hidden_dim': embed_dim,
                    'alpha': alpha,
                    'learning_rate': lr,
                    'premack_training': n_premack_train,
                    'equality_accuracy': equality_acc,
                    'premack_accuracy': premack_acc}

                data.append(d)

        print(rez, "\n")

    return pd.DataFrame(data)


if __name__ == '__main__':

    df = run()
    df.to_csv(os.path.join("results", "inputasoutput-results.csv"), index=None)
