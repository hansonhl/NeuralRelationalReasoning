from itertools import product
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
import time
import warnings
from datasets import EqualityDataset, PremackDataset


class EqualityExperiment:

    def __init__(self,
            dataset_class=EqualityDataset,
            n_hidden=1,
            featurizer=None,
            model=None,
            n_trials=10,
            train_sizes=[4, 40] + list(range(104, 100001, 5000)),
            embed_dims=[2, 10, 25, 50, 100],
            hidden_dims=[2, 10, 25, 50, 100],
            alphas=[0.00001, 0.0001, 0.001],
            learning_rates=[0.0001, 0.001, 0.01],
            test_set_class_size=250,
            pretrain_tree = False):

        self.dataset_class = dataset_class
        self.n_hidden = n_hidden
        self.featurizer = featurizer
        self.model = model
        self.n_trials = n_trials
        self.train_sizes = train_sizes
        self.class_size = int(max(self.train_sizes) / 2)
        self.embed_dims = embed_dims
        self.hidden_dims = hidden_dims
        self.alphas = alphas
        self.learning_rates = learning_rates
        self.pretrain_tree = pretrain_tree
        grid = (self.embed_dims, self.hidden_dims, self.alphas, self.learning_rates)
        self.grid = list(product(*grid))
        self.test_set_class_size = test_set_class_size


    def run(self):
        data = []

        print(f"Grid size: {len(self.grid)} * {self.n_trials}; "
              f"{len(self.grid)*self.n_trials} experiments")

        for embed_dim, hidden_dim, alpha, lr in self.grid:

            print(f"Running trials for embed_dim={embed_dim} hidden_dim={hidden_dim} "
                  f"alpha={alpha} learning_rate={lr} ...", end=" ")

            start = time.time()

            scores = []
            for trial in range(1, self.n_trials+1):

                mod = self.get_model(hidden_dim, alpha, lr,embed_dim)
                if self.pretrain_tree:
                    mod.set_lr_and_l2(0.001, 0.0001)
                    temp = self.class_size
                    self.class_size = 30000
                    X_train, X_test, y_train, y_test, train_dataset = \
                        self.get_new_train_and_test_sets(embed_dim, hidden_dim)
                    mod.pretrain(X_train, y_train)
                    self.class_size = temp
                    temp = []
                    for y in y_test:
                        temp.append(y[0])
                    for y in y_test:
                        temp.append(y[1])
                    y_test=temp
                    preds = mod.predict_intermediate(X_test)
                    acc = accuracy_score(y_test, preds)
                    mod.set_lr_and_l2(lr, alpha)
                X_train, X_test, y_train, y_test, train_dataset = \
                  self.get_new_train_and_test_sets(embed_dim, hidden_dim)

                for train_size in self.train_sizes:

                    train_size_start = 0

                    if train_size < 40:
                        X_batch, y_batch = self.get_minimal_train_set(
                            train_size, embed_dim, hidden_dim, train_dataset)
                        batch_pos = sum([1 for label in y_batch if label == 1])
                    else:
                        X_batch = X_train[train_size_start: train_size]
                        y_batch = y_train[train_size_start: train_size]
                        batch_pos = sum([1 for label in y_train[: train_size] if label == 1])

                    train_size_start = train_size

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        mod.fit(X_batch, y_batch)

                    # Predictions:
                    preds = mod.predict(X_test)

                    # Record data:
                    if isinstance(y_test[0], list):
                        temp = []
                        for y in y_test:
                            temp.append(y[-1])
                        y_test=temp
                    acc = accuracy_score(y_test, preds)
                    scores.append(acc)
                    d = {
                        'trial': trial,
                        'train_size': train_size,
                        'embed_dim': embed_dim,
                        'hidden_dim': hidden_dim,
                        'alpha': alpha,
                        'learning_rate': lr,
                        'accuracy': acc,
                        'batch_pos': batch_pos}
                    data.append(d)

            elapsed_time = round(time.time() - start, 0)

            print(f"mean: {round(np.mean(scores), 2)}; max: {max(scores)}; took {elapsed_time} secs")

        return pd.DataFrame(data)


    def get_model(self, hidden_dim, alpha, lr,embed_dim):
        if self.model is None:
            return MLPClassifier(
                max_iter=1,
                hidden_layer_sizes=tuple([hidden_dim] * self.n_hidden),
                activation='relu',
                alpha=alpha,
                solver='adam',
                learning_rate_init=lr,
                beta_1=0.9,
                beta_2=0.999,
                warm_start=True)
        else:
            return self.model(
                hidden_dim=hidden_dim,
                alpha=alpha,
                lr=lr,
                embed_dim=embed_dim)


    def get_new_train_and_test_sets(self, embed_dim, hidden_dim):
        train_dataset = self.dataset_class(
            embed_dim=embed_dim,
            n_pos=self.class_size,
            n_neg=self.class_size)
        X_train, y_train = train_dataset.create()

        test_dataset = self.dataset_class(
            embed_dim=embed_dim,
            n_pos=self.test_set_class_size,
            n_neg=self.test_set_class_size)
        X_test, y_test = test_dataset.create()

        train_dataset.test_disjoint(test_dataset)

        if self.featurizer is not None:
            X_test = self.featurizer(X_test, embed_dim, hidden_dim)
            X_train = self.featurizer(X_train, embed_dim, hidden_dim)

        return X_train, X_test, y_train, y_test, train_dataset


    def get_minimal_train_set(self, train_size, embed_dim, hidden_dim, test_dataset):
        train_dataset = self.dataset_class(
            embed_dim=embed_dim,
            n_pos=train_size,
            n_neg=train_size)
        X_batch, y_batch = train_dataset.create()

        train_dataset.test_disjoint(test_dataset)

        if self.featurizer is not None:
            X_batch = self.featurizer(X_batch, embed_dim, hidden_dim)

        return X_batch, y_batch
