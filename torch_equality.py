from datasets import EqualityDataset

from itertools import product
import numpy as np
import os
import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from equality_experiment import EqualityExperiment

class TorchEqualityDataset(EqualityDataset, Dataset):
    def __init__(self, embed_dim=10, n_pos=500, n_neg=500, flatten=True):
        self.embed_dim = embed_dim
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.flatten = flatten

        self.all_X, self.all_y = self.create()
        self.X, self.y = self.all_X, self.all_y

    def limit(self, start, end):
        assert start < end
        self.X = self.all_X[start:end]
        self.y = self.all_y[start:end]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class TorchEqualityModule(torch.nn.Module):
    def __init__(self,
                 input_size=20,
                 hidden_layer_size=100,
                 activation="relu"):
        super(TorchEqualityModule, self).__init__()
        self.linear = torch.nn.Linear(input_size,hidden_layer_size)
        if activation == "relu":
            self.activation = torch.nn.ReLU()
        else:
            raise NotImplementedError("Activation method not implemented")
        self.output = torch.nn.Linear(hidden_layer_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        linear_out = self.linear(x)
        hidden_vec = self.activation(linear_out)
        logits = self.output(hidden_vec)
        return self.sigmoid(logits)

class TorchEqualityModel:
    def __init__(self,
                 max_epochs=100,
                 input_size=20,
                 batch_size=1000,
                 hidden_layer_size=100,
                 activation='relu',
                 alpha=0.0001,
                 optimizer='adam',
                 lr=0.01,
                 beta_1=0.9,
                 beta_2=0.999,
                 early_stop_threshold=1e-5,
                 gpu=False):
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stop_threshold = early_stop_threshold
        self.loss = torch.nn.BCELoss()
        self.module = TorchEqualityModule(input_size=input_size,
                                          hidden_layer_size=hidden_layer_size,
                                          activation=activation)
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.module.parameters(),
                                              lr=lr,
                                              betas=(beta_1, beta_2),
                                              weight_decay=alpha)
        else:
            raise NotImplementedError("Optimizer option not implemented")

        self.gpu = gpu
        self.device = torch.device("cuda") if gpu else torch.device("cpu")
        self.module = self.module.to(self.device)

    def fit(self, dataset):
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        num_iters = 0
        prev_loss = float("inf")
        early_stop = False
        for epoch in range(self.max_epochs):
            for X, y in dataloader:
                self.module.zero_grad()
                X = X.float().to(self.device)
                y = y.float().to(self.device)
                y_pred = self.module(X).reshape(-1)
                loss = self.loss(y_pred, y)
                loss.backward()
                self.optimizer.step()
                num_iters += 1
                if prev_loss - loss <= self.early_stop_threshold:
                    early_stop = True
                    break
            if early_stop:
                break

    def predict(self, dataset):
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        all_preds = []
        with torch.no_grad():
            for X, _ in dataloader:
                X = X.float().to(self.device)
                y_pred = self.module(X).reshape(-1)
                if self.gpu:
                    y_pred = y_pred.to(torch.device("cpu"))
                all_preds += list(y_pred)
        return [1 if y >= 0.5 else 0 for y in all_preds]

    def predict_one(self, X):
        with torch.no_grad():
            X = X.float().to(self.device)
            y_pred = self.module(X).reshape(-1)
            if self.gpu:
                y_pred = y_pred.to(torch.device("cpu"))
        return [1 if y >= 0.5 else 0 for y in y_pred]


class TorchEqualityExperiment(EqualityExperiment):
    def get_model(self, hidden_dim, alpha, lr, embed_dim):
        return TorchEqualityModel(hidden_layer_size=hidden_dim,
                                  alpha=alpha,
                                  lr=lr,
                                  input_size=embed_dim*2)

    def run_once(self, data, embed_dim, hidden_dim, alpha, lr):
        print(f"Running trials for embed_dim={embed_dim} hidden_dim={hidden_dim} "
              f"alpha={alpha} lr={lr} ...", end=" ")

        start = time.time()

        scores = []

        for trial in range(1, self.n_trials+1):

            mod = self.get_model(hidden_dim, alpha, lr, embed_dim)

            train_dataset, test_dataset = self.get_new_train_and_test_sets(embed_dim)

            # Record the result with no training if the model allows it:
            preds = mod.predict(test_dataset)
            acc = accuracy_score(test_dataset.y, preds)
            scores.append(acc)
            d = {
                'trial': trial,
                'train_size': 0,
                'embed_dim': embed_dim,
                'hidden_dim': hidden_dim,
                'alpha': alpha,
                'learning_rate': lr,
                'accuracy': acc,
                'batch_pos': 0,
                'batch_neg': 0}
            if hasattr(self, "pretraining_metadata"):
                d.update(self.pretraining_metadata)
            data.append(d)

            for train_size in self.train_sizes:
                assert train_size >= 40
                train_dataset.limit(0, train_size)
                batch_pos = sum([1 for label in train_dataset.y if label == 1])

                mod.fit(train_dataset)

                # Predictions:
                preds = mod.predict(test_dataset)
                acc = accuracy_score(test_dataset.y, preds)
                scores.append(acc)
                d = {
                    'trial': trial,
                    'train_size': train_size,
                    'embed_dim': embed_dim,
                    'hidden_dim': hidden_dim,
                    'alpha': alpha,
                    'learning_rate': lr,
                    'accuracy': acc,
                    'batch_pos': batch_pos,
                    'batch_neg': len(train_dataset) - batch_pos}
                if hasattr(self, "pretraining_metadata"):
                    d.update(self.pretraining_metadata)
                data.append(d)

        elapsed_time = round(time.time() - start, 0)

        print(f"mean: {round(np.mean(scores), 2)}; max: {max(scores)}; took {elapsed_time} secs")

    def run(self):
        data = []

        print(f"Grid size: {len(self.grid)} * {self.n_trials}; "
              f"{len(self.grid)*self.n_trials} experiments")

        for embed_dim, hidden_dim, alpha, lr in self.grid:
            self.run_once(data, embed_dim, hidden_dim, alpha, lr)

        self.data_df = pd.DataFrame(data)
        return self.data_df

    def get_new_train_and_test_sets(self, embed_dim):
        train_dataset = self.dataset_class(
            embed_dim=embed_dim,
            n_pos=self.class_size,
            n_neg=self.class_size)

        test_dataset = self.dataset_class(
            embed_dim=embed_dim,
            n_pos=self.test_set_class_size,
            n_neg=self.test_set_class_size)

        train_dataset.test_disjoint(test_dataset)

        return train_dataset, test_dataset

    def get_minimal_train_set(self, train_size, embed_dim, other_dataset):
        class_size = int(train_size / 2)
        train_dataset = self.dataset_class(
            embed_dim=embed_dim,
            n_pos=class_size,
            n_neg=class_size)

        train_dataset.test_disjoint(other_dataset)

        return train_dataset
