from itertools import product
import numpy as np
import pandas as pd
import random
import string
from torch_fuzzy_lm import FuzzyPatternLM, START_SYMBOL, END_SYMBOL


class Dataset:
    def __init__(self, train_vocab, test_vocab):
        self.train_vocab = train_vocab
        self.test_vocab = test_vocab
        self.train = self.generate(self.train_vocab)
        self.test = self.generate(self.test_vocab)

    def generate(self, vocab):
        dataset = []
        for ex in self.example_generator(vocab):
            dataset.append([START_SYMBOL] + ex + [END_SYMBOL])
        return dataset


class DatasetABA(Dataset):

    @staticmethod
    def example_generator(vocab):
        for c1 in vocab:
            for c2 in vocab:
                if c1 != c2:
                    yield [c2, c1, c2]

    @staticmethod
    def is_error(p, test_len):
        return len(p) != test_len or p[1] != p[-2] or p[1] == p[2]


class FuzzyPatternLMExperiment:
    def __init__(self,
            dataset_class,
            pretrain=False,
            n_trials=10,
            embed_dim=50,
            hidden_dim=50,
            num_layers=1,
            dropout=0,
            max_iter=10,
            eta=0.05,
            train_vocab_size=100,
            test_vocab=list(string.ascii_letters)):
        self.dataset_class = dataset_class
        self.pretrain = pretrain
        self.n_trials = n_trials
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.max_iter = max_iter
        self.eta = eta
        self.num_layers = num_layers
        self.dropout = dropout
        self.train_vocab_size = train_vocab_size
        self.train_vocab = list(map(str, range(self.train_vocab_size)))
        self.test_vocab = test_vocab
        self.dataset = self.dataset_class(self.train_vocab, self.test_vocab)
        self.full_vocab = self.train_vocab + self.test_vocab
        self.full_vocab += [START_SYMBOL, END_SYMBOL]

    def pretrain_model(self, model):
        X_train, y_train = self.generate_equality_dataset(self.train_vocab)
        model.fit(X_train, y_train)
        return model

    @staticmethod
    def generate_equality_dataset(vocab):
        X_neg = [[START_SYMBOL, w1, w2, END_SYMBOL] for w1, w2 in product(vocab, repeat=2) if w1 != w2]
        X_pos = [[START_SYMBOL, w, w, END_SYMBOL] for w in vocab]
        x = int(len(X_neg) / len(X_pos))
        X_pos *= x
        y_pos = [1] * len(X_pos)
        y_neg = [0] * len(X_neg)
        X = X_pos + X_neg
        y = y_pos + y_neg
        return X, y

    def run(self):
        data = []
        for trial in range(1, self.n_trials+1):

            model = FuzzyPatternLM(
                vocab=self.full_vocab,
                embed_dim=self.embed_dim,
                hidden_dim=self.hidden_dim,
                max_iter=self.max_iter,
                num_layers=self.num_layers,
                dropout=self.dropout,
                warm_start=True,
                eta=self.eta)

            if self.pretrain:
                model = self.pretrain_model(model)

            model.fit(self.dataset.train, eval_func=self.evaluate)
            preds = model.results.copy()
            for p in preds:
                p.update({'trial': trial})
            data += preds
        return data

    def evaluate(self, model, verbose=False):
        all_preds = set()
        test_len = len(self.dataset.test[0])
        prompts = sorted({tuple(ex[: 2]) for ex in self.dataset.test})
        for prompt in prompts:
            pred = tuple(model.predict_one(prompt))
            all_preds.add(pred)
        data = {
            'correct': [],
            'incorrect': []}
        for p in all_preds:
            if self.dataset.is_error(p, test_len):
                data['incorrect'].append(p)
            else:
                data['correct'].append(p)
        data['n_correct'] = len(data['correct'])
        data['n_incorrect'] = len(data['incorrect'])
        data['accuracy'] = data['n_correct'] / len(all_preds)
        if verbose:
            print(f"{data['n_incorrect']} errors for {len(all_preds)} "
                  f"test examples; accuracy is {data['accuracy']}")
        return data
