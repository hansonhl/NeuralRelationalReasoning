import numpy as np
import sys
import torch
import torch.nn as nn
from utils import progress_bar


class RepEmbeddingModel(nn.Module):
    def __init__(self,
            vocab_size,
            embed_dim,
            hidden_dim,
            embedding=None,
            freeze_embedding=False):
        super().__init__()
        if embedding is None:
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.embedding.weight.requires_grad = not freeze_embedding
        else:
            self.embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(embedding), freeze=freeze_embedding)
        self.W = nn.Linear(embed_dim, hidden_dim)
        self.f = nn.ReLU()

    def forward(self, X):
        embs = self.embedding(X)
        return self.f(self.W(embs))


class RepTaskModel(nn.Module):
    def __init__(self, hidden_dim, embedding_model, output_dim):
        super().__init__()
        self.layer = nn.Linear(hidden_dim, output_dim)
        self.embedding_model = embedding_model

    def forward(self, X):
        embs = self.embedding_model(X)
        logits = self.layer(embs)
        return logits


class RepLearner:
    def __init__(self,
            vocab_size,
            embed_dim,
            hidden_dim,
            output_dim=2,
            n_tasks=3,
            embedding=None,
            freeze_embedding=False,
            batch_size=128,
            eta=0.01,
            max_iter=100,
            device=None):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_tasks = n_tasks
        self.embedding = embedding
        self.freeze_embedding = freeze_embedding
        self.batch_size = batch_size
        self.eta = eta
        self.max_iter = max_iter
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        # Initialize the model here instead of in `fit` so that we
        # can study the parameters before training as well as after.
        self.embedding_model = RepEmbeddingModel(
            vocab_size, embed_dim, hidden_dim,
            embedding=embedding,
            freeze_embedding=self.freeze_embedding)
        self.models = [RepTaskModel(self.hidden_dim, self.embedding_model, self.output_dim)
                       for _ in range(n_tasks)]
        self.embedding = self.embedding_model.embedding.weight.detach().cpu().numpy()

    def fit(self, X, y):
        X = torch.tensor(X)
        ys = [torch.tensor(y) for y in zip(*y)]
        dataset = torch.utils.data.TensorDataset(X, *ys)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True)
        for model in self.models:
            model.to(self.device)
            model.train()
        self.optimizers = [torch.optim.Adam(mod.parameters(), lr=self.eta)
                           for mod in self.models]
        loss = nn.CrossEntropyLoss()
        for iteration in range(1, self.max_iter+1):
            epoch_error = 0.0
            for batch in dataloader:
                X_batch = batch[0].to(self.device)
                ys_batch = batch[1: ]
                errs = []
                for y_batch, mod in zip(ys_batch, self.models):
                    y_batch = y_batch.to(self.device)
                    preds = mod(X_batch)
                    errs.append(loss(preds, y_batch))
                epoch_error += sum(errs)
                for opt, err in zip(self.optimizers, errs):
                    opt.zero_grad()
                    err.backward()
                    opt.step()
            progress_bar(
                "Pretraining - finished epoch {} of {}; error is {}".format(
                    iteration, self.max_iter, epoch_error))
        # Make sure the embedding parameter is updated:
        self.embedding = self.embedding_model.embedding.weight.detach().cpu().numpy()
        return self

    def predict_proba(self, X, device=None):
        # It can be good to do prediction on a CPU if `X` needs
        # to be large.
        if device is not None:
            device = torch.device(device)
        else:
            device = self.device
        for model in self.models:
            model.to(device)
            model.eval()
        with torch.no_grad():
            X = torch.tensor(X)
            X = X.to(device)
            preds = []
            for model in self.models:
                p = model(X)
                p = torch.softmax(p, dim=1).cpu().numpy()
                preds.append(p)
        # Return the model to the original device:
        model.to(self.device)
        return preds

    def predict(self, X):
        probs = self.predict_proba(X)
        return list(zip(*[p.argmax(axis=1) for p in probs]))
