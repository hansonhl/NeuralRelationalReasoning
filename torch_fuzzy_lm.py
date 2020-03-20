import torch
import torch.nn as nn
import sys


START_SYMBOL = "<s>"
END_SYMBOL = "</s>"


class FuzzyPatternModule(nn.Module):
    def __init__(self,
            vocab_size,
            embed_dim,
            hidden_dim,
            rnn_cell_class,
            num_layers=1,
            dropout=0,
            output_activation=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.rnn_cell_class = rnn_cell_class
        self.num_layers = num_layers
        self.dropout = dropout
        self.rnn = self.rnn_cell_class(
            self.embed_dim,
            self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True)
        if output_activation:
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = lambda x: x
        self.output_layer = nn.Linear(self.hidden_dim, self.embed_dim)
        self.classifier_layer = nn.Linear(self.embed_dim, 2)

    def forward(self, seq, hidden):
        output, hidden = self.rnn(seq, hidden)
        output = self.output_activation(self.output_layer(output))
        return hidden, output


class FuzzyPatternLM:
    def __init__(self,
            vocab,
            embed_dim,
            hidden_dim,
            rnn_cell_class=nn.LSTM,
            num_layers=1,
            alpha=0,
            dropout=0,
            eta=0.05,
            max_iter=10,
            warm_start=False,
            output_activation=None,
            verbose=10,
            device=None):
        self.vocab = sorted(vocab)
        self.vocab_size = len(vocab)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.rnn_cell_class = rnn_cell_class
        self.num_layers = num_layers
        self.alpha = alpha
        self.dropout = dropout
        self.eta = eta
        self.max_iter = max_iter
        self.warm_start = warm_start
        self.output_activation = output_activation
        self.verbose = verbose
        self.model = FuzzyPatternModule(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            rnn_cell_class=self.rnn_cell_class,
            output_activation=self.output_activation,
            num_layers=self.num_layers,
            dropout=self.dropout)
        self.optimizer_func = torch.optim.Adam
        self.start_symbol_index = self.vocab.index(START_SYMBOL)
        self.end_symbol_index = self.vocab.index(END_SYMBOL)
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        # Freezing the embedding space seems like it might help
        # with generalization into new vocabularies:
        self.embedding.weight.requires_grad = False
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

    def fit(self, X, y=None, eval_func=None):
        if not self.warm_start or not hasattr(self, "optimizer"):
            self.optimizer = self.optimizer_func(
                self.model.parameters(), lr=self.eta, weight_decay=self.alpha)
            self.results = []

        # Map the sequences into lists of indices into `self.embedding`:
        X = [[self.vocab.index(c) for c in seq] for seq in X]
        X = torch.LongTensor(X)

        if y is not None:
            self.fit_classifier(X, y)
        else:
            self.fit_lm(X, eval_func=eval_func)

    def fit_classifier(self, X, y):
        X = self.embedding(X)
        y = torch.LongTensor(y)
        loss = nn.CrossEntropyLoss()
        for iteration in range(1, self.max_iter+1):
            epoch_error = 0.0
            hidden, output = self.model(X, hidden=None)
            logits = self._get_logits(output)
            err = loss(logits, y)
            epoch_error += err.item()
            self.optimizer.zero_grad()
            err.backward()
            self.optimizer.step()
            if self.verbose and iteration % self.verbose == 0:
                self.progress_bar(f"Epoch {iteration}; err = {epoch_error}")

    def _get_logits(self, output):
        output = output[:, -1, : ]
        logits = self.model.classifier_layer(output)
        return logits

    def fit_lm(self, X, eval_func=None, eval_increment=10):
        # Convert the input into a list of 1-dimensional sequences.
        # This makes each row a batch of individual timesteps, which
        # helps with the recurrence.
        X = X.unsqueeze(0).T

        X = X.to(self.device)

        loss = nn.MSELoss()

        self.model.to(self.device)
        self.embedding.to(self.device)

        self.model.train()

        for iteration in range(1, self.max_iter+1):
            hidden = None
            timestep = self.embedding(X[0])
            err = 0.0
            # Iterate through the remaining characters:
            for i in range(1, len(X)):
                hidden, output = self.model(timestep, hidden)
                timestep = self.embedding(X[i])
                # Autoencoder-style loss based on the embeddings -- compares
                # the current timestep with the predicted output from the
                # previous timestep:
                err += loss(output, timestep)
            self.optimizer.zero_grad()
            err.backward()
            self.optimizer.step()
            epoch_error = err.item() / len(X)
            if self.verbose and iteration % self.verbose == 0:
                self.progress_bar(f"Epoch {iteration}; err = {epoch_error}")
            if eval_func is not None and iteration % eval_increment == 0:
                self.model.to("cpu")
                self.embedding.to("cpu")
                these_results = eval_func(self)
                these_results.update({'iteration': iteration})
                self.results.append(these_results)
                self.model.train()
                self.model.to(self.device)
                self.embedding.to(self.device)

    def predict_one(self, initial_words, max_length=3):
        initial_words = list(initial_words)
        initial_indices = [self.vocab.index(c) for c in initial_words]
        self.model.eval()
        with torch.no_grad():
            initial_indices = self.embedding(torch.LongTensor([initial_indices]))
            preds = initial_words
            hidden, output = self.model(initial_indices, hidden=None)
            # Get the logits from the last state in initial:
            output = output[:, -1, : ].unsqueeze(0)
            p, output = self.get_letter_prediction(output)
            preds.append(p)
            if p == END_SYMBOL:
                return preds
            for i in range(max_length):
                hidden, output = self.model(output, hidden)
                p, output = self.get_letter_prediction(output)
                preds.append(p)
                if p == END_SYMBOL:
                    break
            return preds

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X = [[self.vocab.index(c) for c in seq] for seq in X]
            X = torch.LongTensor(X)
            X = self.embedding(X)
            hidden, output = self.model(X, hidden=None)
            logits = self._get_logits(output)
            preds = torch.softmax(logits, dim=1).cpu().numpy()
            return preds

    def predict(self, X):
        probs = self.predict_proba(X)
        return probs.argmax(axis=1)

    def get_letter_prediction(self, output):
        dists = torch.cdist(self.embedding.weight, output, p=2)
        idx = dists.argmin()
        p = self.vocab[idx.item()]
        output = self.embedding(idx.reshape(1,1))
        return p, output

    @staticmethod
    def progress_bar(msg):
        sys.stderr.write('\r')
        sys.stderr.write(msg)
        sys.stderr.flush()
