from itertools import cycle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scikits.bootstrap as bootstrap


plt.style.use("rmts")


class ComparativeViz:

    COLORS = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']

    def __init__(self,
            df,
            experiment_type="equality",
            secondary_col="embed_dim",
            accuracy_col="accuracy",
            train_size_col="train_size",
            title=None,
            fixed_col_val=None,
            max_cols=['alpha', 'learning_rate'],
            max_cols_method='mean',
            errorbars=True,
            xlim=None,
            ylim=[0.46, 1.01],
            output_dirname="fig",
            xlabel="Train examples",
            ylabel="Mean accuracy (20 runs)",
            legend_placement="upper left",
            xtick_interval=20000):

        self.df = df
        self.experiment_type = experiment_type
        self.secondary_col = secondary_col
        self._fixed_col_val = fixed_col_val
        self.train_size_col = train_size_col
        self._title = title
        self.accuracy_col = accuracy_col
        self.max_cols = max_cols
        self.max_cols_method = max_cols_method
        self.errorbars = errorbars
        self._set_texts()
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xtick_interval = xtick_interval
        if self.xtick_interval is not None:
            self._set_xticks()
        self.xlim = xlim
        self.ylim = ylim
        self.legend_placement = legend_placement
        self.output_dirname = output_dirname

    @property
    def fixed_col_val(self):
        return self._fixed_col_val

    @fixed_col_val.setter
    def fixed_col_val(self, val):
        self._fixed_col_val = val
        self._set_texts()

    def create(self, to_file=True):
        fig, ax = plt.subplots(figsize=(7, 5))
        colorcycle = cycle(self.COLORS)

        if self.fixed_col_val is not None:
            df = self.df[self.df[self.fixed_col] == self.fixed_col_val]
        else:
            df = self.df

        mean_accuracies = df.groupby(self.secondary_col).apply(
            lambda group_df: self._plot_secondary(
                group_df, ax, color=next(colorcycle)))

        ax.set_title(self.title)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_ylim(self.ylim)
        if self.xtick_interval is not None:
            ax.set_xticks(self.xticks)
        if to_file:
            self._to_file()

        return mean_accuracies

    def create_all(self):
        self.fixed_col_val = None
        self.create()
        fixeds = sorted(self.df[self.fixed_col].unique())
        for val in fixeds:
            self.fixed_col_val = val
            self.create()

    def _plot_secondary(self, group_df, ax, color):
        name = group_df.name
        if self.max_cols is not None:
            if self.max_cols_method == 'smallest':
                group_df = self._get_best_values_from_smallest_train_size_col(group_df)
            else:
                group_df = self._get_best_vals(group_df)
        grp = group_df.groupby(self.train_size_col)
        grp_acc = grp[self.accuracy_col]
        mu = grp_acc.mean()
        ax.plot(mu.index, mu, color=color, lw=2, label=name)
        #ax.text(max(self.xticks), mu.iloc[-1], name, va='center', fontsize=16)
        if self.xlim is not None:
            ax.set_xlim(self.xlim)
        if self.errorbars:
            upper, lower = self._bootstrap_errbars(grp_acc, mu)
            #ax.errorbar(mu.index, mu, yerr=[lower, upper], linestyle='', color=color, lw=1)
            ax.fill_between(mu.index, lower, upper, color=color, alpha=0.2)
        ax.legend(loc=self.legend_placement)
        return mu

    def _get_best_vals(self, group_df):
        maxes = group_df.groupby(self.max_cols).apply(
            lambda x: x[self.accuracy_col].mean()).idxmax()
        for colname, val in zip(self.max_cols, maxes):
            group_df = group_df[group_df[colname] == val]
        return group_df

    def _get_best_values_from_smallest_train_size_col(self, group_df):
        min_train_size = group_df[self.train_size_col].min()
        zero = group_df[group_df[self.train_size_col] == min_train_size]
        maxes = zero.groupby(self.max_cols).apply(
            lambda x: x[self.accuracy_col].mean()).idxmax()
        for colname, val in zip(self.max_cols, maxes):
            group_df = group_df[group_df[colname] == val]
        return group_df

    def _to_file(self):
        output_filename = (
            f"{self.experiment_type}-{self.train_size_col}-"
            f"{self.secondary_col}-{self.fixed_col}={self.fixed_col_val}.pdf")
        output_filename = os.path.join(self.output_dirname, output_filename)
        plt.tight_layout()
        plt.savefig(output_filename, dpi=200)
    def _set_texts(self):
        if self.secondary_col == "embed_dim":
            self.fixed_col = "hidden_dim"
        else:
            self.fixed_col = "embed_dim"
        if self.secondary_col == "embed_dim":
            self.title = "Embedding dimensionality"
            self.fixed_label = "hidden"
        else:
            self.title = "Hidden dimensionality"
            self.fixed_label = "embedding"
        if self.fixed_col_val is not None:
            self.title += f"; {self.fixed_label} = {self.fixed_col_val}"
        if self._title is not None:
            self.title = self._title

    def _set_xticks(self):
        xtick_vals = self.df[self.train_size_col]
        self.xticks = list(np.arange(xtick_vals.min(), xtick_vals.max()+1, self.xtick_interval))
        if xtick_vals.max() not in self.xticks:
           self.xticks.append(xtick_vals.max())

    @staticmethod
    def _bootstrap_errbars(accuracy_df, mu):
        upper, lower = zip(*accuracy_df.apply(bootstrap.ci))
        #lower = mu - lower
        #upper = upper - mu
        return lower, upper


if __name__ == '__main__':

    df = pd.read_csv(os.path.join("results", "pretrain_probs-experiments.csv"))

    viz = ComparativeViz(
        df,
        experiment_type="pretrain_probs",
        output_dirname="tmp",
        fixed_col_val=50,
        errorbars=False)

    #viz.create_all()

    print(viz.create())
