# Relational reasoning and generalization using non-symbolic neural networks

This is the code repository for

> Anonymous. 2020. 'Relational reasoning and generalization using non-symbolic neural networks'.  Submission to the 42nd Annual Meeting of the Cognitive Science Society.

The supplementary materials for the paper are `relational-learning-cogsci2020-supplement.pdf`.


## Requirements

This code requires Python 3.6 or higher. Specific requirements are given in `requirements.txt`. For installing [TensorFlow](https://www.tensorflow.org) and [PyTorch](https://pytorch.org), we recommend following the specific instructions provided at thise projects' websites.


## Model 1: Same-different relations

* The core model is an `sklearn.neural_network.MLPClassifier`.

* `equality_experiment.py` is a framework for running the experiments, including all hyperparameter tuning.

* The experiment runs are in `run_basic_equality.ipynb`.

* The results are `results/equality-results-large.csv` and `results/equality-results-small.csv`. The latter does denser exploration of smaller training sets.


## Model 2: Sequential same-different (ABA task)

* The core model is a language model with a mean-squared error loss, implemented in PyTorch. The code is `torch_fuzzy_lm.py`.

* `fuzzy_lm_experiment.py` is a framework for running the experiments, including all hyperparameter tuning.

* The experiment runs are in `run_fuzzy_lm.ipynb`.

* The results are `results/fuzzy-lm-results-vocab50.csv`, ``results/fuzzy-lm-results-vocab20.csv`, and `results/fuzzy-lm-results-vocab10.csv`, which help to reveal the impact of vocab (train set) size on the results.


## Model 3: Hierarchical same-different relations

* For the train-from-scratch versions:
  * The core model is an `sklearn.neural_network.MLPClassifier`, as in Model 1.
  * `run_flat_premack.ipynb` runs the experiments, using `equality_experiment.py`.
  * The results are `results/flatpremack-results-h1.csv` (one hidden later) and `results/flatpremack-results-h2.csv` (two hidden layers).

* For the pretraining regime:
  * The core model is `tf_tree.py`.
  * `run_inputasoutput.py` runs the experiments, including all hyperparameter tuning.
  * The results are `results/inputasoutput-results.csv


## Visualization

The notebook `create_visualizations.ipynb` runs all the visualization, using `comparative_viz.py` and the results files in `results`. The visualizations are written to the `fig` directory.


## Other files

* `datasets.py`: Classes for creating datasets for the experiments.

* `view_best_hyperparameters.ipynb`: can be used to see which hyperparameters are optimal for the experiments, using the files in `results`.

* `rmts.mplstyle`: matplotlib style file for visualizations. This needs to be placed in the directory that matplotlib looks for such files, which depends somewhat on the system: https://matplotlib.org/users/style_sheets.html

* The `test` directory contains unit tests.
