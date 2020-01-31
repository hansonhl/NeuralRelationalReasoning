import tensorflow as tf
import numpy as np
import os



class TfTree():
    def __init__(self,
            embed_dim=50,
            hidden_dim=50,
            hidden_activation=tf.nn.relu,
            batch_size=200,
            max_iter=1,
            lr=0.01,
            alpha= 0.0001,
            pretrain=False,
            name = ""):
        self.embed_dim = embed_dim
        self.intermediate_supervision = False
        self.final_supervision = True
        self.hidden_dim = hidden_dim
        self.hidden_activation = hidden_activation
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.lr = lr
        self.alpha = alpha
        self.model_dir = os.path.join("PremackTree",  str(lr) + str(alpha) + str(embed_dim) + str(hidden_dim) + name)

    def fit(self, X, y):
        self.classes_ = [0,1]
        self.n_classes_ = len(self.classes_)
        self.estimator = tf.estimator.Estimator(
            model_fn=self._model_fn,
            model_dir=self.model_dir)
        tf.estimator.WarmStartSettings(ckpt_to_initialize_from=self.model_dir)
        input_fn = lambda: self._train_input_fn(X, y)
        self.estimator.train(input_fn)
        return self

    def pretrain(self,X,y):
        self.classes_ = [0,1]
        self.n_classes_ = len(self.classes_)
        self.estimator = tf.estimator.Estimator(
            model_fn=self._model_fn,
            model_dir=self.model_dir)
        input_fn = lambda: self._train_input_fn(X, y)
        self.final_supervision = False
        self.intermediate_supervision = True
        self.estimator.train(input_fn)
        self.final_supervision = True
        self.intermediate_supervision = False

    def set_lr_and_l2(self,lr, l2):
        self.lr = lr
        self.alpha = l2


    def _train_input_fn(self, X, y):
        X = np.array(X)
        y = np.array(y)
        dataset = tf.data.Dataset.from_tensor_slices(({'X': X}, {'y': y}))
        dataset = (dataset
                    .shuffle(len(X))
                    .repeat(self.max_iter)
                    .batch(self.batch_size))
        return dataset

    def _test_input_fn(self, X):
        dataset = tf.data.Dataset.from_tensor_slices({'X': X})
        dataset = dataset.batch(len(X))
        return dataset

    def _model_fn(self, features, labels, mode):
        features = features['X']
        # Graph:
        left_node = tf.layers.dense(
            features[:,:self.embed_dim*2],
            self.hidden_dim,
            activation=self.hidden_activation,
            name="tree")
        left_logits = tf.layers.dense(
            left_node,
            self.n_classes_,
            name="squish")
        right_node = tf.layers.dense(
            features[:,self.embed_dim*2:],
            self.hidden_dim,
            activation=self.hidden_activation,
            name="tree",
            reuse=True)
        right_logits = tf.layers.dense(
            right_node,
            self.n_classes_,
            name="squish",
            reuse=True)
        final_node = tf.layers.dense(
            tf.concat([right_node,left_node],1),
            self.hidden_dim,
            activation=self.hidden_activation,
            name="tree",
            reuse=True)
        final_logits = tf.layers.dense(
            final_node,
            self.n_classes_,
            name="squish",
            reuse=True)
        # Predictions:
        right_preds = tf.argmax(right_logits, axis=-1)
        left_preds = tf.argmax(left_logits, axis=-1)
        final_preds = tf.argmax(final_logits, axis=-1)
        # Predicting:
        if mode == tf.estimator.ModeKeys.PREDICT:
            results = { 'final_pred': final_preds, 'right_pred': right_preds, 'left_pred' :left_preds }
            return tf.estimator.EstimatorSpec(mode, predictions=results)
        else:
            labels = labels['y']
            right_labels = labels[:,0]
            left_labels = labels[:,1]
            final_labels = labels[:,2]
            loss = 0
            if self.final_supervision:
                with tf.variable_scope('tree', reuse=True):
                    loss += self.alpha * tf.nn.l2_loss(tf.get_variable('kernel',dtype=tf.float64))
                loss += tf.losses.sparse_softmax_cross_entropy(
                    logits=final_logits, labels=final_labels)
                #Hacky solution to get graphs to match
                with tf.variable_scope('tree', reuse=True):
                    loss += self.alpha * tf.nn.l2_loss(tf.get_variable('kernel',dtype=tf.float64))*0
                loss += tf.losses.sparse_softmax_cross_entropy(
                    logits=right_logits, labels=right_labels)*0
                loss += tf.losses.sparse_softmax_cross_entropy(
                    logits=left_logits, labels=left_labels)*0
            if self.intermediate_supervision:
                with tf.variable_scope('tree', reuse=True):
                    loss += self.alpha * tf.nn.l2_loss(tf.get_variable('kernel',dtype=tf.float64))
                loss += tf.losses.sparse_softmax_cross_entropy(
                    logits=right_logits, labels=right_labels)
                loss += tf.losses.sparse_softmax_cross_entropy(
                    logits=left_logits, labels=left_labels)
                #Hacky solution to get graphs to match
                with tf.variable_scope('tree', reuse=True):
                    loss += self.alpha * tf.nn.l2_loss(tf.get_variable('kernel',dtype=tf.float64))*0
                loss += tf.losses.sparse_softmax_cross_entropy(
                    logits=final_logits, labels=final_labels)*0
            metrics = {
                'final_accuracy': tf.metrics.accuracy(final_labels, final_preds),
                'right_accuracy': tf.metrics.accuracy(right_labels, right_preds),
                'left_accuracy': tf.metrics.accuracy(left_labels, left_preds),
            }
            # Evaluation mode to enable early stopping based on metrics:
            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(
                    mode, loss=loss, eval_metric_ops=metrics)
            # Training:
            elif mode == tf.estimator.ModeKeys.TRAIN:
                global_step = tf.train.get_or_create_global_step()
                train_op = tf.train.AdamOptimizer(self.lr).minimize(
                    loss, global_step=global_step)
                return tf.estimator.EstimatorSpec(
                    mode, loss=loss, train_op=train_op)

    def predict_proba(self, X):
        input_fn = lambda: self._test_input_fn(X)
        return [x['proba'] for x in self.estimator.predict(input_fn)]

    def predict(self, X):
        input_fn = lambda: self._test_input_fn(X)
        return [self.classes_[x['final_pred']] for x in self.estimator.predict(input_fn)]

    def predict_intermediate(self, X):
        input_fn = lambda: self._test_input_fn(X)
        return [self.classes_[x['right_pred']] for x in self.estimator.predict(input_fn)] + [self.classes_[x['left_pred']] for x in self.estimator.predict(input_fn)]
