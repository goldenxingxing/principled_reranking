#!/usr/bin/python3
# -*- coding: utf-8 -*-
import tfplus
import tensorflow as tf
from alps.util import logger
from alps.core.global_vars import global_context
from tensorflow.contrib.rnn import GRUCell, static_bidirectional_rnn, LSTMCell, MultiRNNCell
from tensorflow.contrib.layers.python.layers.embedding_ops import embedding_lookup_unique
import numpy as np
import keras


class EmbeddingDNN(tf.estimator.Estimator):

    def __init__(self, config, params):

        def _model_fn(features, labels, mode, params):
            """
                _model_fn
            """
            experiment = global_context().experiment
            self.max_time_len = params['listcnt']
            model_name = params.get('model_name', 'PRM')
            if experiment.eval is None or not experiment.eval.parallel_validation:
                if mode == tf.estimator.ModeKeys.TRAIN:
                    keras.backend.set_learning_phase(1)
                    tf.keras.backend.set_learning_phase(1)
                else:
                    keras.backend.set_learning_phase(0)
                    tf.keras.backend.set_learning_phase(0)

            for key, val in features.items():
                if isinstance(val, tuple):
                    features[key] = list(val)

            if model_name == 'PRM':
                model = PRM(params['eb_dim'], params['hidden_size'], self.max_time_len, params['keep_prob'],
                            params['l2_reg'], params['lr'],
                            max_norm=params['max_norm'], itm_dens_num=params['itm_dens_num'],
                            profile_num=params['profile_num'])
            elif model_name == 'PRM_O1':
                model = PRM_O1(params['eb_dim'], params['hidden_size'], self.max_time_len, params['keep_prob'],
                               params['l2_reg'], params['lr'],
                               max_norm=params['max_norm'], itm_dens_num=params['itm_dens_num'],
                               profile_num=params['profile_num'])
            elif model_name == 'PRM_O2':
                model = PRM_O2(params['eb_dim'], params['hidden_size'], self.max_time_len, params['keep_prob'],
                               params['l2_reg'], params['lr'],
                               max_norm=params['max_norm'], itm_dens_num=params['itm_dens_num'],
                               profile_num=params['profile_num'])
            elif model_name == 'PRM_O6':
                model = PRM_O6(params['eb_dim'], params['hidden_size'], self.max_time_len, params['keep_prob'],
                               params['l2_reg'], params['lr'],
                               max_norm=params['max_norm'], itm_dens_num=params['itm_dens_num'],
                               profile_num=params['profile_num'])
            elif model_name == 'SetRank':
                model = SetRank(params['eb_dim'], params['hidden_size'], self.max_time_len, params['keep_prob'],
                                params['l2_reg'], params['lr'],
                                max_norm=params['max_norm'], itm_dens_num=params['itm_dens_num'],
                                profile_num=params['profile_num'])
            elif model_name == 'DLCM':
                model = DLCM(params['eb_dim'], params['hidden_size'], self.max_time_len, params['keep_prob'],
                             params['l2_reg'], params['lr'],
                             max_norm=params['max_norm'], itm_dens_num=params['itm_dens_num'],
                             profile_num=params['profile_num'])
            elif model_name == 'GSF':
                model = GSF(params['eb_dim'], params['hidden_size'], self.max_time_len, params['keep_prob'],
                            params['l2_reg'], params['lr'],
                            max_norm=params['max_norm'], itm_dens_num=params['itm_dens_num'],
                            profile_num=params['profile_num'])
            elif model_name == 'miDNN':
                model = miDNN(params['eb_dim'], params['hidden_size'], self.max_time_len, params['keep_prob'],
                              params['l2_reg'], params['lr'],
                              max_norm=params['max_norm'], itm_dens_num=params['itm_dens_num'])
            elif model_name == 'PEAR':
                model = PEAR(params['eb_dim'], params['hidden_size'], self.max_time_len, params['keep_prob'],
                             params['l2_reg'], params['lr'], max_norm=params['max_norm'],
                             itm_dens_num=params['itm_dens_num'], profile_num=params['profile_num'],
                             n_head=1, cross_n_head=1, use_hist_seq=False)
            elif model_name == 'RAISE':
                model = RAISE(params['eb_dim'], params['hidden_size'], self.max_time_len, params['keep_prob'],
                              params['l2_reg'], params['lr'], max_norm=params['max_norm'],
                              itm_dens_num=params['itm_dens_num'], profile_num=params['profile_num'],
                              a_head=1, d_model=32, d_inner_hid=64)
            elif model_name == 'SetRank_O6':
                model = SetRank_O6(params['eb_dim'], params['hidden_size'], self.max_time_len, params['keep_prob'],
                                   params['l2_reg'], params['lr'],
                                   max_norm=params['max_norm'], itm_dens_num=params['itm_dens_num'],
                                   profile_num=params['profile_num'])
            elif model_name == 'DLCM_O6':
                model = DLCM_O6(params['eb_dim'], params['hidden_size'], self.max_time_len, params['keep_prob'],
                                params['l2_reg'], params['lr'],
                                max_norm=params['max_norm'], itm_dens_num=params['itm_dens_num'],
                                profile_num=params['profile_num'])
            elif model_name == 'GSF_O6':
                model = GSF_O6(params['eb_dim'], params['hidden_size'], self.max_time_len, params['keep_prob'],
                               params['l2_reg'], params['lr'],
                               max_norm=params['max_norm'], itm_dens_num=params['itm_dens_num'],
                               profile_num=params['profile_num'])
            elif model_name == 'miDNN_O6':
                model = miDNN_O6(params['eb_dim'], params['hidden_size'], self.max_time_len, params['keep_prob'],
                                 params['l2_reg'], params['lr'],
                                 max_norm=params['max_norm'], itm_dens_num=params['itm_dens_num'])
            elif model_name == 'PEAR_O6':
                model = PEAR_O6(params['eb_dim'], params['hidden_size'], self.max_time_len, params['keep_prob'],
                                params['l2_reg'], params['lr'], max_norm=params['max_norm'],
                                itm_dens_num=params['itm_dens_num'], profile_num=params['profile_num'],
                                n_head=1, cross_n_head=1, use_hist_seq=False)
            elif model_name == 'RAISE_O6':
                model = RAISE_O6(params['eb_dim'], params['hidden_size'], self.max_time_len, params['keep_prob'],
                                 params['l2_reg'], params['lr'], max_norm=params['max_norm'],
                                 itm_dens_num=params['itm_dens_num'], profile_num=params['profile_num'],
                                 a_head=1, d_model=32, d_inner_hid=64)
            elif model_name == 'SetRank_O1':
                model = SetRank_O1(params['eb_dim'], params['hidden_size'], self.max_time_len, params['keep_prob'],
                                   params['l2_reg'], params['lr'],
                                   max_norm=params['max_norm'], itm_dens_num=params['itm_dens_num'],
                                   profile_num=params['profile_num'])
            elif model_name == 'DLCM_O1':
                model = DLCM_O1(params['eb_dim'], params['hidden_size'], self.max_time_len, params['keep_prob'],
                                params['l2_reg'], params['lr'],
                                max_norm=params['max_norm'], itm_dens_num=params['itm_dens_num'],
                                profile_num=params['profile_num'])
            elif model_name == 'GSF_O1':
                model = GSF_O1(params['eb_dim'], params['hidden_size'], self.max_time_len, params['keep_prob'],
                               params['l2_reg'], params['lr'],
                               max_norm=params['max_norm'], itm_dens_num=params['itm_dens_num'],
                               profile_num=params['profile_num'])
            elif model_name == 'miDNN_O1':
                model = miDNN_O1(params['eb_dim'], params['hidden_size'], self.max_time_len, params['keep_prob'],
                                 params['l2_reg'], params['lr'],
                                 max_norm=params['max_norm'], itm_dens_num=params['itm_dens_num'])
            elif model_name == 'PEAR_O1':
                model = PEAR_O1(params['eb_dim'], params['hidden_size'], self.max_time_len, params['keep_prob'],
                                params['l2_reg'], params['lr'], max_norm=params['max_norm'],
                                itm_dens_num=params['itm_dens_num'], profile_num=params['profile_num'],
                                n_head=1, cross_n_head=1, use_hist_seq=False)
            elif model_name == 'RAISE_O1':
                model = RAISE_O1(params['eb_dim'], params['hidden_size'], self.max_time_len, params['keep_prob'],
                                 params['l2_reg'], params['lr'], max_norm=params['max_norm'],
                                 itm_dens_num=params['itm_dens_num'], profile_num=params['profile_num'],
                                 a_head=4, d_model=32, d_inner_hid=64)
            elif model_name == 'SetRank_O2':
                model = SetRank_O2(params['eb_dim'], params['hidden_size'], self.max_time_len, params['keep_prob'],
                                   params['l2_reg'], params['lr'],
                                   max_norm=params['max_norm'], itm_dens_num=params['itm_dens_num'],
                                   profile_num=params['profile_num'])
            elif model_name == 'DLCM_O2':
                model = DLCM_O2(params['eb_dim'], params['hidden_size'], self.max_time_len, params['keep_prob'],
                                params['l2_reg'], params['lr'],
                                max_norm=params['max_norm'], itm_dens_num=params['itm_dens_num'],
                                profile_num=params['profile_num'])
            elif model_name == 'GSF_O2':
                model = GSF_O2(params['eb_dim'], params['hidden_size'], self.max_time_len, params['keep_prob'],
                               params['l2_reg'], params['lr'],
                               max_norm=params['max_norm'], itm_dens_num=params['itm_dens_num'],
                               profile_num=params['profile_num'])
            elif model_name == 'miDNN_O2':
                model = miDNN_O2(params['eb_dim'], params['hidden_size'], self.max_time_len, params['keep_prob'],
                                 params['l2_reg'], params['lr'],
                                 max_norm=params['max_norm'], itm_dens_num=params['itm_dens_num'])
            elif model_name == 'PEAR_O2':
                model = PEAR_O2(params['eb_dim'], params['hidden_size'], self.max_time_len, params['keep_prob'],
                                params['l2_reg'], params['lr'], max_norm=params['max_norm'],
                                itm_dens_num=params['itm_dens_num'], profile_num=params['profile_num'],
                                n_head=1, cross_n_head=1, use_hist_seq=False)
            elif model_name == 'RAISE_O2':
                model = RAISE_O2(params['eb_dim'], params['hidden_size'], self.max_time_len, params['keep_prob'],
                                 params['l2_reg'], params['lr'], max_norm=params['max_norm'],
                                 itm_dens_num=params['itm_dens_num'], profile_num=params['profile_num'],
                                 a_head=1, d_model=32, d_inner_hid=64)
            else:
                model = PRM(params['eb_dim'], params['hidden_size'], self.max_time_len, params['keep_prob'],
                            params['l2_reg'], params['lr'],
                            max_norm=params['max_norm'], itm_dens_num=params['itm_dens_num'],
                            profile_num=params['profile_num'])

            prediction = model.call(features, 1 if mode == tf.estimator.ModeKeys.TRAIN else 0)

            if mode == tf.estimator.ModeKeys.PREDICT:
                export_outputs = {
                    tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                        tf.estimator.export.PredictOutput(prediction)
                }
                return tf.estimator.EstimatorSpec(
                    mode,
                    predictions=prediction,
                    export_outputs=export_outputs
                )

            # 定义loss
            loss = model.build_loss(prediction, labels['label'],
                                    1 if mode == tf.estimator.ModeKeys.TRAIN else 0)
            eval_metric_ops = model.build_eva_metrics(prediction, labels['label'])

            eval_metric_ops['p1_eq'] = SequenceEqualValidator(model.y_pred_B2, prediction)
            eval_metric_ops['p2_eq'] = SequenceEqualValidator(model.y_pred_B3, prediction)

            eval_metric_ops['p1_eqs'] = SequenceEqualScoreValidator(model.y_pred_B2, prediction)
            eval_metric_ops['p2_eqs'] = SequenceEqualScoreValidator(model.y_pred_B3, prediction)

            if mode == tf.estimator.ModeKeys.EVAL:
                # Alps1 Validation = Estimator metrics
                return tf.estimator.EstimatorSpec(
                    mode, loss=loss, eval_metric_ops=eval_metric_ops)

            # 定义training过程
            fetch_dict = {"loss": loss}

            train_op = model.opt(loss)

            return tf.estimator.EstimatorSpec(mode,
                                              loss=loss,
                                              train_op=train_op,
                                              eval_metric_ops=eval_metric_ops,
                                              training_hooks=[])

        super(EmbeddingDNN, self).__init__(config=config,
                                           params=params,
                                           model_fn=_model_fn)

    @property
    def name(self):
        """
            name
        """
        return "EmbeddingDNN"


def tau_function(x):
    return tf.where(x > 0, tf.exp(x), tf.zeros_like(x))


def attention_score(x):
    return tau_function(x) / tf.add(tf.reduce_sum(tau_function(x), axis=1, keepdims=True), 1e-20)


class DenseEmbeddingsLayer(tf.keras.layers.Layer):

    def __init__(self,
                 variable_name,
                 embedding_dim=8,
                 key_dtype=tf.int64,
                 value_dtype=tf.float32,
                 num_shard=4,
                 trainable=True,
                 collections=None,
                 **kwargs):
        super(DenseEmbeddingsLayer, self).__init__(**kwargs)
        self.variable_name = variable_name
        self.embedding_dim = embedding_dim
        self.key_dtype = key_dtype
        self.num_shard = num_shard
        self.value_dtype = value_dtype
        self.trainable = trainable
        self.collections = [tf.GraphKeys.GLOBAL_VARIABLES]
        if collections is not None:
            self.collections.append(collections)

    def build(self, input_shape):
        with tf.variable_scope("embedding_table", reuse=tf.AUTO_REUSE):
            self.embedding_table = tfplus.get_kv_variable(
                self.variable_name,
                embedding_dim=self.embedding_dim,
                key_dtype=self.key_dtype,
                value_dtype=self.value_dtype,
                initializer=tf.keras.initializers.he_normal(),
                # initializer=None,
                trainable=self.trainable,
                collections=self.collections,
                partitioner=tf.fixed_size_partitioner(num_shards=self.num_shard), enter_threshold=0)
        self.built = True

    def call(self, x, mask=None):
        return embedding_lookup_unique(self.embedding_table, x)

    def compute_output_shape(self, input_shape):
        result = list(input_shape)
        result.append(self.embedding_dim)
        return result


class BaseModel(object):
    def __init__(self, max_time_len, hidden_size, eb_dim, itm_spar_num, itm_dens_num, profile_num, max_norm,
                 keep_prob, reg_lambda, lr, use_user_feas=False):
        self.max_time_len = max_time_len
        # self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self.emb_dim = eb_dim
        self.itm_spar_num = itm_spar_num
        self.itm_dens_num = itm_dens_num
        # self.hist_spar_num = hist_spar_num
        # self.hist_dens_num = hist_dens_num
        self.profile_num = profile_num
        self.max_grad_norm = max_norm
        self.ft_num = itm_spar_num * eb_dim + itm_dens_num
        self.keep_prob = keep_prob
        self.reg_lambda = reg_lambda
        self.lr = lr
        self.use_user_feas = use_user_feas

    def process_embedding(self, features):
        # embedding 初始化
        itemEmbeddingLayer = DenseEmbeddingsLayer(variable_name="item_emb_mtx", embedding_dim=self.emb_dim)
        userEmbeddingLayer = DenseEmbeddingsLayer(variable_name="user_emb_mtx", embedding_dim=self.emb_dim)
        self.itm_spar_emb = itemEmbeddingLayer(features['item_sparse'])
        self.itm_dens_ph = tf.reshape(features['item_dense'], [-1, self.max_time_len, self.itm_dens_num])

        self.item_seq = tf.concat(
            [tf.reshape(self.itm_spar_emb, [-1, self.max_time_len, int(self.itm_spar_num * self.emb_dim)]),
             self.itm_dens_ph], axis=-1)

        if self.use_user_feas:
            self.usr_spar_emb = userEmbeddingLayer(features['user_sparse'])
            user_feas = tf.reshape(self.usr_spar_emb, (-1, 1, int(self.profile_num * self.emb_dim)))
            user_feas = tf.tile(user_feas, (1, self.max_time_len, 1))
            self.item_seq = tf.concat([self.item_seq, user_feas], axis=-1)

        self.seq_length_ph = tf.reshape(features['list_len'], (tf.shape(features['list_len'])[0],))
        return self.item_seq

    def build_fc_net(self, inp, mode, scope='fc'):
        with tf.variable_scope(scope):
            bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1', training=mode)
            fc1 = tf.layers.dense(bn1, 200, activation=tf.nn.relu, name='fc1')
            dp1 = tf.nn.dropout(fc1, self.keep_prob, name='dp1')
            fc2 = tf.layers.dense(dp1, 80, activation=tf.nn.relu, name='fc2')
            dp2 = tf.nn.dropout(fc2, self.keep_prob, name='dp2')
            fc3 = tf.layers.dense(dp2, 2, activation=None, name='fc3')
            score = tf.nn.softmax(fc3)
            score = tf.reshape(score[:, :, 0], [-1, self.max_time_len])
            # output
            seq_mask = tf.sequence_mask(self.seq_length_ph, maxlen=self.max_time_len, dtype=tf.float32)
            y_pred = seq_mask * score
        return y_pred

    def build_mlp_net(self, inp, seq_length_ph, mode, layer=(500, 200, 80), scope='mlp'):
        with tf.variable_scope(scope):
            inp = tf.layers.batch_normalization(inputs=inp, name='mlp_bn', training=mode)
            for i, hidden_num in enumerate(layer):
                fc = tf.layers.dense(inp, hidden_num, activation=tf.nn.relu, name='fc' + str(i))
                inp = tf.nn.dropout(fc, self.keep_prob, name='dp' + str(i))
            final = tf.layers.dense(inp, 1, activation=None, name='fc_final')
            score = tf.reshape(final, [-1, self.max_time_len])
            seq_mask = tf.sequence_mask(seq_length_ph, maxlen=self.max_time_len, dtype=tf.float32)
            y_pred = seq_mask * score
        return y_pred

    def build_logloss(self, y_pred, y_label):
        # loss
        loss = tf.losses.log_loss(y_label, y_pred)
        return loss

    def build_norm_logloss(self, y_pred, y_label):
        loss = - tf.reduce_sum(
            y_label / (tf.reduce_sum(y_label, axis=-1, keepdims=True) + 1e-8) * tf.log(y_pred + 1e-8))
        return loss

    def build_mseloss(self, y_pred, y_label):
        loss = tf.losses.mean_squared_error(y_label, y_pred)
        return loss

    def build_eva_metrics(self, y_pred, y_label):

        eval_metric_ops = {}
        from alps.metrics.base import accuracy, auc
        eval_metric_ops["auc"] = auc(y_label, y_pred)
        eval_metric_ops['ndcg'] = NDCGValidator(y_label, y_pred)
        eval_metric_ops['map@5'] = MAPValidator(y_label, y_pred, k=5)
        eval_metric_ops['map@10'] = MAPValidator(y_label, y_pred, k=10)
        eval_metric_ops['rec@5'] = ClickValidator(y_label, y_pred, k=5)
        eval_metric_ops['rec@10'] = ClickValidator(y_label, y_pred, k=10)

        eval_metric_ops['map@15'] = MAPValidator(y_label, y_pred, k=15)
        eval_metric_ops['map@20'] = MAPValidator(y_label, y_pred, k=20)
        eval_metric_ops['rec@15'] = ClickValidator(y_label, y_pred, k=15)
        eval_metric_ops['rec@20'] = ClickValidator(y_label, y_pred, k=20)

        return eval_metric_ops

    def build_attention_loss(self, y_pred, y_label):
        self.label_wt = attention_score(y_label)
        self.pred_wt = attention_score(y_pred)
        # self.pred_wt = y_pred

        self.label_wt = tf.maximum(self.label_wt, 1e-6)
        self.pred_wt = tf.maximum(self.pred_wt, 1e-6)
        loss = tf.losses.log_loss(self.label_wt, self.pred_wt)
        # self.loss = tf.losses.mean_squared_error(self.label_wt, self.pred_wt)
        return loss

    def opt(self, loss):
        for v in tf.trainable_variables():
            if 'bias' not in v.name and 'emb' not in v.name:
                loss += self.reg_lambda * tf.nn.l2_loss(v)
                # self.loss += self.reg_lambda * tf.norm(v, ord=1)

        # self.lr = tf.train.exponential_decay(
        #     self.lr_start, self.global_step, self.lr_decay_step,
        #     self.lr_decay_rate, staircase=True, name="learning_rate")

        self.optimizer = tfplus.train.AdamOptimizer(self.lr)
        # tf.train.AdamOptimizer(self.lr)

        from tensorflow.python.training.training_util import get_or_create_global_step
        if self.max_grad_norm > 0:
            grads_and_vars = self.optimizer.compute_gradients(loss)
            for idx, (grad, var) in enumerate(grads_and_vars):
                if grad is not None:
                    grads_and_vars[idx] = (tf.clip_by_norm(grad, self.max_grad_norm), var)
            train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=get_or_create_global_step())
        else:
            grads = self.optimizer.compute_gradients(loss)
            train_op = self.optimizer.apply_gradients(grads, global_step=get_or_create_global_step())

        return train_op

    def multihead_attention(self,
                            queries,
                            keys,
                            num_units=None,
                            num_heads=2,
                            scope="multihead_attention",
                            reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            if num_units is None:
                num_units = queries.get_shape().as_list()[-1]

            Q = tf.layers.dense(queries, num_units, activation=None)  # (N, T_q, C)
            K = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)
            V = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)

            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
            key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)
            outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

            query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
            query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
            outputs *= query_masks  # broadcasting. (N, T_q, C)
            outputs = tf.nn.dropout(outputs, self.keep_prob)
            outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        return outputs

    def positionwise_feed_forward(self, inp, d_hid, d_inner_hid, mode, dropout=0.9):
        with tf.variable_scope('pos_ff'):
            inp = tf.layers.batch_normalization(inputs=inp, name='bn1', training=mode)
            l1 = tf.layers.conv1d(inp, d_inner_hid, 1, activation='relu')
            l2 = tf.layers.conv1d(l1, d_hid, 1)
            dp = tf.nn.dropout(l2, dropout, name='dp')
            dp = dp + inp
            output = tf.layers.batch_normalization(inputs=dp, name='bn2', training=mode)
        return output

    def bilstm(self, inp, hidden_size, scope='bilstm', reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, name='cell_fw')
            lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, name='cell_bw')

            outputs, state_fw, state_bw = static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, inp, dtype='float32')
        return outputs, state_fw, state_bw

    def get_sinusoidal_positional_encoding(self, maxlen, emb_dim,
                                           scope="positional_encoding"):
        '''Sinusoidal Positional_Encoding. See 3.5
        inputs: 3d tensor. (N, T, E)
        maxlen: scalar. Must be >= T
        masking: Boolean. If True, padding positions are set to zeros.
        scope: Optional scope for `variable_scope`.

        returns
        3d tensor that has the same shape as inputs.
        '''

        # E = inputs.get_shape().as_list()[-1]  # static
        # N, T = tf.shape(inputs)[0], tf.shape(inputs)[1]  # dynamic
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # position indices
            # position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])  # (N, T)

            # First part of the PE function: sin and cos argument
            position_enc = np.array([
                [pos / np.power(10000, (i - i % 2) / emb_dim) for i in range(emb_dim)]
                for pos in range(maxlen)])

            # Second part, apply the cosine to even columns and sin to odds.
            position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
            position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
            position_enc = tf.convert_to_tensor(position_enc, tf.float32)  # (maxlen, E)

            # lookup
            # outputs = tf.nn.embedding_lookup(position_enc, position_ind)
            #
            # # masks
            # if masking:
            #     outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)

        # return tf.to_float(outputs)
        return position_enc

    def position_from_score(self, pred):
        orig_pos = tf.argsort(pred, direction='DESCENDING', stable=True)
        orig_pos = tf.argsort(orig_pos, direction='ASCENDING', stable=True)
        return orig_pos

    def contrastive_loss(self, x1, x2):
        Y1 = self.position_from_score(x1)
        Y2 = self.position_from_score(x2)
        Y = tf.abs(Y2 - Y1)
        mse = tf.pow(x1 - x2, 2)
        loss = tf.cast(Y, tf.float32) * mse
        return tf.reduce_mean(loss)

    def position_adjust_from_score(self, pred):
        """
        按照score输入的score进行排序
        :param pred:
        :return:
        """
        y_pred_pos = tf.cast(tf.argsort(pred, direction='ASCENDING', stable=True), tf.float32)
        y_pred_pos_1 = tf.cast(tf.argsort(y_pred_pos, direction='ASCENDING', stable=True), tf.float32)
        rnd_index = tf.argsort(tf.random.uniform(shape=tf.shape(y_pred_pos_1)), direction='DESCENDING', stable=True)
        rnd_index = tf.cast(rnd_index < 1, tf.float32) * 1.5
        return self.position_from_score(y_pred_pos_1 - rnd_index)

    def position_adjust_bak(self, pred):
        pos = tf.tile(tf.reshape(tf.range(tf.shape(pred)[1]), (1, -1)), (tf.shape(pred)[0], 1))
        pos = tf.cast(pos, tf.float32)
        rnd_index = tf.cast(tf.argsort(tf.random.uniform(shape=tf.shape(pos)), direction='DESCENDING', stable=True),
                            tf.float32)
        rnd_index = tf.cast(rnd_index < 1, tf.float32) * 1.5 * (tf.sign(pos) * 2.0 - 1.0)
        y_pred_pos = tf.cast(tf.argsort(pos - rnd_index, direction='ASCENDING', stable=True), tf.float32)
        y_pred_pos_1 = tf.argsort(y_pred_pos, direction='ASCENDING', stable=True)
        return y_pred_pos_1

    def position_adjust(self, pred):
        """
        随机乱序
        :param pred:
        :return:
        """
        print(tf.shape(pred)[1])
        pos = tf.tile(tf.reshape(tf.range(tf.shape(pred)[1]), (1, -1)), (tf.shape(pred)[0], 1))
        pos = tf.cast(pos, tf.float32)
        rnd = tf.random.uniform(shape=tf.shape(pos))
        rnd_0, rnd_1 = tf.split(rnd, [1, tf.shape(pred)[1] - 1], axis=-1)
        rnd_index = tf.cast(tf.argsort(rnd_1, direction='DESCENDING', stable=True), tf.float32)
        rnd_index = tf.cast(rnd_index < 1, tf.float32) * 1.5
        rnd_index = tf.concat([rnd_0 * 0., rnd_index], axis=-1)
        y_pred_pos = tf.cast(tf.argsort(pos - rnd_index, direction='ASCENDING', stable=True), tf.float32)
        y_pred_pos_1 = tf.argsort(y_pred_pos, direction='ASCENDING', stable=True)
        return y_pred_pos_1


class PRM(BaseModel):
    def __init__(self, eb_dim, hidden_size, max_time_len, keep_prob, reg_lambda, lr,
                 profile_num=8, itm_spar_num=5, itm_dens_num=1, max_norm=None, d_model=64,
                 d_inner_hid=128, n_head=1):
        self.d_model = d_model
        self.d_inner_hid = d_inner_hid
        self.n_head = n_head
        super(PRM, self).__init__(max_time_len, hidden_size, eb_dim, itm_spar_num, itm_dens_num, profile_num, max_norm,
                                  keep_prob, reg_lambda, lr)

    def call(self, features, mode):
        self.process_embedding(features)

        self.pos_dim = pos_dim = self.item_seq.get_shape().as_list()[-1]
        self.pos_mtx = tf.get_variable("pos_mtx", [self.max_time_len, pos_dim],
                                       initializer=tf.truncated_normal_initializer)
        item_seq = self.add_pos_embedding(self.item_seq, self.pos_mtx, pos_dim % 2)
        self.y_pred = self.build_prm_fc_function(item_seq, mode)

        self.build_predict(self.y_pred, mode)

        return self.y_pred

    def add_pos_embedding(self, item_emb, pos_emb, pad_flag):
        item_emb = item_emb + pos_emb
        if pad_flag:
            item_emb = tf.pad(item_emb, [[0, 0], [0, 0], [0, 1]])
        return item_emb

    def build_loss(self, y_pred, y_label, mode):
        loss = self.build_logloss(y_pred, y_label)
        return loss

    def build_prm_fc_function(self, inp, mode):
        with tf.variable_scope("build_prm_fc_function", reuse=tf.AUTO_REUSE):
            inp = self.multihead_attention(inp, inp, num_units=self.d_model,
                                           num_heads=self.n_head)
            inp = self.positionwise_feed_forward(inp, self.d_model, self.d_inner_hid, mode, self.keep_prob)

            mask = tf.expand_dims(tf.sequence_mask(self.seq_length_ph, maxlen=self.max_time_len, dtype=tf.float32),
                                  axis=-1)
            inp = inp * mask

            bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1', training=mode)
            fc1 = tf.layers.dense(bn1, self.d_model, activation=tf.nn.relu, name='fc1')
            dp1 = tf.nn.dropout(fc1, self.keep_prob, name='dp1')
            fc2 = tf.layers.dense(dp1, 1, activation=None, name='fc2')
            score = tf.nn.softmax(tf.reshape(fc2, [-1, self.max_time_len]))
            # output
            seq_mask = tf.sequence_mask(self.seq_length_ph, maxlen=self.max_time_len, dtype=tf.float32)
        return seq_mask * score

    def build_predict(self, pred, mode):
        # principal 1: 基于B生成C
        y_pred_pos = self.position_from_score(pred)
        pos_mtx = tf.gather(self.pos_mtx, y_pred_pos)
        y_pred_02 = self.build_prm_fc_function(self.add_pos_embedding(self.item_seq, pos_mtx, self.pos_dim % 2), mode)

        # primcipal 2: 随机乱序，生成C
        y_pred_pos = self.position_adjust(pred)

        rand_pos_mtx = tf.gather(self.pos_mtx, y_pred_pos)
        y_pred_03 = self.build_prm_fc_function(
            self.add_pos_embedding(self.item_seq, rand_pos_mtx, self.pos_dim % 2), mode)

        self.y_pred_B2 = y_pred_02
        self.y_pred_B3 = y_pred_03


class PEAR(BaseModel):
    """
    https://arxiv.org/pdf/2203.12267.pdf

        网络结构设置
        1. concat user feature and item feature 得到X，X对应self.item_seq

        feature-level interaction
        2. 两层的DNN 做特征 confuse

        etc
        1. add a special classification token (i.e., CLS) with learnable parameters at the end of the initial ranking list


        item-level interation

        在历史序列上做cross，在候选序列上也做cross，最后做历史和候选之间的cross

        unit: each comprising a self-attention layer and a merged crossattention layer


        实验设置
        优化器: Adam
        序列特征长度：128
        feature-level interaction MLP: 500, 500
        dropout rate: 0.1
        batch size: 200 (MicroVideo-1.7M), 400 (News-Dataset)
        learning rate: {1e-3, 1e-4, 5e-5, 1e-5}
        heads: 1 (MicroVideo-1.7M), 2 (News-Dataset)
        hidden dimensionality: 500

    """

    def __init__(self, eb_dim, hidden_size, max_time_len, keep_prob, reg_lambda, lr,
                 profile_num=8, itm_spar_num=5, itm_dens_num=1, max_norm=None, d_model=500,
                 d_inner_hid=128, n_head=1, cross_n_head=1, use_hist_seq=False):
        self.d_model = d_model
        self.d_inner_hid = d_inner_hid
        self.n_head = n_head
        self.cross_n_head = cross_n_head
        self.num_units = hidden_size
        self.use_hist_seq = use_hist_seq
        super(PEAR, self).__init__(max_time_len, hidden_size, eb_dim, itm_spar_num, itm_dens_num, profile_num, max_norm,
                                   keep_prob, reg_lambda, lr)

    def process_embedding(self, features):
        # embedding 初始化
        embeddingLayer = DenseEmbeddingsLayer(variable_name="emb_mtx", embedding_dim=self.emb_dim)
        self.itm_spar_emb = embeddingLayer(features['item_sparse'])
        self.usr_spar_emb = embeddingLayer(features['user_sparse'])
        self.itm_dens_ph = tf.reshape(features['item_dense'], (-1, self.max_time_len, self.itm_dens_num))

        self.item_seq = tf.concat(
            [tf.reshape(self.itm_spar_emb, [-1, self.max_time_len, int(self.itm_spar_num * self.emb_dim)]),
             self.itm_dens_ph], axis=-1)

        fc = self.item_seq
        for idx, w in enumerate([500, 500]):
            fc = tf.layers.dense(fc, w, activation=tf.nn.relu, name='fc' + str(idx))

        self.item_seq = fc

        self.seq_length_ph = tf.reshape(features['list_len'], (tf.shape(features['list_len'])[0],))

        if self.use_hist_seq:
            self.usr_hist_emb = embeddingLayer(features['hist_seq'])

        cls_emb = tf.get_variable(shape=(1, 1, self.item_seq.shape[-1].value), dtype=tf.float32, name='cls_emb')
        self.cls_emb = tf.tile(cls_emb, (tf.shape(self.item_seq)[0], 1, 1))

        return self.item_seq

    def call(self, features, mode):
        self.process_embedding(features)

        self.pos_dim = pos_dim = self.item_seq.get_shape().as_list()[-1]
        self.pos_mtx = tf.get_variable("pos_mtx", [self.max_time_len, pos_dim],
                                       initializer=tf.truncated_normal_initializer)
        item_seq = self.item_seq + self.pos_mtx
        self.y_pred_full, self.y_pred = self.build_pear_function(item_seq, mode)

        self.build_predict(self.y_pred, mode)
        return self.y_pred

    def build_predict(self, pred, mode):
        # principal 1: 基于B生成C
        y_pred_pos = self.position_from_score(pred)
        pos_mtx = tf.gather(self.pos_mtx, y_pred_pos)
        _, y_pred_02 = self.build_pear_function(self.add_pos_embedding(self.item_seq, pos_mtx, False), mode)

        # primcipal 2: 随机乱序，生成C
        y_pred_pos = self.position_adjust(pred)

        rand_pos_mtx = tf.gather(self.pos_mtx, y_pred_pos)
        _, y_pred_03 = self.build_pear_function(self.add_pos_embedding(self.item_seq, rand_pos_mtx, False), mode)

        self.y_pred_B2 = y_pred_02
        self.y_pred_B3 = y_pred_03

    def build_pear_function(self, inp, mode):
        with tf.variable_scope('pear', reuse=tf.AUTO_REUSE):
            inp = tf.concat([inp, self.cls_emb], axis=1)
            tgt_attn_emb = self.multihead_attention(inp, inp, num_units=self.d_model,
                                                    num_heads=self.n_head)

            logger.info(f'tgt_attn_emb {tgt_attn_emb}')
            if self.use_hist_seq:
                usr_hist_emb = self.multihead_attention(self.usr_hist_emb, self.usr_hist_emb, num_units=self.d_model,
                                                        num_heads=self.n_head)
                crs_attn_emb = self.merged_cross_attention(self.usr_hist_emb, inp, num_units=self.d_model,
                                                           num_heads=self.cross_n_head)
                hidden = tf.concat([usr_hist_emb, crs_attn_emb, tgt_attn_emb], axis=-1)
            else:
                hidden = tgt_attn_emb
            y_pred_full = tf.layers.dense(hidden, 1, activation=tf.nn.sigmoid, name='output')

            y_pred_full = tf.reshape(y_pred_full, [-1, self.max_time_len + 1])

            y_pred = y_pred_full[:, :self.max_time_len]

        return y_pred_full, y_pred

    def add_pos_embedding(self, item_emb, pos_emb, pad_flag):
        item_emb = item_emb + pos_emb
        if pad_flag:
            item_emb = tf.pad(item_emb, [[0, 0], [0, 0], [0, 1]])
        return item_emb

    def build_loss(self, y_pred, y_label, mode):
        logger.info(f'y_pred {y_pred}, y_label {y_label}')

        # 增加辅助loss， 即当前trace是否有点击
        y_label = tf.concat([y_label, tf.reduce_max(y_label, axis=-1, keepdims=True)], axis=-1)
        loss = self.build_logloss(self.y_pred_full, y_label)
        return loss

    def merged_cross_attention(self,
                               queries,
                               keys,
                               num_units=None,
                               num_heads=2,
                               scope="multihead_attention",
                               reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            if num_units is None:
                num_units = queries.get_shape().as_list()[-1]

            Q = tf.layers.dense(queries, num_units, activation=None)  # (N, T_q, C)

            K1 = tf.layers.dense(queries, num_units, activation=None)  # (N, T_q, C)
            K2 = tf.layers.dense(keys, num_units, activation=None)  # (N, T_q, C)

            K = tf.concat([K1, K2], axis=1)

            V1 = tf.layers.dense(queries, num_units, activation=None)  # (N, T_k, C)
            V2 = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)

            V = tf.concat([V1, V2], axis=1)

            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
            key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)
            outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

            query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
            query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
            outputs *= query_masks  # broadcasting. (N, T_q, C)
            outputs = tf.nn.dropout(outputs, self.keep_prob)
            outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        return outputs


class SetRank(BaseModel):
    def __init__(self, eb_dim, hidden_size, max_time_len, keep_prob, reg_lambda, lr, profile_num=8, itm_spar_num=5,
                 itm_dens_num=1, max_norm=None, d_model=256, d_inner_hid=64, n_head=8):
        self.d_model = d_model
        self.d_inner_hid = d_inner_hid
        self.n_head = n_head
        super().__init__(max_time_len, hidden_size, eb_dim, itm_spar_num, itm_dens_num, profile_num, max_norm,
                         keep_prob, reg_lambda, lr)

    def call(self, features, mode):
        self.process_embedding(features)
        self.pos_dim = pos_dim = self.item_seq.get_shape().as_list()[-1]
        self.pos_mtx = tf.get_variable("pos_mtx", [self.max_time_len, pos_dim],
                                       initializer=tf.truncated_normal_initializer)
        item_seq = self.item_seq + self.pos_mtx
        self.y_pred = self.build_setrank_function(item_seq, mode)
        self.build_predict(self.y_pred, mode)
        return self.y_pred

    def add_pos_embedding(self, inp, pe):
        return inp + pe

    def build_setrank_function(self, inp, mode):
        with tf.variable_scope('setrank', reuse=tf.AUTO_REUSE):
            inp = self.multihead_attention(inp, inp, num_units=self.d_model, num_heads=self.n_head)

            inp = self.positionwise_feed_forward(inp, self.d_model, self.d_inner_hid, mode, dropout=self.keep_prob)

            mask = tf.expand_dims(tf.sequence_mask(self.seq_length_ph, maxlen=self.max_time_len, dtype=tf.float32),
                                  axis=-1)
            seq_rep = inp * mask

            y_pred = self.build_fc_net(seq_rep, mode)

        return y_pred

    def build_loss(self, y_pred, y_label, mode):
        loss = self.build_attention_loss(y_pred, y_label)
        return loss

    def build_predict(self, pred, mode):
        # principal 1: 基于B生成C
        y_pred_pos = self.position_from_score(pred)
        pos_mtx = tf.gather(self.pos_mtx, y_pred_pos)
        y_pred_02 = self.build_setrank_function(self.add_pos_embedding(self.item_seq, pos_mtx), mode)

        # primcipal 2: 随机乱序，生成C
        y_pred_pos = self.position_adjust(pred)

        rand_pos_mtx = tf.gather(self.pos_mtx, y_pred_pos)
        y_pred_03 = self.build_setrank_function(self.add_pos_embedding(self.item_seq, rand_pos_mtx), mode)

        self.y_pred_B2 = y_pred_02
        self.y_pred_B3 = y_pred_03


class DLCM(BaseModel):
    def __init__(self, eb_dim, hidden_size, max_time_len, keep_prob, reg_lambda, lr, profile_num=8, itm_spar_num=5,
                 itm_dens_num=1, max_norm=None, d_model=256, d_inner_hid=64, n_head=8):
        self.d_model = d_model
        self.d_inner_hid = d_inner_hid
        self.n_head = n_head
        super().__init__(max_time_len, hidden_size, eb_dim, itm_spar_num, itm_dens_num, profile_num, max_norm,
                         keep_prob, reg_lambda, lr)

    def call(self, features, mode):
        self.process_embedding(features)
        self.pos_dim = pos_dim = self.item_seq.get_shape().as_list()[-1]
        self.pos_mtx = tf.get_variable("pos_mtx", [self.max_time_len, pos_dim],
                                       initializer=tf.truncated_normal_initializer)
        item_seq = self.item_seq + self.pos_mtx

        self.y_pred = self.build_dlcm_function(item_seq, mode)
        self.build_predict(self.y_pred, mode)
        return self.y_pred

    def add_pos_embedding(self, inp, pe):
        return inp + pe

    def build_dlcm_function(self, inp, mode):
        with tf.variable_scope('gru', reuse=tf.AUTO_REUSE):
            seq_ht, seq_final_state = tf.nn.dynamic_rnn(GRUCell(self.hidden_size), inputs=inp,
                                                        sequence_length=self.seq_length_ph, dtype=tf.float32,
                                                        scope='gru1')

            y_pred = self.build_phi_function(seq_ht, seq_final_state, self.hidden_size, mode)

        return y_pred

    def build_loss(self, y_pred, y_label, mode):
        loss = self.build_attention_loss(y_pred, y_label)
        return loss

    def build_phi_function(self, seq_ht, seq_final_state, hidden_size, mode):
        bn1 = tf.layers.batch_normalization(inputs=seq_final_state, name='bn1', training=mode)
        seq_final_fc = tf.layers.dense(bn1, hidden_size, activation=tf.nn.tanh, name='fc1')
        dp1 = tf.nn.dropout(seq_final_fc, self.keep_prob, name='dp1')
        seq_final_fc = tf.expand_dims(dp1, axis=1)
        bn2 = tf.layers.batch_normalization(inputs=seq_ht, name='bn2', training=mode)
        # fc2 = tf.layers.dense(tf.multiply(bn2, seq_final_fc), 2, activation=None, name='fc2')
        # score = tf.nn.softmax(fc2)
        # score = tf.reshape(score[:, :, 0], [-1, self.max_time_len])
        fc2 = tf.layers.dense(tf.multiply(bn2, seq_final_fc), 1, activation=None, name='fc2')
        score = tf.reshape(fc2, [-1, self.max_time_len])
        # sequence mask
        seq_mask = tf.sequence_mask(self.seq_length_ph, maxlen=self.max_time_len, dtype=tf.float32)
        score = score * seq_mask
        score = score - tf.reduce_min(score, 1, keep_dims=True)
        return score

    def build_predict(self, pred, mode):
        # principal 1: 基于B生成C
        y_pred_pos = self.position_from_score(pred)
        pos_mtx = tf.gather(self.pos_mtx, y_pred_pos)
        y_pred_02 = self.build_dlcm_function(self.add_pos_embedding(self.item_seq, pos_mtx), mode)

        # primcipal 2: 随机乱序，生成C
        y_pred_pos = self.position_adjust(pred)

        rand_pos_mtx = tf.gather(self.pos_mtx, y_pred_pos)
        y_pred_03 = self.build_dlcm_function(self.add_pos_embedding(self.item_seq, rand_pos_mtx), mode)

        self.y_pred_B2 = y_pred_02
        self.y_pred_B3 = y_pred_03


class GSF(BaseModel):
    def __init__(self, eb_dim, hidden_size, max_time_len, keep_prob, reg_lambda, lr, profile_num=8, itm_spar_num=5,
                 itm_dens_num=1, max_norm=None, group_size=1, activation='relu',
                 hidden_layer_size=(512, 256, 128)):
        self.group_size = group_size
        self.activation = tf.nn.elu if activation == 'elu' else tf.nn.relu
        self.hidden_layer_size = list(hidden_layer_size)
        super().__init__(max_time_len, hidden_size, eb_dim, itm_spar_num, itm_dens_num, profile_num, max_norm,
                         keep_prob, reg_lambda, lr)

    def call(self, features, mode):
        self.process_embedding(features)

        self.pos_dim = pos_dim = self.item_seq.get_shape().as_list()[-1]
        self.pos_mtx = tf.get_variable("pos_mtx", [self.max_time_len, pos_dim],
                                       initializer=tf.truncated_normal_initializer)
        item_seq = self.item_seq + self.pos_mtx
        self.y_pred = self.build_gsf_function(item_seq, mode)

        self.build_predict(self.y_pred, mode)
        return self.y_pred

    def add_pos_embedding(self, inp, pe):
        return inp + pe

    def build_gsf_function(self, inp, mode):
        with tf.variable_scope('gsf', reuse=tf.AUTO_REUSE):
            input_list = tf.unstack(inp, axis=1)
            input_data = tf.concat(input_list, axis=0)
            output_data = input_data

            input_data_list = tf.split(output_data, self.max_time_len, axis=0)
            output_sizes = self.hidden_layer_size + [self.group_size]

            output_data_list = [0 for _ in range(self.max_time_len)]
            group_list = []
            self.get_possible_group([], group_list)
            for group in group_list:
                group_input = tf.concat([input_data_list[idx]
                                         for idx in group], axis=1)
                group_score_list = self.build_gsf_fc_function(group_input, output_sizes, self.activation, mode)
                for i in range(self.group_size):
                    output_data_list[group[i]] += group_score_list[i]
            y_pred = tf.concat(output_data_list, axis=1)
            y_pred = tf.nn.softmax(y_pred, axis=-1)
        return y_pred

    def build_loss(self, y_pred, y_label, mode):
        loss = self.build_norm_logloss(y_pred, y_label)
        return loss

    def build_gsf_fc_function(self, inp, hidden_size, activation, mode, scope="gsf_nn"):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            for j in range(len(hidden_size)):
                bn = tf.layers.batch_normalization(inputs=inp, name='bn' + str(j), training=mode)
                if j != len(hidden_size) - 1:
                    inp = tf.layers.dense(bn, hidden_size[j], activation=activation, name='fc' + str(j))
                else:
                    inp = tf.layers.dense(bn, hidden_size[j], activation=tf.nn.sigmoid, name='fc' + str(j))
        return tf.split(inp, self.group_size, axis=1)

    def get_possible_group(self, group, group_list):
        if len(group) == self.group_size:
            group_list.append(group)
            return
        else:
            for i in range(self.max_time_len):
                self.get_possible_group(group + [i], group_list)

    def build_predict(self, pred, mode):
        # principal 1: 基于B生成C
        y_pred_pos = self.position_from_score(pred)
        pos_mtx = tf.gather(self.pos_mtx, y_pred_pos)

        y_pred_02 = self.build_gsf_function(self.add_pos_embedding(self.item_seq, pos_mtx), mode)
        # primcipal 2: 随机乱序，生成C
        y_pred_pos = self.position_adjust(pred)

        rand_pos_mtx = tf.gather(self.pos_mtx, y_pred_pos)

        y_pred_03 = self.build_gsf_function(self.add_pos_embedding(self.item_seq, rand_pos_mtx), mode)

        self.y_pred_B2 = y_pred_02
        self.y_pred_B3 = y_pred_03


class miDNN(BaseModel):
    def __init__(self, eb_dim, hidden_size, max_time_len, keep_prob, reg_lambda, lr, profile_num=8, itm_spar_num=5,
                 itm_dens_num=1, max_norm=None, hidden_layer_size=(512, 256, 128)):
        self.hidden_layer_size = list(hidden_layer_size)
        super().__init__(max_time_len, hidden_size, eb_dim, itm_spar_num, itm_dens_num, profile_num, max_norm,
                         keep_prob, reg_lambda, lr)

    def build_loss(self, y_pred, y_label, mode):
        loss = self.build_logloss(y_pred, y_label)
        return loss

    def call(self, features, mode):
        self.process_embedding(features)
        self.pos_dim = pos_dim = self.item_seq.get_shape().as_list()[-1]
        self.pos_mtx = tf.get_variable("pos_mtx", [self.max_time_len, pos_dim],
                                       initializer=tf.truncated_normal_initializer)
        item_seq = self.item_seq + self.pos_mtx
        self.y_pred = self.build_miDNN_function(item_seq, mode)

        self.build_predict(self.y_pred, mode)
        return self.y_pred

    def build_miDNN_function(self, inp, mode):
        fmax = tf.reduce_max(tf.reshape(inp, [-1, self.max_time_len, self.ft_num]), axis=1,
                             keep_dims=True)
        fmin = tf.reduce_min(tf.reshape(inp, [-1, self.max_time_len, self.ft_num]), axis=1,
                             keep_dims=True)
        global_seq = (inp - fmin) / (fmax - fmin + 1e-8)
        inp = tf.concat([inp, global_seq], axis=-1)

        y_pred = self.build_miDNN_net(inp, self.hidden_layer_size, mode)

        return y_pred

    def add_pos_embedding(self, inp, pe):
        return inp + pe

    def build_miDNN_net(self, inp, layer, mode, scope='mlp'):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            inp = tf.layers.batch_normalization(inputs=inp, name='mlp_bn', training=mode)
            for i, hidden_num in enumerate(layer):
                fc = tf.layers.dense(inp, hidden_num, activation=tf.nn.relu, name='fc' + str(i))
                inp = tf.nn.dropout(fc, self.keep_prob, name='dp' + str(i))
            final = tf.layers.dense(inp, 1, activation=tf.nn.sigmoid, name='fc_final')
            score = tf.reshape(final, [-1, self.max_time_len])
            seq_mask = tf.sequence_mask(self.seq_length_ph, maxlen=self.max_time_len, dtype=tf.float32)
            y_pred = seq_mask * score
        return y_pred

    def build_predict(self, pred, mode):
        # principal 1: 基于B生成C
        y_pred_pos = self.position_from_score(pred)
        pos_mtx = tf.gather(self.pos_mtx, y_pred_pos)

        y_pred_02 = self.build_miDNN_function(self.add_pos_embedding(self.item_seq, pos_mtx), mode)
        # primcipal 2: 随机乱序，生成C
        y_pred_pos = self.position_adjust(pred)

        rand_pos_mtx = tf.gather(self.pos_mtx, y_pred_pos)
        y_pred_03 = self.build_miDNN_function(self.add_pos_embedding(self.item_seq, rand_pos_mtx), mode)
        self.y_pred_B2 = y_pred_02
        self.y_pred_B3 = y_pred_03


class RAISE(BaseModel):
    """
    settings:
    1. d=32
    2. n=50
    3. MLP 实验了1到4层的效果
    4. 评论使用条数为20，并使用bert作为nlp预训练的语料
    5. 学习率 1e-1, 1e-2, 1e-3, 1e-4
    6. batchsize 256, 512, 1024
    7. dropout rate [0.1..0.5]
    8. transform_matrices t in {1,2,4,8,10}
    9. DTE blocks in {1,2,3,5,8,10}
    """

    def __init__(self, eb_dim, hidden_size, max_time_len, keep_prob, reg_lambda, lr,
                 profile_num=8, itm_spar_num=5, itm_dens_num=1, max_norm=None, d_model=32,
                 d_inner_hid=64, a_head=1):
        self.d_model = d_model
        self.d_inner_hid = d_inner_hid
        self.a_head = a_head
        super(RAISE, self).__init__(max_time_len, hidden_size, eb_dim, itm_spar_num, itm_dens_num, profile_num,
                                    max_norm,
                                    keep_prob, reg_lambda, lr)

    def process_embedding(self, features):
        # embedding 初始化
        itemEmbeddingLayer = DenseEmbeddingsLayer(variable_name="item_emb_mtx", embedding_dim=self.emb_dim)
        userEmbeddingLayer = DenseEmbeddingsLayer(variable_name="user_emb_mtx", embedding_dim=self.emb_dim)
        self.itm_spar_emb = itemEmbeddingLayer(features['item_sparse'])
        self.itm_dens_ph = tf.reshape(features['item_dense'], [-1, self.max_time_len, self.itm_dens_num])
        self.seq_length_ph = tf.reshape(features['list_len'], (tf.shape(features['list_len'])[0],))
        item_seq = tf.concat(
            [tf.reshape(self.itm_spar_emb, [-1, self.max_time_len, int(self.itm_spar_num * self.emb_dim)]),
             self.itm_dens_ph], axis=-1)

        self.usr_spar_emb = userEmbeddingLayer(features['user_sparse'])
        user_feas = tf.reshape(self.usr_spar_emb, (-1, 1, int(self.profile_num * self.emb_dim)))
        logger.info(f'item_seq {item_seq}, user_feas {user_feas}')
        # item_seq = tf.concat([item_seq, user_feas], axis=-1)
        # MF 得到p_u，q_i
        # p_u bs.1.es
        user_emb = tf.layers.dense(user_feas, units=self.d_model)
        # q_i bs.ls.es
        item_emb = tf.layers.dense(item_seq, units=self.d_model)

        # bs.1.ls
        # NOTE 没做预训练
        # mf_predict = tf.matmul(user_emb, item_emb, transpose_b=True)

        # 目前的样本无item/user 粒度的反馈明细数据
        # NOTE skip
        # user_review, item_review = self.calc_intention_aware_representaion(features)
        user_review, item_review = None, None

        logger.info(f'user_emb, item_emb {user_emb}, {item_emb}')
        self.item_seq = self.calc_intention_aware_sequential_representaion(user_emb, item_emb, user_review, item_review)
        self.dynammic_attention = self.calc_dynamic_attention(user_emb, item_emb, user_review, item_review)

        return self.item_seq

    def calc_dynamic_attention(self, user_emb, item_emb, user_review, item_review):
        logger.info(f'calc_dynamic_attention input {user_emb}, {item_emb}')
        p_su = tf.reshape(user_emb, (-1, self.d_model)) + tf.reduce_sum(item_emb, axis=1) / self.max_time_len
        logger.info(f'calc_dynamic_attention input {p_su}')
        if user_review is not None:
            q_su = tf.reduce_sum(user_review + item_review, axis=1) / self.max_time_len
            p = tf.concat([p_su, q_su], axis=-1)
        else:
            p = p_su

        dynammic_attention = tf.layers.dense(p, units=self.a_head,
                                             activation=tf.nn.relu)
        dynammic_attention = tf.layers.dense(dynammic_attention, units=self.a_head)
        # bs.x
        dynammic_attention = tf.nn.softmax(dynammic_attention)

        logger.info(f'calc_dynamic_attention {dynammic_attention}')
        # x.d.d
        W_q = tf.get_variable('W_q', shape=(self.a_head, self.d_model, self.d_model))
        W_k = tf.get_variable('W_k', shape=(self.a_head, self.d_model, self.d_model))
        W_v = tf.get_variable('W_v', shape=(self.a_head, self.d_model, self.d_model))

        logger.info(f'calc_dynamic_attention {W_q}, {W_k}, {W_v}')
        # bs.d.d
        W_q = tf.einsum('ab,bcd->acd', dynammic_attention, W_q)
        W_k = tf.einsum('ab,bcd->acd', dynammic_attention, W_k)
        W_v = tf.einsum('ab,bcd->acd', dynammic_attention, W_v)

        self.W = [W_q, W_k, W_v]

        logger.info(f'calc_intention_aware_sequential_representaion {dynammic_attention}, {W_q}, {W_k}, {W_v}')
        return dynammic_attention

    def calc_intention_aware_sequential_representaion(self, user_emb, item_emb, user_review, item_review):
        s_im = tf.layers.dense(tf.concat([tf.tile(user_emb, (1, self.max_time_len, 1)), item_emb], axis=-1),
                               units=self.hidden_size)
        if user_review is not None:
            s_re = tf.layers.dense(tf.concat([user_review, item_review], axis=-1), units=self.hidden_size)

            s = tf.concat([s_im, s_re], axis=-1)
        else:
            s = s_im

        s = tf.layers.dense(s, units=self.d_model)
        logger.info(f'calc_intention_aware_sequential_representaion output {s}')
        return s

    def dynamic_transformer_units(self,
                                  queries,
                                  keys,
                                  num_units=None,
                                  num_heads=1,
                                  scope="multihead_attention",
                                  reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            # QKV bs.ls.d
            # W bs.d.d

            Q = tf.matmul(queries, self.W[0])  # (N, T_q, C)
            K = tf.matmul(keys, self.W[0])  # (N, T_k, C)
            V = tf.matmul(queries, self.W[0])  # (N, T_k, C)

            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
            key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)
            outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

            query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
            query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
            outputs *= query_masks  # broadcasting. (N, T_q, C)
            outputs = tf.nn.dropout(outputs, self.keep_prob)
            outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        return outputs

    def build_raise_fc_function(self, inp, mode):
        with tf.variable_scope("build_raise_fc_function", reuse=tf.AUTO_REUSE):
            inp = self.multihead_attention(inp, inp, num_units=self.d_model,
                                           num_heads=1)
            inp = self.positionwise_feed_forward(inp, self.d_model, self.d_inner_hid, mode, self.keep_prob)

            mask = tf.expand_dims(tf.sequence_mask(self.seq_length_ph, maxlen=self.max_time_len, dtype=tf.float32),
                                  axis=-1)
            inp = inp * mask

            bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1', training=mode)
            fc1 = tf.layers.dense(bn1, self.d_model, activation=tf.nn.relu, name='fc1')
            dp1 = tf.nn.dropout(fc1, self.keep_prob, name='dp1')
            fc2 = tf.layers.dense(dp1, 1, activation=None, name='fc2')
            score = tf.nn.softmax(tf.reshape(fc2, [-1, self.max_time_len]))
            # output
            seq_mask = tf.sequence_mask(self.seq_length_ph, maxlen=self.max_time_len, dtype=tf.float32)
        return seq_mask * score

    def calc_intention_aware_representaion(self, features):
        """
        # Intention-aware Review Representation
        # 这一步的设置计算用户的评价和item的评价之间的相关性

        :param features:
        :return:
            user_review bs.sl.es
            item_review bs.sl.es
        """
        itemREmbeddingLayer = DenseEmbeddingsLayer(variable_name="item_review_emb_mtx", embedding_dim=self.emb_dim)
        userREmbeddingLayer = DenseEmbeddingsLayer(variable_name="user_review_emb_mtx", embedding_dim=self.emb_dim)
        transformRMatrix = tf.get_variable(name='transform_r', shape=(self.emb_dim, self.emb_dim))
        # bs.m.es
        user_review = userREmbeddingLayer(features['user_review'])
        # bs.n.es
        item_review = itemREmbeddingLayer(features['item_review'])

        # flatten time len to batch size
        m = tf.shape(user_review)[1]
        n = tf.shape(item_review)[1]

        item_review = tf.reshape(item_review, (-1, n, self.emb_dim))
        from tensorflow.contrib.seq2seq import tile_batch
        user_review = tile_batch(user_review, self.max_time_len)

        AB = tf.einsum('abc,cd->abd', user_review, transformRMatrix)
        C = tf.einsum('abd,acd->abc', AB, item_review)

        # 输出shape 为bs.m.n

        # bs.m.n bs.m.es -> bs.n.m bs.es.m -> bs.n.es/m
        user_review = tf.matmul(C, user_review, transpose_a=True, transpose_b=True) / item_review.shape[1].value

        # bs.m.n bs.n.es -> bs.m.es
        item_review = tf.matmul(C, item_review, transpose_b=True) / user_review.shape[1].value

        user_review = tf.reduce_sum(tf.reshape(user_review, (-1, self.max_time_len, n, self.emb_dim)), axis=-2)
        item_review = tf.reduce_sum(tf.reshape(item_review, (-1, self.max_time_len, m, self.emb_dim)), axis=-2)

        return user_review, item_review

    def call(self, features, mode):
        self.process_embedding(features)

        self.pos_dim = pos_dim = self.item_seq.get_shape().as_list()[-1]
        self.pos_mtx = tf.get_variable("pos_mtx", [self.max_time_len, pos_dim],
                                       initializer=tf.truncated_normal_initializer)
        item_seq = self.add_pos_embedding(self.item_seq, self.pos_mtx, False)
        self.y_pred = self.build_raise_fc_function(item_seq, mode)

        self.build_predict(self.y_pred, mode)
        return self.y_pred

    def build_predict(self, pred, mode):
        # principal 1: 基于B生成C
        y_pred_pos = self.position_from_score(pred)
        pos_mtx = tf.gather(self.pos_mtx, y_pred_pos)
        y_pred_02 = self.build_raise_fc_function(self.add_pos_embedding(self.item_seq, pos_mtx, False), mode)

        # primcipal 2: 随机乱序，生成C
        y_pred_pos = self.position_adjust(pred)

        rand_pos_mtx = tf.gather(self.pos_mtx, y_pred_pos)
        y_pred_03 = self.build_raise_fc_function(self.add_pos_embedding(self.item_seq, rand_pos_mtx, False), mode)

        self.y_pred_B2 = y_pred_02
        self.y_pred_B3 = y_pred_03

    def add_pos_embedding(self, item_emb, pos_emb, pad_flag):
        item_emb = item_emb + pos_emb
        if pad_flag:
            item_emb = tf.pad(item_emb, [[0, 0], [0, 0], [0, 1]])
        return item_emb

    def build_loss(self, y_pred, y_label, mode):
        loss = self.build_logloss(y_pred, y_label)
        return loss


class PRM_O1(PRM):
    def __init__(self, eb_dim, hidden_size, max_time_len, keep_prob, reg_lambda, lr, profile_num=8, itm_spar_num=5,
                 itm_dens_num=1, max_norm=None, d_model=64, d_inner_hid=128, n_head=1):
        super().__init__(eb_dim, hidden_size, max_time_len, keep_prob, reg_lambda, lr, profile_num, itm_spar_num,
                         itm_dens_num, max_norm, d_model, d_inner_hid, n_head)

    def build_loss(self, y_pred, y_label, mode):
        # step 1
        y_pred_01 = y_pred

        y_pred_pos = self.position_from_score(y_pred_01)
        pos_mtx = tf.gather(self.pos_mtx, y_pred_pos)

        y_pred_02 = self.build_prm_fc_function(self.add_pos_embedding(self.item_seq, pos_mtx, self.pos_dim % 2), mode)

        loss_01 = tf.losses.log_loss(y_label, y_pred_01)
        loss_02 = tf.losses.log_loss(y_label, y_pred_02)

        # 约束rank和score接近
        loss_mse = self.contrastive_loss(y_pred_01, y_pred_02)

        self.loss = loss_02 + loss_01 + loss_mse

        return self.loss


class PRM_O2(PRM_O1):
    """
    shuffling of the rerank input does not change the output result
    """

    def __init__(self, eb_dim, hidden_size, max_time_len, keep_prob, reg_lambda, lr, profile_num=8, itm_spar_num=5,
                 itm_dens_num=1, max_norm=None, d_model=64, d_inner_hid=128, n_head=1):
        super().__init__(eb_dim, hidden_size, max_time_len, keep_prob, reg_lambda, lr, profile_num, itm_spar_num,
                         itm_dens_num, max_norm, d_model, d_inner_hid, n_head)

    def build_loss(self, y_pred, y_label, mode):
        # step 1
        y_pred_01 = y_pred

        y_pred_pos = self.position_adjust_from_score(y_pred_01)

        rand_pos_mtx = tf.gather(self.pos_mtx, y_pred_pos)
        y_pred_03 = self.build_prm_fc_function(
            self.add_pos_embedding(self.item_seq, rand_pos_mtx, self.pos_dim % 2), mode)

        loss_01 = tf.losses.log_loss(y_label, y_pred_01)
        loss_02 = tf.losses.log_loss(y_label, y_pred_03)

        # 约束rank和score接近
        loss_mse = self.contrastive_loss(y_pred_01, y_pred_03)

        self.loss = loss_02 + loss_01 + loss_mse

        return self.loss


class PRM_O6(PRM_O1):
    """
    001 & 002
    """

    def __init__(self, eb_dim, hidden_size, max_time_len, keep_prob, reg_lambda, lr, profile_num=8, itm_spar_num=5,
                 itm_dens_num=1, max_norm=None, d_model=64, d_inner_hid=128, n_head=1):
        super().__init__(eb_dim, hidden_size, max_time_len, keep_prob, reg_lambda, lr, profile_num, itm_spar_num,
                         itm_dens_num, max_norm, d_model, d_inner_hid, n_head)

    def build_loss(self, y_pred, y_label, mode):
        # step 1
        y_pred_01 = y_pred

        y_pred_pos = self.position_from_score(y_pred_01)
        pos_mtx = tf.gather(self.pos_mtx, y_pred_pos)

        y_pred_02 = self.build_prm_fc_function(self.add_pos_embedding(self.item_seq, pos_mtx, self.pos_dim % 2), mode)

        y_pred_pos = self.position_adjust_from_score(y_pred_01)

        rand_pos_mtx = tf.gather(self.pos_mtx, y_pred_pos)
        # pos_mtx = tf.gather(self.pos_mtx, rand_pos)
        y_pred_03 = self.build_prm_fc_function(
            self.add_pos_embedding(self.item_seq, rand_pos_mtx, self.pos_dim % 2), mode)

        loss_01 = tf.losses.log_loss(y_label, y_pred_01)
        loss_02 = tf.losses.log_loss(y_label, y_pred_02)
        loss_03 = tf.losses.log_loss(y_label, y_pred_03)

        # 约束rank和score接近
        loss_mse_12 = self.contrastive_loss(y_pred_01, y_pred_02)
        loss_mse_13 = self.contrastive_loss(y_pred_01, y_pred_03)
        loss_mse_23 = self.contrastive_loss(y_pred_02, y_pred_03)

        self.loss = loss_02 + loss_01 + loss_03 + loss_mse_12 + loss_mse_13 + loss_mse_23

        return self.loss


class SetRank_O6(SetRank):
    def __init__(self, eb_dim, hidden_size, max_time_len, keep_prob, reg_lambda, lr, profile_num=8, itm_spar_num=5,
                 itm_dens_num=1, max_norm=None, d_model=256, d_inner_hid=64, n_head=8):
        super().__init__(eb_dim, hidden_size, max_time_len, keep_prob, reg_lambda, lr, profile_num, itm_spar_num,
                         itm_dens_num, max_norm,
                         d_model, d_inner_hid, n_head)

    def build_loss(self, y_pred, y_label, mode):
        y_pred_01 = y_pred

        y_pred_pos = self.position_from_score(y_pred_01)
        pos_mtx = tf.gather(self.pos_mtx, y_pred_pos)

        y_pred_02 = self.build_setrank_function(self.add_pos_embedding(self.item_seq, pos_mtx), mode)

        y_pred_pos = self.position_adjust_from_score(y_pred_01)

        rand_pos_mtx = tf.gather(self.pos_mtx, y_pred_pos)
        # pos_mtx = tf.gather(self.pos_mtx, rand_pos)
        y_pred_03 = self.build_setrank_function(self.add_pos_embedding(self.item_seq, rand_pos_mtx), mode)

        loss_01 = self.build_attention_loss(y_pred_01, y_label)
        loss_02 = self.build_attention_loss(y_pred_02, y_label)
        loss_03 = self.build_attention_loss(y_pred_03, y_label)

        # 约束rank和score接近
        loss_mse_12 = self.contrastive_loss(y_pred_01, y_pred_02)
        loss_mse_13 = self.contrastive_loss(y_pred_01, y_pred_03)
        loss_mse_23 = self.contrastive_loss(y_pred_02, y_pred_03)

        self.loss = loss_02 + loss_01 + loss_03 + loss_mse_12 + loss_mse_13 + loss_mse_23

        return self.loss


class DLCM_O6(DLCM):
    def __init__(self, eb_dim, hidden_size, max_time_len, keep_prob, reg_lambda, lr, profile_num=8, itm_spar_num=5,
                 itm_dens_num=1, max_norm=None, d_model=256, d_inner_hid=64, n_head=8):
        super().__init__(eb_dim, hidden_size, max_time_len, keep_prob, reg_lambda, lr, profile_num, itm_spar_num,
                         itm_dens_num, max_norm, d_model, d_inner_hid, n_head)

    def build_loss(self, y_pred, y_label, mode):
        y_pred_01 = y_pred

        y_pred_pos = self.position_from_score(y_pred_01)
        pos_mtx = tf.gather(self.pos_mtx, y_pred_pos)

        y_pred_02 = self.build_dlcm_function(self.add_pos_embedding(self.item_seq, pos_mtx), mode)

        y_pred_pos = self.position_adjust_from_score(y_pred_01)

        rand_pos_mtx = tf.gather(self.pos_mtx, y_pred_pos)
        # pos_mtx = tf.gather(self.pos_mtx, rand_pos)
        y_pred_03 = self.build_dlcm_function(self.add_pos_embedding(self.item_seq, rand_pos_mtx), mode)

        loss_01 = self.build_attention_loss(y_pred_01, y_label)
        loss_02 = self.build_attention_loss(y_pred_02, y_label)
        loss_03 = self.build_attention_loss(y_pred_03, y_label)

        # 约束rank和score接近
        loss_mse_12 = self.contrastive_loss(y_pred_01, y_pred_02)
        loss_mse_13 = self.contrastive_loss(y_pred_01, y_pred_03)
        loss_mse_23 = self.contrastive_loss(y_pred_02, y_pred_03)

        loss = loss_02 + loss_01 + loss_03 + loss_mse_12 + loss_mse_13 + loss_mse_23

        return loss


class GSF_O6(GSF):
    def __init__(self, eb_dim, hidden_size, max_time_len, keep_prob, reg_lambda, lr, profile_num=8, itm_spar_num=5,
                 itm_dens_num=1, max_norm=None, group_size=1, activation='relu',
                 hidden_layer_size=(512, 256, 128)):
        super().__init__(eb_dim, hidden_size, max_time_len, keep_prob, reg_lambda, lr, profile_num, itm_spar_num,
                         itm_dens_num, max_norm, group_size, activation, hidden_layer_size)

    def build_loss(self, y_pred, y_label, mode):
        y_pred_01 = y_pred

        y_pred_pos = self.position_from_score(y_pred_01)
        pos_mtx = tf.gather(self.pos_mtx, y_pred_pos)

        y_pred_02 = self.build_gsf_function(self.add_pos_embedding(self.item_seq, pos_mtx), mode)

        y_pred_pos = self.position_adjust_from_score(y_pred_01)

        rand_pos_mtx = tf.gather(self.pos_mtx, y_pred_pos)

        y_pred_03 = self.build_gsf_function(self.add_pos_embedding(self.item_seq, rand_pos_mtx), mode)

        loss_01 = self.build_norm_logloss(y_pred_01, y_label)
        loss_02 = self.build_norm_logloss(y_pred_02, y_label)
        loss_03 = self.build_norm_logloss(y_pred_03, y_label)

        # 约束rank和score接近
        loss_mse_12 = self.contrastive_loss(y_pred_01, y_pred_02)
        loss_mse_13 = self.contrastive_loss(y_pred_01, y_pred_03)
        loss_mse_23 = self.contrastive_loss(y_pred_02, y_pred_03)

        loss = loss_02 + loss_01 + loss_03 + loss_mse_12 + loss_mse_13 + loss_mse_23
        return loss


class miDNN_O6(miDNN):
    def __init__(self, eb_dim, hidden_size, max_time_len, keep_prob, reg_lambda, lr, profile_num=8, itm_spar_num=5,
                 itm_dens_num=1, max_norm=None, hidden_layer_size=(512, 256, 128)):
        super().__init__(eb_dim, hidden_size, max_time_len, keep_prob, reg_lambda, lr, profile_num, itm_spar_num,
                         itm_dens_num, max_norm, hidden_layer_size)

    def build_loss(self, y_pred, y_label, mode):
        y_pred_01 = y_pred

        y_pred_pos = self.position_from_score(y_pred_01)
        pos_mtx = tf.gather(self.pos_mtx, y_pred_pos)

        y_pred_02 = self.build_miDNN_function(self.add_pos_embedding(self.item_seq, pos_mtx), mode)

        y_pred_pos = self.position_adjust_from_score(y_pred_01)

        rand_pos_mtx = tf.gather(self.pos_mtx, y_pred_pos)

        y_pred_03 = self.build_miDNN_function(self.add_pos_embedding(self.item_seq, rand_pos_mtx), mode)

        loss_01 = self.build_logloss(y_pred_01, y_label)
        loss_02 = self.build_logloss(y_pred_02, y_label)
        loss_03 = self.build_logloss(y_pred_03, y_label)

        # 约束rank和score接近
        loss_mse_12 = self.contrastive_loss(y_pred_01, y_pred_02)
        loss_mse_13 = self.contrastive_loss(y_pred_01, y_pred_03)
        loss_mse_23 = self.contrastive_loss(y_pred_02, y_pred_03)

        loss = loss_02 + loss_01 + loss_03 + loss_mse_12 + loss_mse_13 + loss_mse_23
        return loss


class PEAR_O6(PEAR):

    def __init__(self, eb_dim, hidden_size, max_time_len, keep_prob, reg_lambda, lr,
                 profile_num=8, itm_spar_num=5, itm_dens_num=1, max_norm=None, d_model=500,
                 d_inner_hid=128, n_head=1, cross_n_head=1, use_hist_seq=False):
        super(PEAR_O6, self).__init__(eb_dim, hidden_size, max_time_len, keep_prob, reg_lambda, lr,
                                      profile_num, itm_spar_num, itm_dens_num, max_norm, d_model,
                                      d_inner_hid, n_head, cross_n_head, use_hist_seq)

    def build_loss(self, y_pred, y_label, mode):
        logger.info(f'y_pred {y_pred}, y_label {y_label}')

        # 增加辅助loss， 即当前trace是否有点击
        y_label = tf.concat([y_label, tf.reduce_max(y_label, axis=-1, keepdims=True)], axis=-1)

        y_pred_01 = self.y_pred_full

        y_pred_pos = self.position_from_score(y_pred)
        pos_mtx = tf.gather(self.pos_mtx, y_pred_pos)

        y_pred_02, _ = self.build_pear_function(self.add_pos_embedding(self.item_seq, pos_mtx, False), mode)

        y_pred_pos = self.position_adjust_from_score(y_pred)

        rand_pos_mtx = tf.gather(self.pos_mtx, y_pred_pos)

        y_pred_03, _ = self.build_pear_function(self.add_pos_embedding(self.item_seq, rand_pos_mtx, False), mode)

        loss_01 = self.build_logloss(y_pred_01, y_label)
        loss_02 = self.build_logloss(y_pred_02, y_label)
        loss_03 = self.build_logloss(y_pred_03, y_label)

        # 约束rank和score接近
        loss_mse_12 = self.contrastive_loss(y_pred_01, y_pred_02)
        loss_mse_13 = self.contrastive_loss(y_pred_01, y_pred_03)
        loss_mse_23 = self.contrastive_loss(y_pred_02, y_pred_03)

        loss = loss_02 + loss_01 + loss_03 + loss_mse_12 + loss_mse_13 + loss_mse_23
        return loss


class RAISE_O6(RAISE):
    def __init__(self, eb_dim, hidden_size, max_time_len, keep_prob, reg_lambda, lr, profile_num=8, itm_spar_num=5,
                 itm_dens_num=1, max_norm=None, d_model=256, d_inner_hid=64, a_head=1):
        super().__init__(eb_dim, hidden_size, max_time_len, keep_prob, reg_lambda, lr, profile_num, itm_spar_num,
                         itm_dens_num, max_norm,
                         d_model, d_inner_hid, a_head)

    def build_loss(self, y_pred, y_label, mode):
        y_pred_01 = y_pred
        tf.summary.histogram('y_pred_01', y_pred_01)
        y_pred_pos = self.position_from_score(y_pred_01)
        pos_mtx = tf.gather(self.pos_mtx, y_pred_pos)

        y_pred_02 = self.build_raise_fc_function(self.add_pos_embedding(self.item_seq, pos_mtx, False), mode)

        tf.summary.histogram('y_pred_02', y_pred_02)
        y_pred_pos = self.position_adjust_from_score(y_pred_01)

        rand_pos_mtx = tf.gather(self.pos_mtx, y_pred_pos)
        # pos_mtx = tf.gather(self.pos_mtx, rand_pos)
        y_pred_03 = self.build_raise_fc_function(self.add_pos_embedding(self.item_seq, rand_pos_mtx, False), mode)

        tf.summary.histogram('y_pred_03', y_pred_03)
        loss_01 = self.build_attention_loss(y_pred_01, y_label)
        loss_02 = self.build_attention_loss(y_pred_02, y_label)
        loss_03 = self.build_attention_loss(y_pred_03, y_label)

        # 约束rank和score接近
        loss_mse_12 = self.contrastive_loss(y_pred_01, y_pred_02)
        loss_mse_13 = self.contrastive_loss(y_pred_01, y_pred_03)
        loss_mse_23 = self.contrastive_loss(y_pred_02, y_pred_03)

        tf.summary.scalar('loss_01', loss_01)
        tf.summary.scalar('loss_02', loss_02)
        tf.summary.scalar('loss_03', loss_03)
        tf.summary.scalar('loss_mse_12', loss_mse_12)
        tf.summary.scalar('loss_mse_13', loss_mse_13)
        tf.summary.scalar('loss_mse_23', loss_mse_23)
        self.loss = loss_02 + loss_01 + loss_03 + loss_mse_12 + loss_mse_13 + loss_mse_23

        return self.loss


class SetRank_O1(SetRank_O6):
    def __init__(self, eb_dim, hidden_size, max_time_len, keep_prob, reg_lambda, lr, profile_num=8, itm_spar_num=5,
                 itm_dens_num=1, max_norm=None, d_model=256, d_inner_hid=64, n_head=8):
        super().__init__(eb_dim, hidden_size, max_time_len, keep_prob, reg_lambda, lr, profile_num, itm_spar_num,
                         itm_dens_num, max_norm,
                         d_model, d_inner_hid, n_head)

    def build_loss(self, y_pred, y_label, mode):
        y_pred_01 = y_pred

        y_pred_pos = self.position_from_score(y_pred_01)
        pos_mtx = tf.gather(self.pos_mtx, y_pred_pos)

        y_pred_02 = self.build_setrank_function(self.add_pos_embedding(self.item_seq, pos_mtx), mode)

        loss_01 = self.build_attention_loss(y_pred_01, y_label)
        loss_02 = self.build_attention_loss(y_pred_02, y_label)

        # 约束rank和score接近
        loss_mse_12 = self.contrastive_loss(y_pred_01, y_pred_02)

        self.loss = loss_02 + loss_01 + loss_mse_12

        return self.loss


class DLCM_O1(DLCM_O6):
    def __init__(self, eb_dim, hidden_size, max_time_len, keep_prob, reg_lambda, lr, profile_num=8, itm_spar_num=5,
                 itm_dens_num=1, max_norm=None, d_model=256, d_inner_hid=64, n_head=8):
        super().__init__(eb_dim, hidden_size, max_time_len, keep_prob, reg_lambda, lr, profile_num, itm_spar_num,
                         itm_dens_num, max_norm, d_model, d_inner_hid, n_head)

    def build_loss(self, y_pred, y_label, mode):
        y_pred_01 = y_pred

        y_pred_pos = self.position_from_score(y_pred_01)
        pos_mtx = tf.gather(self.pos_mtx, y_pred_pos)

        y_pred_02 = self.build_dlcm_function(self.add_pos_embedding(self.item_seq, pos_mtx), mode)

        loss_01 = self.build_attention_loss(y_pred_01, y_label)
        loss_02 = self.build_attention_loss(y_pred_02, y_label)

        # 约束rank和score接近
        loss_mse_12 = self.contrastive_loss(y_pred_01, y_pred_02)

        loss = loss_01 + loss_02 + loss_mse_12

        return loss


class GSF_O1(GSF_O6):
    def __init__(self, eb_dim, hidden_size, max_time_len, keep_prob, reg_lambda, lr, profile_num=8, itm_spar_num=5,
                 itm_dens_num=1, max_norm=None, group_size=1, activation='relu',
                 hidden_layer_size=(512, 256, 128)):
        super().__init__(eb_dim, hidden_size, max_time_len, keep_prob, reg_lambda, lr, profile_num, itm_spar_num,
                         itm_dens_num, max_norm, group_size, activation, hidden_layer_size)

    def build_loss(self, y_pred, y_label, mode):
        y_pred_01 = y_pred

        y_pred_pos = self.position_from_score(y_pred_01)
        pos_mtx = tf.gather(self.pos_mtx, y_pred_pos)

        y_pred_02 = self.build_gsf_function(self.add_pos_embedding(self.item_seq, pos_mtx), mode)

        loss_01 = self.build_norm_logloss(y_pred_01, y_label)
        loss_02 = self.build_norm_logloss(y_pred_02, y_label)

        # 约束rank和score接近
        loss_mse_12 = self.contrastive_loss(y_pred_01, y_pred_02)

        loss = loss_02 + loss_01 + loss_mse_12
        return loss


class miDNN_O1(miDNN_O6):
    def __init__(self, eb_dim, hidden_size, max_time_len, keep_prob, reg_lambda, lr, profile_num=8, itm_spar_num=5,
                 itm_dens_num=1, max_norm=None, hidden_layer_size=(512, 256, 128)):
        super().__init__(eb_dim, hidden_size, max_time_len, keep_prob, reg_lambda, lr, profile_num, itm_spar_num,
                         itm_dens_num, max_norm, hidden_layer_size)

    def build_loss(self, y_pred, y_label, mode):
        y_pred_01 = y_pred

        y_pred_pos = self.position_from_score(y_pred_01)
        pos_mtx = tf.gather(self.pos_mtx, y_pred_pos)

        y_pred_02 = self.build_miDNN_function(self.add_pos_embedding(self.item_seq, pos_mtx), mode)

        loss_01 = self.build_logloss(y_pred_01, y_label)
        loss_02 = self.build_logloss(y_pred_02, y_label)

        # 约束rank和score接近
        loss_mse_12 = self.contrastive_loss(y_pred_01, y_pred_02)

        loss = loss_02 + loss_01 + loss_mse_12
        return loss


class PEAR_O1(PEAR_O6):

    def __init__(self, eb_dim, hidden_size, max_time_len, keep_prob, reg_lambda, lr,
                 profile_num=8, itm_spar_num=5, itm_dens_num=1, max_norm=None, d_model=500,
                 d_inner_hid=128, n_head=1, cross_n_head=1, use_hist_seq=False):
        super(PEAR_O1, self).__init__(eb_dim, hidden_size, max_time_len, keep_prob, reg_lambda, lr,
                                      profile_num, itm_spar_num, itm_dens_num, max_norm, d_model,
                                      d_inner_hid, n_head, cross_n_head, use_hist_seq)

    def build_loss(self, y_pred, y_label, mode):
        logger.info(f'y_pred {y_pred}, y_label {y_label}')

        # 增加辅助loss， 即当前trace是否有点击
        y_label = tf.concat([y_label, tf.reduce_max(y_label, axis=-1, keepdims=True)], axis=-1)

        y_pred_01 = self.y_pred_full

        y_pred_pos = self.position_from_score(y_pred)
        pos_mtx = tf.gather(self.pos_mtx, y_pred_pos)

        y_pred_02, _ = self.build_pear_function(self.add_pos_embedding(self.item_seq, pos_mtx, False), mode)

        loss_01 = self.build_logloss(y_pred_01, y_label)
        loss_02 = self.build_logloss(y_pred_02, y_label)

        # 约束rank和score接近
        loss_mse_12 = self.contrastive_loss(y_pred_01, y_pred_02)

        loss = loss_02 + loss_01 + loss_mse_12
        return loss


class RAISE_O1(RAISE_O6):
    def __init__(self, eb_dim, hidden_size, max_time_len, keep_prob, reg_lambda, lr, profile_num=8, itm_spar_num=5,
                 itm_dens_num=1, max_norm=None, d_model=256, d_inner_hid=64, a_head=1):
        super().__init__(eb_dim, hidden_size, max_time_len, keep_prob, reg_lambda, lr, profile_num, itm_spar_num,
                         itm_dens_num, max_norm,
                         d_model, d_inner_hid, a_head)

    def build_loss(self, y_pred, y_label, mode):
        y_pred_01 = y_pred
        tf.summary.histogram('y_pred_01', y_pred_01)
        y_pred_pos = self.position_from_score(y_pred_01)
        pos_mtx = tf.gather(self.pos_mtx, y_pred_pos)

        y_pred_02 = self.build_raise_fc_function(self.add_pos_embedding(self.item_seq, pos_mtx, False), mode)

        tf.summary.histogram('y_pred_02', y_pred_02)
        loss_01 = self.build_attention_loss(y_pred_01, y_label)
        loss_02 = self.build_attention_loss(y_pred_02, y_label)

        # 约束rank和score接近
        loss_mse_12 = self.contrastive_loss(y_pred_01, y_pred_02)

        tf.summary.scalar('loss_01', loss_01)
        tf.summary.scalar('loss_02', loss_02)
        tf.summary.scalar('loss_mse_12', loss_mse_12)
        self.loss = loss_02 + loss_01 + loss_mse_12

        return self.loss


class SetRank_O2(SetRank_O6):
    def __init__(self, eb_dim, hidden_size, max_time_len, keep_prob, reg_lambda, lr, profile_num=8, itm_spar_num=5,
                 itm_dens_num=1, max_norm=None, d_model=256, d_inner_hid=64, n_head=8):
        super().__init__(eb_dim, hidden_size, max_time_len, keep_prob, reg_lambda, lr, profile_num, itm_spar_num,
                         itm_dens_num, max_norm,
                         d_model, d_inner_hid, n_head)

    def build_loss(self, y_pred, y_label, mode):
        y_pred_01 = y_pred

        y_pred_pos = self.position_adjust_from_score(y_pred_01)

        rand_pos_mtx = tf.gather(self.pos_mtx, y_pred_pos)
        y_pred_03 = self.build_setrank_function(self.add_pos_embedding(self.item_seq, rand_pos_mtx), mode)

        loss_01 = self.build_attention_loss(y_pred_01, y_label)
        loss_03 = self.build_attention_loss(y_pred_03, y_label)

        # 约束rank和score接近
        loss_mse_13 = self.contrastive_loss(y_pred_01, y_pred_03)
        self.loss = loss_01 + loss_03 + loss_mse_13

        return self.loss


class DLCM_O2(DLCM_O6):
    def __init__(self, eb_dim, hidden_size, max_time_len, keep_prob, reg_lambda, lr, profile_num=8, itm_spar_num=5,
                 itm_dens_num=1, max_norm=None, d_model=256, d_inner_hid=64, n_head=8):
        super().__init__(eb_dim, hidden_size, max_time_len, keep_prob, reg_lambda, lr, profile_num, itm_spar_num,
                         itm_dens_num, max_norm, d_model, d_inner_hid, n_head)

    def build_loss(self, y_pred, y_label, mode):
        y_pred_01 = y_pred

        y_pred_pos = self.position_adjust_from_score(y_pred_01)

        rand_pos_mtx = tf.gather(self.pos_mtx, y_pred_pos)
        y_pred_03 = self.build_dlcm_function(self.add_pos_embedding(self.item_seq, rand_pos_mtx), mode)

        loss_01 = self.build_attention_loss(y_pred_01, y_label)
        loss_03 = self.build_attention_loss(y_pred_03, y_label)

        # 约束rank和score接近
        loss_mse_13 = self.contrastive_loss(y_pred_01, y_pred_03)
        self.loss = loss_01 + loss_03 + loss_mse_13

        return self.loss


class GSF_O2(GSF_O6):
    def __init__(self, eb_dim, hidden_size, max_time_len, keep_prob, reg_lambda, lr, profile_num=8, itm_spar_num=5,
                 itm_dens_num=1, max_norm=None, group_size=1, activation='relu',
                 hidden_layer_size=(512, 256, 128)):
        super().__init__(eb_dim, hidden_size, max_time_len, keep_prob, reg_lambda, lr, profile_num, itm_spar_num,
                         itm_dens_num, max_norm, group_size, activation, hidden_layer_size)

    def build_loss(self, y_pred, y_label, mode):
        y_pred_01 = y_pred

        y_pred_pos = self.position_adjust_from_score(y_pred_01)

        rand_pos_mtx = tf.gather(self.pos_mtx, y_pred_pos)
        y_pred_03 = self.build_gsf_function(self.add_pos_embedding(self.item_seq, rand_pos_mtx), mode)

        loss_01 = self.build_norm_logloss(y_pred_01, y_label)
        loss_03 = self.build_norm_logloss(y_pred_03, y_label)

        # 约束rank和score接近
        loss_mse_13 = self.contrastive_loss(y_pred_01, y_pred_03)
        self.loss = loss_01 + loss_03 + loss_mse_13

        return self.loss


class miDNN_O2(miDNN_O6):
    def __init__(self, eb_dim, hidden_size, max_time_len, keep_prob, reg_lambda, lr, profile_num=8, itm_spar_num=5,
                 itm_dens_num=1, max_norm=None, hidden_layer_size=(512, 256, 128)):
        super().__init__(eb_dim, hidden_size, max_time_len, keep_prob, reg_lambda, lr, profile_num, itm_spar_num,
                         itm_dens_num, max_norm, hidden_layer_size)

    def build_loss(self, y_pred, y_label, mode):
        y_pred_01 = y_pred

        y_pred_pos = self.position_adjust_from_score(y_pred_01)

        rand_pos_mtx = tf.gather(self.pos_mtx, y_pred_pos)
        y_pred_03 = self.build_miDNN_function(self.add_pos_embedding(self.item_seq, rand_pos_mtx), mode)

        loss_01 = self.build_logloss(y_pred_01, y_label)
        loss_03 = self.build_logloss(y_pred_03, y_label)

        # 约束rank和score接近
        loss_mse_13 = self.contrastive_loss(y_pred_01, y_pred_03)
        self.loss = loss_01 + loss_03 + loss_mse_13

        return self.loss


class PEAR_O2(PEAR_O6):

    def __init__(self, eb_dim, hidden_size, max_time_len, keep_prob, reg_lambda, lr,
                 profile_num=8, itm_spar_num=5, itm_dens_num=1, max_norm=None, d_model=500,
                 d_inner_hid=128, n_head=1, cross_n_head=1, use_hist_seq=False):
        super(PEAR_O2, self).__init__(eb_dim, hidden_size, max_time_len, keep_prob, reg_lambda, lr,
                                      profile_num, itm_spar_num, itm_dens_num, max_norm, d_model,
                                      d_inner_hid, n_head, cross_n_head, use_hist_seq)

    def build_loss(self, y_pred, y_label, mode):
        logger.info(f'y_pred {y_pred}, y_label {y_label}')

        # 增加辅助loss， 即当前trace是否有点击
        y_label = tf.concat([y_label, tf.reduce_max(y_label, axis=-1, keepdims=True)], axis=-1)

        y_pred_01 = self.y_pred_full

        y_pred_pos = self.position_adjust_from_score(y_pred)

        rand_pos_mtx = tf.gather(self.pos_mtx, y_pred_pos)

        y_pred_03, _ = self.build_pear_function(self.add_pos_embedding(self.item_seq, rand_pos_mtx, False), mode)

        loss_01 = self.build_logloss(y_pred_01, y_label)
        loss_03 = self.build_logloss(y_pred_03, y_label)

        # 约束rank和score接近
        loss_mse_13 = self.contrastive_loss(y_pred_01, y_pred_03)

        loss = loss_01 + loss_03 + loss_mse_13
        return loss


class RAISE_O2(RAISE_O6):
    def __init__(self, eb_dim, hidden_size, max_time_len, keep_prob, reg_lambda, lr, profile_num=8, itm_spar_num=5,
                 itm_dens_num=1, max_norm=None, d_model=256, d_inner_hid=64, a_head=1):
        super().__init__(eb_dim, hidden_size, max_time_len, keep_prob, reg_lambda, lr, profile_num, itm_spar_num,
                         itm_dens_num, max_norm,
                         d_model, d_inner_hid, a_head)

    def build_loss(self, y_pred, y_label, mode):
        y_pred_01 = y_pred
        tf.summary.histogram('y_pred_01', y_pred_01)
        y_pred_pos = self.position_adjust_from_score(y_pred_01)

        rand_pos_mtx = tf.gather(self.pos_mtx, y_pred_pos)
        # pos_mtx = tf.gather(self.pos_mtx, rand_pos)
        y_pred_03 = self.build_raise_fc_function(self.add_pos_embedding(self.item_seq, rand_pos_mtx, False), mode)

        tf.summary.histogram('y_pred_03', y_pred_03)
        loss_01 = self.build_attention_loss(y_pred_01, y_label)
        loss_03 = self.build_attention_loss(y_pred_03, y_label)

        # 约束rank和score接近
        loss_mse_13 = self.contrastive_loss(y_pred_01, y_pred_03)

        tf.summary.scalar('loss_01', loss_01)
        tf.summary.scalar('loss_03', loss_03)
        tf.summary.scalar('loss_mse_13', loss_mse_13)
        self.loss = loss_01 + loss_03 + loss_mse_13

        return self.loss


from alps.metrics._base_validator import BaseValidator


class NDCGValidator(BaseValidator):
    def __init__(self, labels=None, predictions=None, average="macro", auc_kwargs=None, **kwargs):
        super().__init__(labels, predictions, **kwargs)
        self._average = average
        self.metric_names = ['ndcg']
        self._auc_kwargs = auc_kwargs or {}
        if self._average is not None:
            self._auc_kwargs['average'] = self._average

    def compute_metrics_using_numpy(self, labels, predictions):
        from sklearn.metrics import ndcg_score
        auc_result = ndcg_score(labels, predictions)
        return np.float32(auc_result)

    def build_tf_metric_ops(self):
        return tf.metrics.mean_squared_error(predictions=self._predictions, labels=self._labels)

    def get_metric_comparator(self):
        from alps.framework.exporter.base import MetricComparator, Goal
        return MetricComparator(self.metric_names[0], Goal.MAXIMIZE)


class MAPValidator(BaseValidator):
    def __init__(self, labels=None, predictions=None, average="macro", auc_kwargs=None, **kwargs):
        super().__init__(labels, predictions, **kwargs)
        self._average = average
        self.at_k = kwargs['k']
        self.metric_names = ['map@' + str(self.at_k)]
        self._auc_kwargs = auc_kwargs or {}
        if self._average is not None:
            self._auc_kwargs['average'] = self._average

    def compute_metrics_using_numpy(self, labels, predictions):
        final = np.argsort(predictions, axis=-1)
        click = np.take_along_axis(labels, final, axis=-1)  # reranked labels
        click = click[:, ::-1]

        indices = np.asarray(range(1, np.shape(click)[-1] + 1)).reshape(1, -1)

        sub_labels = click[:, :self.at_k]
        sub_indices = indices[:, :self.at_k]
        ap_count = np.cumsum(sub_labels, axis=-1)
        ap_value = ap_count / sub_indices
        ap_value = np.cumsum(ap_value * sub_labels, axis=-1)

        ap_value = ap_value[:, -1]
        ap_count = ap_count[:, -1]

        map = ap_value / np.maximum(ap_count, 1)
        return np.float32(np.mean(map))

    def build_tf_metric_ops(self):
        return tf.metrics.mean_squared_error(predictions=self._predictions, labels=self._labels)

    def get_metric_comparator(self):
        from alps.framework.exporter.base import MetricComparator, Goal
        return MetricComparator(self.metric_names[0], Goal.MAXIMIZE)


class ClickValidator(BaseValidator):
    def __init__(self, labels=None, predictions=None, average="macro", auc_kwargs=None, **kwargs):
        super().__init__(labels, predictions, **kwargs)
        self._average = average
        self.at_k = kwargs['k']
        self.metric_names = ['rec@' + str(self.at_k)]
        self._auc_kwargs = auc_kwargs or {}
        if self._average is not None:
            self._auc_kwargs['average'] = self._average

    def compute_metrics_using_numpy(self, labels, predictions):
        final = np.argsort(predictions, axis=-1)
        click = np.take_along_axis(labels, final, axis=-1)  # reranked labels
        click = click[:, ::-1]
        return np.mean(np.sum(click[:, :self.at_k], axis=-1))

    def build_tf_metric_ops(self):
        return tf.metrics.mean_squared_error(predictions=self._predictions, labels=self._labels)

    def get_metric_comparator(self):
        from alps.framework.exporter.base import MetricComparator, Goal
        return MetricComparator(self.metric_names[0], Goal.MAXIMIZE)


class SequenceEqualValidator(BaseValidator):
    def __init__(self, labels=None, predictions=None, average="macro", auc_kwargs=None, **kwargs):
        super().__init__(labels, predictions, **kwargs)
        self._average = average
        self.metric_names = ['sq']
        self._auc_kwargs = auc_kwargs or {}
        if self._average is not None:
            self._auc_kwargs['average'] = self._average

    def compute_metrics_using_numpy(self, labels, predictions):
        seq1 = self.position_from_score(labels)
        seq2 = self.position_from_score(predictions)
        seq_num = np.sum(np.ones_like(seq1, dtype='float'), axis=-1)
        eq_num = np.sum(np.equal(seq1, seq2).astype(float), axis=-1)

        score = np.equal(eq_num, seq_num).astype(float)
        # return np.mean(eq_num / seq_num)
        return np.mean(score)

    def position_from_score(self, pred):
        orig_pos = np.argsort(pred, kind='stable')[::-1]
        orig_pos = np.argsort(orig_pos, kind='stable')
        return orig_pos

    def build_tf_metric_ops(self):
        return tf.metrics.mean_squared_error(predictions=self._predictions, labels=self._labels)

    def get_metric_comparator(self):
        from alps.framework.exporter.base import MetricComparator, Goal
        return MetricComparator(self.metric_names[0], Goal.MAXIMIZE)


class SequenceEqualScoreValidator(BaseValidator):
    def __init__(self, labels=None, predictions=None, average="macro", auc_kwargs=None, **kwargs):
        super().__init__(labels, predictions, **kwargs)
        self._average = average
        self.metric_names = ['sq']
        self._auc_kwargs = auc_kwargs or {}
        if self._average is not None:
            self._auc_kwargs['average'] = self._average

    def compute_metrics_using_numpy(self, labels, predictions):
        seq1 = self.position_from_score(labels)
        seq2 = self.position_from_score(predictions)
        seq_num = np.sum(np.ones_like(seq1, dtype='float'), axis=-1)
        eq_num = np.sum(np.equal(seq1, seq2).astype(float), axis=-1)

        return np.mean(eq_num / seq_num)

    def position_from_score(self, pred):
        orig_pos = np.argsort(pred, kind='stable')[::-1]
        orig_pos = np.argsort(orig_pos, kind='stable')
        return orig_pos

    def build_tf_metric_ops(self):
        return tf.metrics.mean_squared_error(predictions=self._predictions, labels=self._labels)

    def get_metric_comparator(self):
        from alps.framework.exporter.base import MetricComparator, Goal
        return MetricComparator(self.metric_names[0], Goal.MAXIMIZE)
