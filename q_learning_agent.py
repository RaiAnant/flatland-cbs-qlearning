from tensorflow.keras.layers import Input, Dense
from spektral.layers import GATConv, DiffPool

import tensorflow as tf
import numpy as np


def deep_q_network(amt_nodes, amt_features, amt_actions):
    """A q-network consisting of graph attention layers, differential pooling and a MLP.

    :param amt_nodes: The amount of nodes stored within the observation.
    :param amt_features: The amount of features stored within the observation.
    :param amt_actions: The amount of actions which the q-network can predict.
    :return: A q-network.
    """
    feature_matrix_input = Input(shape=(amt_nodes, amt_features))
    adj_matrix_input = Input(shape=(amt_nodes, amt_nodes))

    # GAT convolutions
    feature_matrix = GATConv(amt_features * 3)([feature_matrix_input, adj_matrix_input])
    feature_matrix = GATConv(amt_features * 2)([feature_matrix, adj_matrix_input])
    feature_matrix = GATConv(amt_features)([feature_matrix, adj_matrix_input])

    # Differential pooling layers
    feature_matrix, adj_matrix = DiffPool(14)([feature_matrix, adj_matrix_input])
    feature_matrix, adj_matrix = DiffPool(7)([feature_matrix, adj_matrix])
    feature_matrix, _ = DiffPool(1)([feature_matrix, adj_matrix])

    # Dense layers predicting the action
    feature_matrix = Dense(amt_features * 3, activation="relu")(feature_matrix)
    feature_matrix = Dense(amt_features * 2, activation="relu")(feature_matrix)
    feature_matrix = Dense(amt_actions, activation="linear")(feature_matrix)

    model = tf.keras.Model(inputs=[feature_matrix_input, adj_matrix_input], outputs=feature_matrix)
    model.summary()

    dummy_nodes = np.zeros((1, amt_nodes, amt_features))
    dummy_adjacency = np.zeros((1, amt_nodes, amt_nodes))
    model([dummy_nodes, dummy_adjacency])

    return model
