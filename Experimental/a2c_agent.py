from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense

from spektral.layers import GATConv, DiffPool

import tensorflow as tf


class ActorCritic(Model):

    def __init__(self, amt_actions, amt_nodes, amt_features):
        super().__init__()

        # Config
        self.amt_actions = amt_actions
        self.amt_nodes = amt_nodes
        self.amt_features = amt_features

        # GAT convolutions
        self.first_conv = GATConv(amt_features * 3)
        self.second_conv = GATConv(amt_features * 2)
        self.third_conv = GATConv(amt_features)

        # Differential pooling layers
        self.first_pool = DiffPool(14)
        self.second_pool = DiffPool(7)
        self.third_pool = DiffPool(1)

        self.actor = Dense(amt_actions)
        self.critic = Dense(1)

    def get_config(self):
        return {
            "amt_actions": self.amt_actions,
            "amt_nodes": self.amt_nodes,
            "amt_features": self.amt_features
        }

    def call(self, inputs, training=None, mask=None):
        feature_matrix = inputs[0]
        adj_matrix = inputs[1]

        if not adj_matrix.dtype == tf.float32:
            adj_matrix = tf.cast(adj_matrix, tf.float32)

        # Embedding
        feature_matrix = self.first_conv([feature_matrix, adj_matrix])
        feature_matrix = self.second_conv([feature_matrix, adj_matrix])
        feature_matrix = self.third_conv([feature_matrix, adj_matrix])

        # Pooling
        feature_matrix, adj_matrix = self.first_pool([feature_matrix, adj_matrix])
        feature_matrix, adj_matrix = self.second_pool([feature_matrix, adj_matrix])
        feature_matrix, _ = self.third_pool([feature_matrix, adj_matrix])

        # Actor / Critic
        return self.actor(feature_matrix), self.critic(feature_matrix)
