from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import scipy.io as sio
import os
import scipy.sparse as sp
#import numpy


from gcn.utils import *
from gcn.models import GCN, MLP
from os.path import join


# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'myData', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed' , 'myData'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.') # 200:default
flags.DEFINE_integer('hidden1', 20, 'Number of units in hidden layer 1.') #16 : default
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).') # 0.5 :default
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).') #10:default
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.') #3 : default

# Load own adjacency matrix and features
#dataDir = 'S:\\PGM Image Phylogeny\\Codes\\gcn_copy1\\'
#
#A = sio.loadmat(join(dataDir,'adj.mat')) #,squeeze_me=True
#B = sio.loadmat(join(dataDir,'features.mat'))
#C = sio.loadmat(join(dataDir,'train_mask.mat'))
#D = sio.loadmat(join(dataDir,'test_mask.mat'))
#E = sio.loadmat(join(dataDir,'val_mask.mat'))
#F = sio.loadmat(join(dataDir,'y_train.mat'))
#G = sio.loadmat(join(dataDir,'y_test.mat'))
#H = sio.loadmat(join(dataDir,'y_val.mat'))
#
#adj = A.get('adj')
#features = B.get('features')
#train_mask = C.get('train_mask')
#test_mask = D.get('test_mask')
#val_mask = E.get('val_mask')
#y_train = F.get('y_train')
#y_test = G.get('y_test')
#y_val = H.get('y_val')
#
#for iter in range(train_mask.shape[1]):
#    train_mask[0][iter] = bool(train_mask[0][iter])
#    test_mask[0][iter] = bool(test_mask[0][iter])
#    val_mask[0][iter] = bool(val_mask[0][iter])


# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)

# Some preprocessing
features = preprocess_features(features)

if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, input_dim=features[2][1], logging=True)

# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

    # Validation
    cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# Testing
test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
