
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import scipy.io as sio
import os
import scipy.sparse as sp
import random
import numpy as np
import matplotlib.pyplot as plt
import h5py

from gcn.utils import *
from gcn.metrics import *
from gcn.models import GCN, MLP
from os.path import join
from absl import flags


# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings


flags = tf.app.flags #tf.compat.v1.flags#
FLAGS = flags.FLAGS

#flags.DEFINE_string('dataset', 'myData', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed' , 'myData'
flags.DEFINE_string('model', 'gcn_cheby', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.') # 0.01 : default
flags.DEFINE_integer('epochs', 100, 'Number of epochs to train.') # 200:default
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.') #16 : default
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).') # 0.5 :default
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.') # 5e-4 : default
flags.DEFINE_integer('early_stopping',10, 'Tolerance for early stopping (# of epochs).') #10:default
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.') #3 : default

# Load adjacency matrix and features

dataDir = 'INPUTMATFILES_NDFI\'

# Create variables to store matfiles

A = sio.loadmat(join(dataDir,'adj.mat')) 
B = sio.loadmat(join(dataDir,'PixelFeatures.mat')) 
C = sio.loadmat(join(dataDir,'train_mask.mat'))
D = sio.loadmat(join(dataDir,'test_mask.mat'))
E = sio.loadmat(join(dataDir,'val_mask.mat'))
F = sio.loadmat(join(dataDir,'y_train.mat'))
G = sio.loadmat(join(dataDir,'y_test.mat'))
H = sio.loadmat(join(dataDir,'y_val.mat'))
###################################################################################################

adj = A.get('ADJ_ALL_sp') 
features_i = sp.csr_matrix(B['Features_PIXEL']) # features have to be converted to sparse representation to avoid out of memory errors, otherwise use B.get('Features_PIXEL')   
features = features_i.astype(float)
train_mask = C.get('train_mask')
test_mask = D.get('test_mask')
val_mask = E.get('val_mask')
y_train = F.get('y_train')
y_test = G.get('y_test')
y_val = H.get('y_val')

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
    outs_val = sess.run([model.loss, model.accuracy, tf.argmax(model.outputs,1), model.outputs, model.activations], feed_dict=feed_dict_val)
       
    return outs_val[0], outs_val[1], (time.time() - t_test), outs_val[2], outs_val[3], outs_val[4]

# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []
TRAIN_ACC = []
VAL_ACC = []

train_Err = []
val_Err = []

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
 
    # Validation
    cost, acc, duration, pp, sc, emb = evaluate(features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))
    
    TRAIN_ACC.append(outs[2])
    VAL_ACC.append(acc)
    
    train_Err.append(outs[1])
    val_Err.append(cost)
    
    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# Plot the training and validation accuracies
 
epoch_cnt = range(1,len(TRAIN_ACC)+1)
#tit = ''.join(['model:',tf.flags.FLAGS.model, 'lr:', str(tf.flags.FLAGS.learning_rate), 'ep:',str(tf.flags.FLAGS.epochs),'hid:' str(tf.flags.FLAGS.hidden1),'dr:',str(tf.flags.FLAGS.dropout),'wtdec:', str(tf.flags.FLAGS.weight_decay),'es:',str(tf.flags.FLAGS.early_stopping),'deg:',str(tf.flags.FLAGS.max_degree)])
plt.figure()
plt.plot(epoch_cnt, TRAIN_ACC, 'r--')
plt.plot(epoch_cnt, VAL_ACC, 'b-')
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.figure()
plt.plot(epoch_cnt, train_Err, 'r--')
plt.plot(epoch_cnt, val_Err, 'b-')
plt.legend(['Training Loss', 'Validation Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')


# Testing
test_cost, test_acc, test_duration, pred_labels, prediction_scores, test_emb = evaluate(features, support, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

sio.savemat('GNNDepthlabels.mat',{'depth_labels':pred_labs})
