
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

# Load own adjacency matrix and features

# dataDir = 'S:\\PGM Image Phylogeny\\Codes\\gcn_copy1\\INPUTMATFILES_NEW'
# dataDir = 'S:\\PGM Image Phylogeny\\Codes\\gcn_copy1\\INPUTMATFILES_NEW\\Photoshop'
 # dataDir = 'S:\\PGM Image Phylogeny\\Codes\\gcn_copy1\\INPUTMATFILES_NEW\\IrisImages\\IrisImages_5nodes'

#dataDir = 'S:\\PGM Image Phylogeny\\Codes\\gcn_copy1\\INPUTMATFILES_NEW\\Fakedetection\\BalancedDataset'
#dataDir ='S:\\PGM Image Phylogeny\\Codes\\gcn_copy1\\INPUTMATFILES_NEW\\Fakedetection\\ADJ_NEW'
#dataDir = 'S:\\PGM Image Phylogeny\\Codes\\gcn_copy1\\INPUTMATFILES_NEW\\Fakedetection\\BalancedDataset\\NEW_DEGREE'
#dataDir ='S:\\PGM Image Phylogeny\\Codes\\gcn_copy1\\INPUTMATFILES_NEW\\Fakedetection\\BalancedDataset\\BalancedTest'
 
 
# dataDir = 'S:\\PGM Image Phylogeny\\Codes\\gcn_copy1\\INPUTMATFILES_NEW\\Fakedetection\\ADJ_NEW\\Psi0_5'
 
# dataDir ='S:\\PGM Image Phylogeny\\Codes\\gcn_copy1\\INPUTMATFILES_JUNE\\FingerprintImages'
#dataDir = 'S:\\PGM Image Phylogeny\\Codes\\gcn_copy1\\INPUTMATFILES_JUNE\\Photoshop'
#dataDir = 'S:\\PGM Image Phylogeny\\Codes\\gcn_copy1\\INPUTMATFILES_JUNE\\FakeDetection\\GAN\\PSI0_5_NEW_MOREBONAFIDE'
#dataDir ='S:\\PGM Image Phylogeny\\Codes\\gcn_copy1\\INPUTMATFILES_JUNE\\FakeDetection\\GAN\\PSI1_0'
#dataDir = 'S:\\PGM Image Phylogeny\\Codes\gcn_copy1\\INPUTMATFILES_JUNE\\FakeDetection\\GAN\\MUG_NEW_MOREBONAFIDE'

#dataDir = 'S:\\PGM Image Phylogeny\\Codes\\gcn_copy1\\INPUTMATFILES_JUNE\\FakeDetection\\GAN\\ProGAN'

# dataDir = 'S:\\PGM Image Phylogeny\\Codes\\gcn_copy1\\INPUTMATFILES_JUNE\\FakeDetection\\GAN\\FaceForensics'
 
#dataDir = 'S:\\PGM Image Phylogeny\\Codes\\gcn_copy1\\INPUTMATFILES_ArbitnumFaceImages'

# dataDir = 'S:\\PGM Image Phylogeny\\Codes\\gcn_copy1\\INPUTMATFILES_Geom'
 
# dataDir = 'S:\\PGM Image Phylogeny\\Codes\\gcn_copy1\\INPUTMATFILES_WVU\\3_4_5'
 ## TIFS RESULTS BELOW final
#dataDir = 'S:\\PGM Image Phylogeny\\Codes\\gcn_copy1\\INPUTMATFILES_WVU\\NEWTRAINING_ONLYIMAGES_TESTADJNEW\\1_2_6' # NEWTRAINING
#dataDir = 'S:\\PGM Image Phylogeny\\Codes\\gcn_copy1\\INPUTMATFILES_UCID_PHOTO'
#dataDir = 'S:\\PGM Image Phylogeny\\Codes\\gcn_copy1\\INPUTMATFILES_NDFI\\Set1'
#dataDir = 'S:\\DeepFakeDetection\\Codes\\INPUTMATFILESGCNWITHMORPH'
# dataDir = 'S:\\DeepFakeDetection\\Codes\\INPUTMATFILESGCN_ONLYMORPH'
# dataDir = 'S:\\FaceMAD\\INPUTMATFILES\\INPUTMATFILESGCN_AMSL_DIFFERENTIALMORPH'
#dataDir = 'S:\\PGM Image Phylogeny\\Codes\\gcn_copy1\\INPUTMATFILES_UCID_ICASSP'
#dataDir = 'S:\\PGM Image Phylogeny\\Codes\\gcn_copy1\\INPUTMATFILES_UNCWMORPH'
 
dataDir = 'S:\\PGM Image Phylogeny\\Codes\\gcn_copy1\\INPUTMATFILES_DEEPLEARNING\\BEAUTYGLOW'
 ## variables to store matfiles

#A = sio.loadmat(join(dataDir,'adj_MST_PRNUINIT.mat')) #,squeeze_me=True adj_NEW
#A = sio.loadmat(join(dataDir,'adj.mat')) #,squeeze_me=True adj_NEW
# B = sio.loadmat(join(dataDir,'PixelFeatures_Res.mat'))
#B = sio.loadmat(join(dataDir,'PixelFeatures_Res.mat')) #PixelFeatures.mat ResNetFeatures.mat PixelFeatures_Ori PixelFeatures_Res PixelFeatures_psi0_7
# B = sio.loadmat(join(dataDir,'PRNUFeatures.mat'))
# B = sio.loadmat(join(dataDir,'ResNetFeatures_NORM.mat'))
#############################################################################################
#A = sio.loadmat(join(dataDir,'adj_PRNU_BALANCEDTRAIN.mat')) # adj_PIXEL adj_PRNU
#B = sio.loadmat(join(dataDir,'PRNUFeatures_BALANCEDTRAIN.mat')) #PRNUFeatures PixelFeatures
#C = sio.loadmat(join(dataDir,'train_mask_BALANCEDTRAIN.mat'))
#D = sio.loadmat(join(dataDir,'test_mask_BALANCEDTRAIN.mat'))
#E = sio.loadmat(join(dataDir,'val_mask_BALANCEDTRAIN.mat'))
#F = sio.loadmat(join(dataDir,'y_train_BALANCEDTRAIN.mat'))
#G = sio.loadmat(join(dataDir,'y_test_BALANCEDTRAIN.mat'))
#H = sio.loadmat(join(dataDir,'y_val_BALANCEDTRAIN.mat'))
############################################################################################
A = sio.loadmat(join(dataDir,'adj.mat')) 
B = sio.loadmat(join(dataDir,'PixelFeatures.mat')) 
C = sio.loadmat(join(dataDir,'train_mask.mat'))
D = sio.loadmat(join(dataDir,'test_mask.mat'))
E = sio.loadmat(join(dataDir,'val_mask.mat'))
F = sio.loadmat(join(dataDir,'y_train.mat'))
G = sio.loadmat(join(dataDir,'y_test.mat'))
H = sio.loadmat(join(dataDir,'y_val.mat'))
###################################################################################################

adj = A.get('ADJ_ALL_sp') # ADJ_ALL_sp    ADJ_ALL_sp_PIXEL
#features_i = sp.csr_matrix(B['Features_PIXEL']) #B.get('Features_PIXEL') #    #B.get('Features_PIXEL')#sp.csr_matrix(B['features_RESNET'])# Features_PIXEL PixelFeatures
features_i = sp.csr_matrix(B['Features_PIXEL']) # Features_PRNU   Features_PIXEL
#features_i = B.get('Features_PRNU')
#features_i = B.get('Features_PIXEL')
train_mask = C.get('train_mask')
test_mask = D.get('test_mask')
val_mask = E.get('val_mask')
y_train = F.get('y_train')
y_test = G.get('y_test')
y_val = H.get('y_val')

features = features_i.astype(float)

#  Load reduced subset
#dataDir = 'S:\\PGM Image Phylogeny\\Codes\\gcn_copy1\\'
#A = sio.loadmat(join(dataDir,'adj_red.mat')) #,squeeze_me=True
#B = sio.loadmat(join(dataDir,'features_red.mat'))
#C = sio.loadmat(join(dataDir,'train_mask_red.mat'))
#D = sio.loadmat(join(dataDir,'test_mask_red.mat'))
#E = sio.loadmat(join(dataDir,'val_mask_red.mat'))
#F = sio.loadmat(join(dataDir,'y_train_red.mat'))
#G = sio.loadmat(join(dataDir,'y_test_red.mat'))
#H = sio.loadmat(join(dataDir,'y_val_red.mat'))
#
#adj = A.get('adj_red')
#features_i = sp.csr_matrix(B['features_red']) #B.get('features')
#train_mask = C.get('train_mask_red')
#test_mask = D.get('test_mask_red')
#val_mask = E.get('val_mask_red')
#y_train = F.get('y_train_red')
#y_test = G.get('y_test_red')
#y_val = H.get('y_val_red')
#
#features = features_i.astype(float)

    
#arrays={}
#f=h5py.File(join(dataDir,'features.mat'))
#for k, v in f.items():
#    arrays[k] = np.array(v)
#test_idx_reorder = random.sample(range(1001, 1501), 500)
#test_idx_range = np.sort(test_idx_reorder)
#
#
#features_list = np.ndarray.tolist(features)
#
#for iter in range(len(test_idx_reorder)):
#   features_list[test_idx_reorder(iter)] = features_list[test_idx_range(iter)]
#   
#features_revert = np.array(features_list)
#
#features = features_revert
#    
#labels_trall = np.vstack((y_train, y_val))
#labels = np.vstack((labels_trall, y_test))
#labels[test_idx_reorder, :] = labels[test_idx_range, :]
#for iter in range(train_mask.shape[1]):
#    train_mask[0][iter] = bool(train_mask[0][iter])
#    test_mask[0][iter] = bool(test_mask[0][iter])
#    val_mask[0][iter] = bool(val_mask[0][iter])

# Load data
#adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)

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
#plt.savefig('S:\\PGM Image Phylogeny\\Results\\GCN_PREDLABELS\\Original_GCN_Cheby_degree3_FakeDetection\\BalancedData\\Accuracy plot.png')

plt.figure()
plt.plot(epoch_cnt, train_Err, 'r--')
plt.plot(epoch_cnt, val_Err, 'b-')
plt.legend(['Training Loss', 'Validation Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
#plt.savefig('S:\\PGM Image Phylogeny\\Results\\GCN_PREDICTIONS\\GAN\\Loss plot Cheby4_ITER100_PRNU_PSI1_0GAN.png')
#plt.savefig('S:\\PGM Image Phylogeny\\Results\\GCN_PREDICTIONS\\GCNCHeby_3\\Loss plot Cheby3_ITER100_PRNU_FaceForensics_MOREBONAFIDE')

# Testing
test_cost, test_acc, test_duration, pred_labs, prediction_scores, test_emb = evaluate(features, support, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
#sio.savemat('S:\\PGM Image Phylogeny\\Results\\GCN_PREDLABELS\\Original_GCN_Cheby_degree3_FakeDetection\\BalancedData_trainandTest\\PREDS_BalancedTestandTrain_PSI0_7_DEGREE4.mat',{'pred':pred_labs})
#sio.savemat('S:\\PGM Image Phylogeny\\Results\\GCN_PREDLABELS\\Original_GCN_Cheby_degree3_FakeDetection\\DeepFakesonly_psi_0_5\\PREDS_PSI0_5_DEGREE3_Unbalanced.mat',{'pred':pred_labs})

# sio.savemat('S:\\PGM Image Phylogeny\\Results\\GCN_PREDICTIONS\\GAN\\Predicted GCN_Chebydegree4_ITER100_PRNU_PSI1_0GAN.mat',{'pred':pred_labs})
# sio.savemat('S:\\PGM Image Phylogeny\\Results\\GCN_PREDICTIONS\\GCNCHeby_3\\Predicted GCN_Chebydegree3_ITER100_PRNU_FaceForensics_MOREBONAFIDE.mat',{'pred':pred_labs})

# sio.savemat('S:\PGM Image Phylogeny\Results\GCN_DeepFakes_ROC\\StyleGAN_PSI0_5_preds.mat',{'pred':pred_labs})
# sio.savemat('S:\PGM Image Phylogeny\Results\GCN_DeepFakes_ROC\\StyleGAN_PSI0_5_scores.mat',{'scores':prediction_scores}) ## For ROC curves

# sio.savemat('S:\\PGM Image Phylogeny\\Results\\GCN_PREDICTIONS\\GCNCHeby_3\\ArbitNumFaceImages\\Predicted GCN_Chebydegree3_ITER100_IPTPRNUINIT_PIXEL_ArbitNumFacetestimages_preds_ADJNORMDIFFASYMM.mat',{'pred':pred_labs})

#.savemat('S:\\PGM Image Phylogeny\\Results\\GCN_PREDICTIONS\\GCNCHeby_3\\Predicted GCN_Chebydegree3_ITER100_IPTPRNUINIT_PIXEL_FINGERPRINT.mat',{'pred':pred_labs})

# sio.savemat('S:\\PGM Image Phylogeny\\Results\\GCN_PREDICTIONS\\GCNCHeby_3\\Geom\\Predicted GCN_Chebydegree3_ITER100_IPTPRNUINIT_PIXEL_Geom_preds_ADJNORMDIFFASYMM.mat',{'pred':pred_labs})

# sio.savemat('S:\\PGM Image Phylogeny\\Results\\GCN_PREDICTIONS\\GCNCHeby_3\\WVU\\3_4_5\\Predicted GCN_Chebydegree3_ITER100_IPTPRNUINIT_PIXEL_3_4_5_preds_ADJNORMDIFFASYMM.mat',{'pred':pred_labs})

## TIFS RESULTS below final
##sio.savemat('S:\\PGM Image Phylogeny\\Results\\GCN_PREDICTIONS\GCNCHeby_3\\WVU\\NEWTRAINING_ONLYIMAGES_TESTADJNEW\\1_2_6\\Predicted GCN_Chebydegree3_ITER100_IPTPRNUINIT_PIXEL_1_2_6_preds_ADJNORMDIFFASYMM.mat',{'pred':pred_labs}) # NEWTRAINING

#sio.savemat('S:\\PGM Image Phylogeny\\Results\\GCN_PREDICTIONS\\GCNCHeby_3\\UCID_Photo\\Predicted GCN_Chebydegree3_ITER100_IPTPRNUINIT_PIXEL_UCID_Photo_preds_ADJNORMDIFFASYMM.mat',{'pred':pred_labs})
#sio.savemat('S:\\PGM Image Phylogeny\\Results\\GCN_PREDICTIONS\\GCNCHeby_3\\NDFI\\Set1\\Predicted GCN_Chebydegree3_ITER100_IPTPRNUINIT_PIXEL_NDFI_preds_ADJNORMDIFFASYMM.mat',{'pred':pred_labs})

## DeepFake
#sio.savemat('S:\\DeepFakeDetection\\Results\\RESULTSWITHMORPHS\\Adj_PRNU_Feats_PRNU_preds_BALANCEDTRAINTEST_CHEBDEG_7.mat',{'pred':pred_labs})
#sio.savemat('S:\\DeepFakeDetection\\Results\\RESULTSWITHMORPHS\\Adj_PRNU_Feats_PRNU_scores_BALANCEDTRAINTEST_CHEBDEG_7.mat',{'scores':prediction_scores}) ## For ROC curves
#sio.savemat('S:\\DeepFakeDetection\\Results\\RESULTSWITHMORPHS\\Adj_PRNU_Feats_PRNU_emb_BALANCEDTRAINTEST_CHEBDEG_7.mat',{'embs':test_emb}) ## For embeddings

#sio.savemat('S:\\DeepFakeDetection\\Results\\RESULTSGCNONLYMORPHS\\Adj_PRNU_Feats_PRNU_preds_BALANCEDTRAINTEST_CHEBDEG_5.mat',{'pred':pred_labs})
#sio.savemat('S:\\DeepFakeDetection\\Results\\RESULTSGCNONLYMORPHS\\Adj_PRNU_Feats_PRNU_scores_BALANCEDTRAINTEST_CHEBDEG_5.mat',{'scores':prediction_scores}) ## For ROC curves
#sio.savemat('S:\\DeepFakeDetection\\Results\\RESULTSGCNONLYMORPHS\\Adj_PRNU_Feats_PRNU_emb_BALANCEDTRAINTEST_CHEBDEG_5.mat',{'embs':test_emb}) ## For embeddings

#sio.savemat('S:\\FaceMAD\RESULTS\\RESULTSGCN_DIFFERENTIAL_AMSL\\Adj_PRNU_Feats_PRNU_preds_BALANCEDTRAINTEST_CHEBDEG_5.mat',{'pred':pred_labs})
#sio.savemat('S:\\FaceMAD\RESULTS\\RESULTSGCN_DIFFERENTIAL_AMSL\\Adj_PRNU_Feats_PRNU_scores_BALANCEDTRAINTEST_CHEBDEG_5.mat',{'scores':prediction_scores}) ## For ROC curves
#sio.savemat('S:\\FaceMAD\RESULTS\\RESULTSGCN_DIFFERENTIAL_AMSL\\Adj_PRNU_Feats_PRNU_emb_BALANCEDTRAINTEST_CHEBDEG_5.mat',{'embs':test_emb}) ## For embeddings

#sio.savemat('S:\\PGM Image Phylogeny\\Results\\GCN_PREDICTIONS\\GCNCHeby_3\\UCID_ICASSP\\Predicted GCN_Chebydegree3_UCIDICASSP.mat',{'pred':pred_labs})

sio.savemat('S:\\CITER_IMAGEPHYLOGENY\\GCNRESULTS\\DEEPLEARNING\\BEAUTYGLOW\\Adj_PRNU_Feats_PIXEL_preds_CHEBDEG_3.mat',{'pred':pred_labs})
sio.savemat('S:\\CITER_IMAGEPHYLOGENY\\GCNRESULTS\\DEEPLEARNING\\BEAUTYGLOW\\Adj_PRNU_Feats_PIXEL_scores_CHEBDEG_3.mat',{'scores':prediction_scores}) ## For ROC curves
sio.savemat('S:\\CITER_IMAGEPHYLOGENY\\GCNRESULTS\\DEEPLEARNING\\BEAUTYGLOW\\Adj_PRNU_Feats_PIXEL_emb_CHEBDEG_3.mat',{'embs':test_emb}) ## For embeddings
