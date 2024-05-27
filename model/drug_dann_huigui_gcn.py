# -*- coding: utf-8 -*-
import imp
from pyexpat import features
import warnings

from pyparsing import nums
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from doctest import OutputChecker
from re import S, X
from tkinter import SEL_LAST, W
from turtle import shape
from typing_extensions import Self
from unicodedata import name
# from matplotlib import units
# from matplotlib.pyplot import axis
import numpy as np
from sklearn.feature_selection import SelectFdr
import tensorflow as tf
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
#from tensorflow.python.ops.math_ops import sigmoid
#from tensorflow.python.ops.variables import trainable_variables
from utils.utils import plot_loss
from utils.utils import plot_accuracy

from utils.utils import AverageMeter
from utils.utils import make_summary


from model.GRL import GRL
from utils.utils import grl_lambda_schedule
from utils.utils import learning_rate_schedule
import datetime
from scipy.spatial.distance import pdist, squareform

from config.config import config
from tensorflow import keras as K
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from scipy.stats import pearsonr

from gcn_model.utils import *
from gcn_model.file_utils import *
from gcn_model.layers import *

from tqdm import tqdm 
#shuffle_data
def shuffle_aligned_list(data):
    """Shuffle arrays in a list by shuffling each array identically."""
    num = data.shape[0]
    p = np.random.permutation(num)
    data_shufflie = data[p]
    return data_shufflie

def batch_generator(data, batch_size, shuffle=True):
    """Generate batches of data.

    Given a list of array-like objects, generate batches of a given
    size by yielding a list of array-like objects corresponding to the
    same slice of each input.
    """
    if shuffle:
        data = shuffle_aligned_list(data)

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= len(data[:,0]):
            
            batch_count = 0

            if shuffle:
                data = shuffle_aligned_list(data)

        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield data[start:end]       

def corruption(x,noise_factor = 0.2):
    noise_vector = x+noise_factor*np.random.randn(x.shape)

    noise_vector = np.clip(noise_vector,0.,1.)
    return noise_vector


def extra_same_elem(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    iset = set1.intersection(set2)
    return list(iset)

def make_SSP(drug_fingerprint):
        ssp_mat = 1. - squareform(pdist(drug_fingerprint.values, 'jaccard'))
        return pd.DataFrame(ssp_mat, index=drug_fingerprint.index, columns=drug_fingerprint.index)

def make_batch_ssp(batch_response_data,pd_ssp):
    batch_drug_name = batch_response_data[:][:,0]
    batch_ssp = pd_ssp.loc[batch_drug_name]
    return batch_ssp.values

def make_batch_ep(batch_response_data,pd_ep,min_max_scaler):
    batch_gene_name = batch_response_data[:][:,1]
    batch_gene_name=[str(i) for i in batch_gene_name]
    batch_ep = pd_ep[batch_gene_name]
    return min_max_scaler.transform(batch_ep.values.T)

def make_batch_labels(batch_response_data):
    batch_onehot_labels = []
    batch_labels = batch_response_data[:][:,3]
    for i in range(batch_labels.shape[0]):
        if batch_labels[i] =='Sensitivity':
            batch_onehot_labels.append([1,0])
        else:
            batch_onehot_labels.append([0,1])
            
    return np.array(batch_onehot_labels)

def make_batch_fp(batch_response_data,pd_fp):
    batch_drug_name = batch_response_data[:][:,2]
    #print(batch_drug_name)
    batch_fp = pd_fp.loc[batch_drug_name]
    return batch_fp.values
def make_batch_druglist(batch_response_data):
    batch_drug_name = batch_response_data[:][:,2]
    return list(batch_drug_name)

def make_batch_regression(batch_response_data):
    batch_labels =batch_response_data[:][:,3]
    return batch_labels

def make_batch_regression1(batch_response_data,min_max_scaler):
    batch_labels =batch_response_data[:][:,3]
    return min_max_scaler.transform(batch_labels.reshape([-1,1]))


def make_test_ep(pd_re,pd_ep,min_max_scaler):
    cell_name = list(pd_re['Primary Cell Line Name'])
    cell_name=[str(i) for i in cell_name]
    batch_ep = pd_ep[cell_name]
    return min_max_scaler.transform(batch_ep.values.T)
def make_test_fp(pd_re,pd_fp):
    drug_name = list(pd_re['Compound'])
    #print(batch_drug_name)
    fp = pd_fp.loc[drug_name]
    return fp.values
def make_test_labels(pd_re,min_max_scaler):
    labels = np.array(pd_re['IC50 (uM)'])
    return min_max_scaler.transform(labels.reshape([-1,1]))


    

class Drug_DANN(object):
    def __init__(self,config,SAE_Newcheck,drug_dict,drug_smile,smile_graph):
        # 閿熸枻鎷峰閿熸枻鎷烽敓鏂ゆ嫹閿熸枻鎷烽敓鏂ゆ嫹
        self.cfg = config
        self.Newcheck = SAE_Newcheck
        self.drug_list = drug_dict
        self.drug_smiles = drug_smile
        self.smile_graph = smile_graph

        # 閿熸枻鎷烽敓鏂ゆ嫹閿熸枻鎷烽敓绉稿槈浼欐嫹閿燂拷?

        self.grl_lambd = tf.placeholder(tf.float32, [])                         # GRL
        self.learning_rate = tf.placeholder(tf.float32, [])                     #
        
        self.response_labels = tf.placeholder(tf.float32, name ='ic50') #閸ョ偛缍婇弽鍥╊劮閿燂拷?
        self.domain_labels = tf.placeholder(tf.float32, shape=(None, 2)) #閸╃喎鍨庣猾缁�5�7垼閿燂拷?

        #�??
        nums_support =1
        self.gcn_placeholders ={
            
            'support': [tf.sparse_placeholder(tf.float32) for i in range(nums_support)],
            'feats': tf.sparse_placeholder(tf.float32, shape=tf.constant((1552,44), dtype=tf.int64)), 
            #'labels_mask': tf.placeholder(tf.int32),
            'dropout': tf.placeholder_with_default(0., shape=()),
            'num_features_nonzero': tf.placeholder(tf.int32)
            }

        #regressisor
        self.w1 = tf.get_variable('w1',shape=[768,256], initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        self.b1 = tf.Variable(np.zeros(256).astype(np.float32))
        self.w2 = tf.get_variable('w2',shape=[256,128], initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        self.b2 = tf.Variable(np.zeros(128).astype(np.float32))
        self.w3 = tf.get_variable('w3',shape=[128,1], initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        self.b3 = tf.Variable(np.zeros(1).astype(np.float32))
        
        self.build_DANN()
        
        self.response_cls_loss =  tf.reduce_mean(tf.square(self.response_labels-self.response_cls))
        self.domain_cls_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.domain_labels,
                                                                        logits=self.domain_cls))
        #
        self.loss = self.response_cls_loss+self.domain_cls_loss

        # 閿熸枻鎷烽敓钘夌簿閿熸枻鎷穉cc閿熸枻鎷穉uc_roc,f1,precsicon,recall
        #self.y_true= tf.argmax(self.response_labels, 1)
        #self.y_pred= tf.argmax(self.response_cls, 1)
        #correct_label_pred = tf.equal(tf.argmax(self.response_labels, 1), tf.argmax(self.response_cls, 1))
        #self.acc = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))

        # 閿熸枻鎷烽敓鏂ゆ嫹妯￠敓閰垫唻鎷烽敓鏂ゆ嫹閿熸枻鎷烽敓鏂ゆ嫹閿熸枻鎷烽敓鏂ゆ嫹閿燂拷?
        self.saver_save = tf.train.Saver(max_to_keep=100)  #

        # 閿熸枻鎷烽敓鏂ゆ嫹瀛︿範閿熸枻�??
        self.global_step = tf.Variable(tf.constant(0),trainable=False)
        #self.process = self.global_step / self.cfg.epoch

        # 閿熸枻鎷峰閿熸枻鎷烽敓鑴氫紮鎷烽敓鏂ゆ�?
        #self.optimizer = MomentumOptimizer(self.learning_rate, momentum=self.cfg.momentum_rate)
        self.optimizer2 = tf.train.MomentumOptimizer(self.learning_rate, momentum=self.cfg.momentum_rate)
        self.optimizer1 = tf.train.AdamOptimizer(self.learning_rate)
        #var_list = [v.name() for v in tf.trainable_variables()]
        self.train_op ={}
        self.train_op['0'] = self.optimizer1.minimize(self.loss,global_step=self.global_step)
        self.train_op['1'] = self.optimizer2.minimize(self.loss,global_step =self.global_step)
            
        
    def my_init1(self,shape,dtype=None,partition_info=None):
        return tf.convert_to_tensor(self.Newcheck.get_tensor('W_Encoder1'),tf.float32)
        
    def my_init2(self,shape,dtype=None,partition_info=None):
        return tf.convert_to_tensor(self.Newcheck.get_tensor('b_Encoder1'),tf.float32)
        
    def my_init3(self,shape,dtype=None,partition_info=None):
        return tf.convert_to_tensor(self.Newcheck.get_tensor('W_Encoder2'),tf.float32)
        
    def my_init4(self,shape,dtype=None,partition_info=None):
        return tf.convert_to_tensor(self.Newcheck.get_tensor('b_Encoder2'),tf.float32)
        
    def my_init5(self,shape,dtype=None,partition_info=None):
        return tf.convert_to_tensor(self.Newcheck.get_tensor('W_Encoder3'),tf.float32)
        
    def my_init6(self,shape,dtype=None,partition_info=None):
        return tf.convert_to_tensor(self.Newcheck.get_tensor('b_Encoder3'),tf.float32)        
              
    def build_DANN(self):
        """
        :return:
        """
        
        # 閿熸枻鎷烽敓鏂ゆ嫹DANN妯￠敓鏂ゆ嫹閿熸枻鎷烽敓鏂ゆ�? K.layers.Input(shape = 閿熸枻鎷?16759閿熸枻鎷烽敓鏂ゆ�? 婧愰敓鏂ゆ嫹閿熸枻鎷风洰閿熸枻鎷烽敓鏂ゆ嫹鎷奸敓鏂ゆ�? 閿熸澃鐧告嫹閿熸枻鎷穊atch*2閿熸枻鎷?16017閿熸枻鎷?
        self.gene_source_input = K.layers.Input(shape=self.cfg.gene_input_shape,batch_size =16,name="source_gene_input")
        #print(self.gene_source_input.shape)
        self.gene_target_input = K.layers.Input(shape=self.cfg.gene_input_shape,batch_size =16,name="target_gene_input")
        #print(self.gene_target_input.shape)
        self.gene_input = K.layers.Concatenate(axis=0,name="gene_input")([self.gene_source_input,self.gene_target_input])
        # 閿熸枻鎷烽敓鏂ゆ嫹閿熸枻鎷峰�?
        print(self.gene_input.shape)
        share_feature = self.featur_extractor(self.gene_input)
        print(share_feature.shape)
        # 閿熸枻鎷烽敓楗轰紮鎷烽敓琛楃櫢鎷烽敓鏂ゆ嫹閿熸枻鎷烽敓鏂ゆ嫹涓烘簮閿熸枻鎷烽敓鏂ゆ嫹閿熸枻鎷烽敓鏂ゆ嫹閿熸枻鎷烽敓鏂ゆ嫹鐩敓鏂ゆ嫹閿熸枻鎷烽敓鏂ゆ嫹閿熸枻鎷烽敓鏂ゆ嫹閿熸枻鎷?
        source_feature,target_feature = \
            K.layers.Lambda(tf.split, arguments={'axis': 0, 'num_or_size_splits': 2})(share_feature)
        source_feature = K.layers.Lambda(lambda x:x,name="source_feature")(source_feature)
        print(source_feature.shape)
        print(source_feature)
        # 閿熸枻鎷峰彇閿熸枻鎷风閿熸枻鎷烽敓鏂ゆ嫹閿熸枻鎷烽敓鏂ゆ嫹閿熸枻鎷烽敓鏂ゆ嫹閿熸枻鎷烽敓鏂ゆ嫹閿熸枻鎷烽敓�???
        self.drug_feature = self.build_drug_gcn_feature()
        print(self.drug_feature.shape)
        self.response_cls = self.build_response_classify_model(source_feature)
        #print()
        
        self.domain_cls = self.build_domain_classify_model(share_feature)
        
        
#閿熸枻鎷烽敓鏂ゆ嫹閿熸枻鎷峰彇閿熸枻鎷烽敓鎹疯鎷烽敓鏂ゆ嫹
    def featur_extractor(self,gene_input):
        """
        """
        #
        
        print(self.Newcheck.get_tensor('W_Encoder1').shape)
        print(self.Newcheck.get_tensor('b_Encoder1').shape)
        x = K.layers.Dense(2048,kernel_initializer=self.my_init1,
                                bias_initializer = self.my_init2, activation='sigmoid')(gene_input)
        
        x = K.layers.Dense(1024,kernel_initializer=self.my_init3,
                                bias_initializer = self.my_init4, activation='sigmoid')(x)

        #bn
        
        x = K.layers.Dense(512,kernel_initializer=self.my_init5,
                                bias_initializer = self.my_init6, activation='sigmoid')(x)
        
        return x
#閸ョ偛缍婇敓�???
    def build_response_classify_model(self,gene_classify_feature):
        """

        """
        
            # 閿熺瓔寤洪敓鏂ゆ嫹绛鹃敓鏂ゆ嫹閿熸枻鎷烽敓鏂ゆ嫹
            
        #x = K.layers.Lambda(lambda x:x,name="response_classify_feature")(gene_classify_feature)
        #print(x.shape)
        #閿熸枻鎷疯嵂閿熸枻鎷烽敓鏂ゆ嫹閿熸枻鎷烽敓鏂ゆ嫹閿熸枻鎷烽敓鏂ゆ嫹閿熸枻鎷烽敓鏂ゆ嫹閿熸枻鎷烽敓鏂ゆ嫹鎷奸敓鏂ゆ�?512+256
        x  = tf.concat([gene_classify_feature,self.drug_feature],axis = 1)
        x = self.fcc_relu(x,self.w1,self.b1)
        x = self.fcc_relu(x,self.w2,self.b2)
        x = self.fcc_sig(x,self.w3,self.b3)

        #x = K.layers.Dense(256,kernel_initializer=K.initializers.he_normal(seed=None),
                                    #bias_initializer = K.initializers.Constant(value=0.01),activation='relu'
                                    #)(x)
        #x = K.layers.Dense(128,kernel_initializer=K.initializers.he_normal(seed=None),
                                    #bias_initializer = K.initializers.Constant(value=0.01),activation='relu'
                                    #)(x)
        #x = K.layers.BatchNormalization()(x,training=False)
        #x = tf.layers.dense(x,units=1,kernel_initializer=tf.random_normal_initializer(),bias_initializer=tf.zeros_initializer(),activation='relu')
        #x = K.layers.Dense(1,kernel_initializer=K.initializers.he_normal(seed=None),
                                    #bias_initializer = K.initializers.Constant(value=0.01),activation='sigmoid',
                            #name = "response_classify_pred")(x)
        return x

#閸╃喎鍨庣猾璇叉�?
    def build_domain_classify_model(self,domain_classify_feature):
        """
        """
        # 閿熺瓔寤洪敓鏂ゆ嫹閿熸枻鎷烽敓鏂ゆ嫹閿燂�??
    
        x = GRL(domain_classify_feature,self.grl_lambd)
        
        x = K.layers.Dense(256,kernel_initializer=K.initializers.he_normal(seed=None),
                                bias_initializer = K.initializers.Constant(value=0.01),activation='relu')(x)
        
        x = K.layers.Dense(128,kernel_initializer=K.initializers.he_normal(seed=None),
                                bias_initializer = K.initializers.Constant(value=0.01),activation='relu')(x)
        #K.backend.set_learning_phase(1)
        #print(type(x))
        #x = K.layers.Dropout(0.5)(x)
        x = K.layers.BatchNormalization()(x,training=False)
        x = K.layers.Dense(2,kernel_initializer=K.initializers.he_normal(seed=None),
                                bias_initializer = K.initializers.Constant(value=0.01)
                        ,name="domain_classify_pred")(x)
        return x
    #
    def build_drug_gcn_feature(self):
        
        Output  = ConvolutionalLayer(input_dim=44,output_dim=32,placeholders=self.gcn_placeholders,dropout=True,sparse_inputs=True,featureless=False,activation=tf.nn.relu)(self.gcn_placeholders['feats'])
        print(Output.shape)
        Output = tf.reshape(Output,[16,3104])
        print(Output.shape)
        drug_fc = K.layers.Dense(256,kernel_initializer=K.initializers.he_normal(seed=None),
                                bias_initializer = K.initializers.Constant(value=0.01))
        Output = drug_fc(Output)
        return Output
    def fcc_relu(self,x,W,b):
        Output = tf.nn.relu(tf.matmul(x,W)+b)
        return Output
    def fcc_sig(self,x,W,b):
        Output = tf.nn.sigmoid(tf.matmul(x,W)+b)
        return Output
        
    def train(self,train_source_datagen,train_target_datagen,val_datagen,train_iter_num,val_iter_num,pd_drug_ccle,pd_drug_gdsc,common_ccle,common_gdsc,min_max_scaler_common,min_max_scaler_labels):    
    #def train(self,train_source_datagen,train_target_datagen,val_datagen,train_iter_num,val_iter_num,pd_drug_ccle,pd_drug_gdsc,common_ccle,common_gdsc,min_max_scaler_labels):
        #
        time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        checkpoint_dir = os.path.join(self.cfg.checkpoints_dir,time)
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        log_dir = os.path.join(self.cfg.logs_dir, time)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        result_dir = os.path.join(self.cfg.result_dir, time)
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

        self.cfg.save_config(time)

       
        train_loss_results = []                    
        train_gene_cls_loss_results = []           
        train_domain_cls_loss_results = []          
        train_accuracy_results = []                 

        
        val_ep = []
        val_loss_results = []                     
        val_drug_cls_loss_results = []           
        val_domain_cls_loss_results = []       
        val_accuracy_results = []                 
        val_acc_max = 10                           
        test_max_r = 0


        
        #閿熸枻鎷烽敓鏂ゆ嫹鎸囬敓鏂ゆ�?:auc_roc,f1
        #auc_roc = metrics.roc_auc_score(test_y,prodict_prob_y)
        
        y_batch_labels = [] #娣囨繂鐡ㄩ弫鐗堝祦閺嶅洨�??
        y_pred_batch_labels = [] #娣囨繂鐡ㄦ０鍕ゴ閺嶅洨�??
#         domain_batch_labels =[]
#         domain_pred_lables =[]
        op_num = '{0}'.format(0)

        with tf.Session() as sess:
            # 閿熸枻鎷峰閿熸枻鎷烽敓鏂ゆ嫹閿熸枻�??
            sess.run(tf.global_variables_initializer())

            # 閿熸枻鎷烽敓鏂ゆ嫹棰勮閿熸枻鎷锋ā閿熸枻�??
#             if pre_model_path is not None:              # pre_model_path閿熶茎纰夋嫹鍧€鍐欓敓鏂ゆ嫹.ckpt
#                 saver_restore = tf.train.import_meta_graph(pre_model_path+".meta")
#                 saver_restore.restore(sess,pre_model_path)
#                 print("restore model from : %s" % (pre_model_path))

            self.merged = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter(log_dir, sess.graph)
            #self.writer1 = tf.summary.FileWriter(os.path.join("./tf_dir"), sess.graph)

            print('\n----------- start to train -----------\n')
            #鍏ㄩ敓琛楄鎷烽敓鏂ゆ嫹
            total_global_step = self.cfg.epoch * train_iter_num
            for ep in np.arange(100):
                # if(ep>1):
                #         op_num = '{0}'.format(1)           
                # 閻€劌鍨垫慨瀣閸戣姤鏆熼崚婵嗩潗閸栨潝oss
                epoch_loss_avg = AverageMeter()
                epoch_drug_cls_loss_avg = AverageMeter()
                epoch_domain_cls_loss_avg = AverageMeter()
                epoch_accuracy = AverageMeter()

               
                print('Epoch {}/{}'.format(ep+1, 100))
                
                batch_domain_labels = np.vstack([np.tile([1., 0.], [self.cfg.batch_size, 1]),
                                           np.tile([0., 1.], [self.cfg.batch_size, 1])])
               
                for i in np.arange(1,train_iter_num+1):
                    if(i%1000==0):
                        print('batch{}/{}'.format(i,train_iter_num+1))
                    batch_gdsc_data = train_source_datagen.__next__()#train_source_datagen.next_batch()
                    batch_ccle_data = train_target_datagen.__next__()#train_target_datagen.next_batch()

                    #batch_fp_gdsc=make_batch_fp(batch_gdsc_data,pd_drug_gdsc)
                    batch_drug_name =make_batch_druglist(batch_gdsc_data)#药物名字列表
                    
                    adj_matrix, node_feats, b_idx = load_batch_graph_data(batch_drug = batch_drug_name,num_nodes=1536,num_graphs=16,dim_feats=44,smile_graph=self.smile_graph,drug_smile=self.drug_smiles,drug_dict=self.drug_list)
                    
                    suppot = [preprocess_adj(adj_matrix,True,True)]
                    features = process_features(node_feats)
                  
                    GCN_DICT = build_dictionary_GCN(feats=features,support=suppot,placeholders=self.gcn_placeholders)
            
                    #print(batch_fp_gdsc)
                    
                    
                    batch_ccle_ep = make_batch_ep(batch_gdsc_data,common_ccle,min_max_scaler_common)
                    batch_gdsc_ep =make_batch_ep(batch_gdsc_data,common_gdsc,min_max_scaler_common)

                    #
                    batch_gdsc_labels = make_batch_regression1(batch_gdsc_data,min_max_scaler_labels)
                    #batch_gdsc_labels = make_batch_regression(batch_gdsc_data)
                    #print(batch_gdsc_data)
                    #
                    global_step = (ep-1)*train_iter_num + i
                    process = global_step * 1.0 / total_global_step
                    leanring_rate = learning_rate_schedule(process,self.cfg.init_learning_rate)
                    grl_lambda = grl_lambda_schedule(process)

                    # 
                    trian_dict ={}
                    fc_dict={self.gene_source_input:batch_gdsc_ep,
                                             self.gene_target_input:batch_ccle_ep,
                                             self.response_labels:batch_gdsc_labels,
                                             self.domain_labels:batch_domain_labels,
                                             self.learning_rate:leanring_rate,
                                             self.grl_lambd:grl_lambda}
                    trian_dict.update(fc_dict)
                    trian_dict.update(GCN_DICT)
                    op,train_loss,train_response_cls_loss,train_domain_cls_loss,y_pred_response_batch = \
                        sess.run([self.train_op['0'],self.loss,self.response_cls_loss,self.domain_cls_loss,self.response_cls],
                                  feed_dict= trian_dict)
                    
                    
                    #print(batch_gdsc_labels)
                    #print(y_pred_batch_labels)

                    origin_y_batch_labels = min_max_scaler_labels.inverse_transform(batch_gdsc_labels)
                    origin_y_pred_batch_labels = min_max_scaler_labels.inverse_transform(y_pred_response_batch)
                    y_batch_labels+= list(origin_y_batch_labels)#gdsc_true_labels
                    y_pred_batch_labels+= list(origin_y_pred_batch_labels)#gdsc_pred_labels
                    
                    epoch_loss_avg.update(train_loss,1)
                    epoch_drug_cls_loss_avg.update(train_response_cls_loss,1)
                    epoch_domain_cls_loss_avg.update(train_domain_cls_loss,1)
                    #epoch_accuracy.update(train_acc,1)

                train_loss_results.append(epoch_loss_avg.average)
                train_gene_cls_loss_results.append(epoch_drug_cls_loss_avg.average)
                train_domain_cls_loss_results.append(epoch_domain_cls_loss_avg.average)
                train_accuracy_results.append(epoch_accuracy.average)

                #print(y_batch_labels)
                #print(y_pred_batch_labels)
                print('Epoch {}/{} train_r2 {} train_rmse {} train_drug_cls_loss {} train_domain_cls_loss {} total_loss {}'.format(ep+1, self.cfg.epoch,r2_score(y_batch_labels,y_pred_batch_labels),mean_squared_error(y_batch_labels,y_pred_batch_labels,squared=False),epoch_drug_cls_loss_avg.average,epoch_domain_cls_loss_avg.average,epoch_loss_avg.average))
                
                interval=1
                if (ep+1) % interval == 0:
                    val_ep.append(ep)
                    val_loss, val_drug_cls_loss,val_domain_cls_loss, \
                        val_rmse,val_r2 = self.eval_on_val_dataset(sess,val_datagen,val_iter_num,ep+1,common_ccle,pd_drug_ccle,min_max_scaler_common,min_max_scaler_labels)
                        #val_rmse,val_r2 = self.eval_on_val_dataset(sess,val_datagen,val_iter_num,ep+1,common_ccle,pd_drug_ccle,min_max_scaler_labels)
                    
                    val_loss_results.append(val_loss)
                    val_drug_cls_loss_results.append(val_drug_cls_loss)
                    val_domain_cls_loss_results.append(val_domain_cls_loss)
                    #val_accuracy_results.append(val_accuracy)
                    
                    str =  "Epoch{:03d}_val_image_cls_loss{}_val_domain_cls_loss{}_val_loss{}" \
                           "_val_rmse{:.3%}_val_r2{:.3%}".format(ep+1,val_drug_cls_loss,val_domain_cls_loss,val_loss,val_rmse,val_r2)
                    print(str)                                          
                    if val_rmse < val_acc_max:             
                        val_acc_max = val_rmse
                        
                    print(val_acc_max)
                
            print('\n----------- end to train -----------\n')        
            
              
    def eval_on_val_dataset(self,sess,val_datagen,val_batch_num,ep,common_ccle,pd_drug_ccle,min_max_scaler_common,min_max_scaler_labels):
    #def eval_on_val_dataset(self,sess,val_datagen,val_batch_num,ep,common_ccle,pd_drug_ccle,min_max_scaler_labels):
        """

        """
        epoch_loss_avg = AverageMeter()
        epoch_drug_cls_loss_avg = AverageMeter()
        epoch_domain_cls_loss_avg = AverageMeter()
        epoch_accuracy = AverageMeter()

        y_val_batch_labels = []
        y_val_pred_batch_labels = []
        y_val_batch_labels_argmax = []
        y_val_pred_batch_labels_argmax =[]
        

        for i in np.arange(1, val_batch_num + 1):
            
            batch_ccle_data = val_datagen.__next__()#val_datagen.next_batch()

            #min_max_scaler = MinMaxScaler()
            #min_max_scaler.fit_transform(common_ccle.values.T)
            # min_max_scaler_ccle_labels = MinMaxScaler()
            # min_max_scaler_ccle_labels.fit_transform(batch_ccle_data[:][:,-1].reshape([-1,1]))
            batch_ccle_ep = make_batch_ep(batch_ccle_data,common_ccle,min_max_scaler_common)

            #batch_fp_ccle=make_batch_fp(batch_ccle_data,pd_drug_ccle)
            batch_drug_name =make_batch_druglist(batch_ccle_data)
       
            adj_matrix, node_feats, b_idx = load_batch_graph_data(batch_drug = batch_drug_name,num_nodes=1536,num_graphs=16,dim_feats=44,smile_graph=self.smile_graph,drug_smile=self.drug_smiles,drug_dict=self.drug_list)
                    
            suppot = [preprocess_adj(adj_matrix,True,True)]
            features = process_features(node_feats)
                  
            GCN_DICT = build_dictionary_GCN(feats=features,support=suppot,placeholders=self.gcn_placeholders)

            batch_ccle_labels = make_batch_regression1(batch_ccle_data,min_max_scaler_labels)
            #batch_ccle_labels = make_batch_regression(batch_ccle_data)
            #print(batch_fp_ccle)
            #domain         
            batch_domain_labels = np.tile([0., 1.], [self.cfg.batch_size * 2, 1])
            val_dict ={}
            fc_dict={self.gene_source_input: batch_ccle_ep,
                                        self.gene_target_input: batch_ccle_ep,   
                                        self.response_labels:batch_ccle_labels,
                                        self.domain_labels: batch_domain_labels}
            val_dict.update(fc_dict)
            val_dict.update(GCN_DICT)
            
            val_loss, val_drug_cls_loss, val_domain_cls_loss,val_response_cls  = \
                sess.run([self.loss, self.response_cls_loss, self.domain_cls_loss, self.response_cls],
                        feed_dict=val_dict) 
            
            epoch_loss_avg.update(val_loss, 1)
            epoch_drug_cls_loss_avg.update(val_drug_cls_loss, 1)
            epoch_domain_cls_loss_avg.update(val_domain_cls_loss, 1)
            #epoch_accuracy.update(val_acc, 1)
            
            origin_y_batch_labels = min_max_scaler_labels.inverse_transform(batch_ccle_labels)
            origin_y_pred_batch_labels = min_max_scaler_labels.inverse_transform(val_response_cls)
            y_val_batch_labels+=list(origin_y_batch_labels)
            y_val_pred_batch_labels+=list(origin_y_pred_batch_labels)
            

        return epoch_loss_avg.average,epoch_drug_cls_loss_avg.average,\
                   epoch_domain_cls_loss_avg.average,mean_squared_error(y_val_batch_labels,y_val_pred_batch_labels,squared=False),r2_score(y_val_batch_labels,y_val_pred_batch_labels)
