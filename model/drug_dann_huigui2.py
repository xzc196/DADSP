import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from doctest import OutputChecker
from pyexpat import features
from re import S, X
from tkinter import SEL_LAST, W
from turtle import shape
from typing_extensions import Self
from unicodedata import name
from warnings import filters
#from matplotlib import units
#from matplotlib.cbook import flatten
#from matplotlib.pyplot import axis
import numpy as np
from sklearn.feature_selection import SelectFdr
import tensorflow as tf
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
#from tensorflow.python.ops.math_ops import sigmoid
#from tensorflow.python.ops.variables import trainable_variables
#from utils.utils import plot_loss
#from utils.utils import plot_accuracy
from utils.utils import AverageMeter
#from utils.utils import make_summary
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
#import tf_geometric as tfg
from functools import reduce

#shuffle_data
def shuffle_aligned_list(data):

    num = data.shape[0]
    p = np.random.permutation(num)
    data_shufflie = data[p]
    return data_shufflie

def batch_generator(data, batch_size, shuffle=True):
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
#
def corruption(x,noise_factor = 0.2):
    noise_vector = x+noise_factor*np.random.randn(x.shape)
    noise_vector = np.clip(noise_vector,0.,1.)
    return noise_vector
#
def extra_same_elem(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    iset = set1.intersection(set2)
    return list(iset)
#ssp
def make_SSP(drug_fingerprint):
        ssp_mat = 1. - squareform(pdist(drug_fingerprint.values, 'jaccard'))
        return pd.DataFrame(ssp_mat, index=drug_fingerprint.index, columns=drug_fingerprint.index)
#
def make_batch_ssp(batch_response_data,pd_ssp):
    batch_drug_name = batch_response_data[:][:,0]
    batch_ssp = pd_ssp.loc[batch_drug_name]
    return batch_ssp.values
#
def make_batch_ep(batch_response_data,pd_ep,min_max_scaler):
    batch_gene_name = batch_response_data[:][:,1]
    batch_gene_name=[str(i) for i in batch_gene_name]
    batch_ep = pd_ep[batch_gene_name]
    return min_max_scaler.transform(batch_ep.values.T)
#
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
#
def make_batch_regression(batch_response_data):
    batch_labels =batch_response_data[:][:,3]
    return batch_labels
#
def make_batch_regression1(batch_response_data,min_max_scaler):
    batch_labels =batch_response_data[:][:,3]
    return min_max_scaler.transform(batch_labels.reshape([-1,1]))

#
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
#
def construct_graph(drug_name,drug_dict,drug_smile,smile_graph):
    A = np.zeros(shape=(96,78))
    x_A = np.array(smile_graph[drug_smile[drug_dict[drug_name]]][1])
    print(x_A.shape)
    A[0:x_A.shape[0]] = x_A
    return tfg.Graph(
        x=A,
        edge_index=np.array(smile_graph[drug_smile[drug_dict[drug_name]]][2]).T
).convert_edge_to_directed()
#graphs
def create_batch_graph(batch_drug_list,drug_dict,drug_smile,smile_graph):
        batch_graphs_list = [construct_graph(drug_name,drug_dict,drug_smile,smile_graph) for drug_name in batch_drug_list]

        batch_graph = tfg.BatchGraph.from_graphs(batch_graphs_list)

        return batch_graph
def make_batch_drug_features(batch_drug_list,drug_dict):
    features = []
    for drug in batch_drug_list:
            feature  = drug_dict[drug]
            feature = feature[:,:,np.newaxis]
            features.append(feature)
    return features

class Drug_DANN(object):
    def __init__(self,config,SAE_Newcheck,drug_dict_gdsc):
        # 
        self.cfg = config
        self.Newcheck = SAE_Newcheck
        self.drug_dict_gdsc = drug_dict_gdsc
        # 
        self.grl_lambd = tf.placeholder(tf.float32, [])                         
        self.learning_rate = tf.placeholder(tf.float32, [])                     
        self.response_labels = tf.placeholder(tf.float32, name ='ic50') 
        self.domain_labels = tf.placeholder(tf.float32, shape=(None, 2)) 
        #
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

        self.loss = self.response_cls_loss+self.domain_cls_loss

        self.saver_save = tf.train.Saver(max_to_keep=100)  #


        self.global_step = tf.Variable(tf.constant(0),trainable=False)
        #self.process = self.global_step / self.cfg.epoch

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
        
       
        self.gene_source_input = K.layers.Input(shape=self.cfg.gene_input_shape,batch_size = 64,name="source_gene_input")
        #print(self.gene_source_input.shape)
        self.gene_target_input = K.layers.Input(shape=self.cfg.gene_input_shape,batch_size = 64,name="target_gene_input")
        #print(self.gene_target_input.shape)
        self.gene_input = K.layers.Concatenate(axis=0,name="gene_input")([self.gene_source_input,self.gene_target_input])
       
        print(self.gene_input.shape)
        share_feature = self.featur_extractor(self.gene_input,"gene_feature")
        print(share_feature.shape)
        
        source_feature,target_feature = \
            K.layers.Lambda(tf.split, arguments={'axis': 0, 'num_or_size_splits': 2})(share_feature)
        source_feature = K.layers.Lambda(lambda x:x,name="source_feature")(source_feature)
        print(source_feature.shape)
        print(source_feature)

        self.drug_feature = self.build_cnn_feature()
        print(self.drug_feature.shape)
        
        self.response_cls = self.build_response_classify_model(source_feature)
        #print()
        
        self.domain_cls = self.build_domain_classify_model(share_feature)
        
        

    def featur_extractor(self,gene_input,name):
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

    def build_response_classify_model(self,gene_classify_feature):
        """
        :return:
        """ 
        #x = K.layers.Lambda(lambda x:x,name="response_classify_feature")(gene_classify_feature)
        #print(x.shape)

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

#
    def build_domain_classify_model(self,domain_classify_feature):


    
        x = GRL(domain_classify_feature,self.grl_lambd)
        
        x = K.layers.Dense(256,kernel_initializer=K.initializers.he_normal(seed=None),
                                bias_initializer = K.initializers.Constant(value=0.01),activation='relu')(x)
        
        x = K.layers.Dense(128,kernel_initializer=K.initializers.he_normal(seed=None),
                                bias_initializer = K.initializers.Constant(value=0.01),activation='relu')(x)
        #print(type(x))
        #x = K.layers.Dropout(0.5)(x)
        x = K.layers.BatchNormalization()(x,training=False)
        x = K.layers.Dense(2,kernel_initializer=K.initializers.he_normal(seed=None),
                                bias_initializer = K.initializers.Constant(value=0.01)
                        ,name="domain_classify_pred")(x)
        return x
    #
    def build_cnn_feature(self):
        self.drug_input = K.layers.Input(shape=[31,188,1],batch_size=64,name="drug_input")
        x = K.layers.Conv2D(filters = 256,kernel_size=5, activation='relu')(self.drug_input)
        x = K.layers.MaxPooling2D(pool_size=2)(x)
        x = K.layers.Conv2D(filters = 128,kernel_size=3, activation='relu')(x)
        x = K.layers.MaxPooling2D(pool_size=2)(x)
        x = K.layers.Flatten()(x)
        output = K.layers.Dense(256,kernel_initializer=K.initializers.he_normal(seed=None),
                                bias_initializer = K.initializers.Constant(value=0.01),activation='relu')(x)
        return output 
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

        #
        train_loss_results = []                     #
        train_gene_cls_loss_results = []           #
        train_domain_cls_loss_results = []          #
        train_accuracy_results = []                 #

        #
        val_ep = []
        val_loss_results = []                     #
        val_drug_cls_loss_results = []           #
        val_domain_cls_loss_results = []          #
        val_accuracy_results = []                 #
        val_acc_max = 10                           #
        #auc_roc = metrics.roc_auc_score(test_y,prodict_prob_y)
        
        y_batch_labels = [] 
        y_pred_batch_labels = [] 
#         domain_batch_labels =[]
#         domain_pred_lables =[]
        op_num = '{0}'.format(0)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())


#             if pre_model_path is not None:              # pre_model_path
#                 saver_restore = tf.train.import_meta_graph(pre_model_path+".meta")
#                 saver_restore.restore(sess,pre_model_path)
#                 print("restore model from : %s" % (pre_model_path))

            self.merged = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter(log_dir, sess.graph)
            #self.writer1 = tf.summary.FileWriter(os.path.join("./tf_dir"), sess.graph)

            print('\n----------- start to train -----------\n')

            total_global_step = self.cfg.epoch * train_iter_num
            for ep in np.arange(self.cfg.epoch):
                # if(ep>1):
                #         op_num = '{0}'.format(1)           
 
                epoch_loss_avg = AverageMeter()
                epoch_drug_cls_loss_avg = AverageMeter()
                epoch_domain_cls_loss_avg = AverageMeter()
                epoch_accuracy = AverageMeter()

               
                print('Epoch {}/{}'.format(ep+1, self.cfg.epoch))
                batch_domain_labels = np.vstack([np.tile([1., 0.], [self.cfg.batch_size, 1]),
                                           np.tile([0., 1.], [self.cfg.batch_size, 1])])

                for i in np.arange(1,train_iter_num+1):

                    
                    batch_gdsc_data = train_source_datagen.__next__()#train_source_datagen.next_batch()
                    batch_ccle_data = train_target_datagen.__next__()#train_target_datagen.next_batch()


                    #batch_fp_gdsc=make_batch_fp(batch_gdsc_data,pd_drug_gdsc)
                    batch_fp_gdsc =make_batch_druglist(batch_gdsc_data)
                    
                    batch_drug_features = make_batch_drug_features(batch_fp_gdsc,self.drug_dict_gdsc)

                    #print(batch_fp_gdsc)
                    
                    
                    batch_ccle_ep = make_batch_ep(batch_gdsc_data,common_ccle,min_max_scaler_common)
                    batch_gdsc_ep =make_batch_ep(batch_gdsc_data,common_gdsc,min_max_scaler_common)


                    batch_gdsc_labels = make_batch_regression1(batch_gdsc_data,min_max_scaler_labels)
                    #batch_gdsc_labels = make_batch_regression(batch_gdsc_data)
                    #print(batch_gdsc_data)

                    global_step = (ep-1)*train_iter_num + i
                    process = global_step * 1.0 / total_global_step
                    leanring_rate = learning_rate_schedule(process,self.cfg.init_learning_rate)
                    grl_lambda = grl_lambda_schedule(process)

                    
                    op,train_loss,train_response_cls_loss,train_domain_cls_loss,y_pred_response_batch = \
                        sess.run([self.train_op['0'],self.loss,self.response_cls_loss,self.domain_cls_loss,self.response_cls],
                                  feed_dict={self.gene_source_input:batch_gdsc_ep,
                                             self.gene_target_input:batch_ccle_ep,
                                             self.response_labels:batch_gdsc_labels,
                                             self.domain_labels:batch_domain_labels,
                                             self.drug_input:batch_drug_features,
                                             self.learning_rate:leanring_rate,
                                             self.grl_lambd:grl_lambda})
                    
                    
                    #print(batch_gdsc_labels)
                    #print(y_pred_batch_labels)
                    origin_y_batch_labels = min_max_scaler_labels.inverse_transform(batch_gdsc_labels)
                    origin_y_pred_batch_labels = min_max_scaler_labels.inverse_transform(y_pred_response_batch)
                    y_batch_labels+= list(origin_y_batch_labels)#gdsc_true_labels
                    y_pred_batch_labels+= list(origin_y_pred_batch_labels)#gdsc_pred_labels
                    
                    
                    #self.writer.add_summary(make_summary('learning_rate', leanring_rate),global_step=global_step)
                    #self.writer1.add_summary(make_summary('learning_rate', leanring_rate), global_step=global_step)

                    epoch_loss_avg.update(train_loss,1)
                    epoch_drug_cls_loss_avg.update(train_response_cls_loss,1)
                    epoch_domain_cls_loss_avg.update(train_domain_cls_loss,1)
                    #epoch_accuracy.update(train_acc,1)

#                     progbar.update(i, [('train_drug_cls_loss', train_response_cls_loss),
#                                        ('train_domain_cls_loss', train_domain_cls_loss),
#                                        ('train_loss', train_loss),
#                                        ("train_acc",train_acc)])

                train_loss_results.append(epoch_loss_avg.average)
                train_gene_cls_loss_results.append(epoch_drug_cls_loss_avg.average)
                train_domain_cls_loss_results.append(epoch_domain_cls_loss_avg.average)
                train_accuracy_results.append(epoch_accuracy.average)


                #print(y_batch_labels)
                #print(y_pred_batch_labels)
                print('Epoch {}/{} train_r2 {} train_rmse {} train_drug_cls_loss {} train_domain_cls_loss {} total_loss {}'.format(ep+1, self.cfg.epoch,r2_score(y_batch_labels,y_pred_batch_labels),mean_squared_error(y_batch_labels,y_pred_batch_labels,squared=False),epoch_drug_cls_loss_avg.average,epoch_domain_cls_loss_avg.average,epoch_loss_avg.average))
                

                
                
                
                
                interval=1
                if (ep+1) % interval == 0:
                    #
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
                        #self.saver_save.save(sess,os.path.join(checkpoint_dir,str+".ckpt"))
                    print(val_acc_max)

                    
                    # r,nrmse,test_loss,test_drug_cls_loss,test_domain_cls_loss = self.test_dataset(sess,pd_17,pd_AZD6244,pd_Nilotinib,pd_Nutlin_3,min_max_scaler_common,min_max_scaler_labels,common_ccle,pd_drug_ccle)
                    # str1 =  "Epoch{:03d}_val_drug_cls_loss{}_val_domain_cls_loss{}" \
                    #        "_test_p{:.3%}_test_nrmse{:.3%}".format(ep+1,test_drug_cls_loss,test_domain_cls_loss,r,nrmse)
                    # print(str1)                                          
                    # if abs(r) > test_max_r:              
                    #     test_max_r = r
                    #     #self.saver_save.save(sess,os.path.join(checkpoint_dir,str+".ckpt"))
                    # print(r)

            
            #path = os.path.join(result_dir, "train_loss.jpg")
            #plot_loss(np.arange(1,len(train_loss_results)+1), [np.array(train_loss_results),
                                #np.array(train_gene_cls_loss_results),np.array(train_domain_cls_loss_results)],
                                #path, "train")
            #path = os.path.join(result_dir, "val_loss.jpg")
            #plot_loss(np.array(val_ep)+1, [np.array(val_loss_results),
                                #np.array(val_drug_cls_loss_results),np.array(val_domain_cls_loss_results)],
                               #path, "val")
            #train_acc = np.array(train_accuracy_results)[np.array(val_ep)]
            #path = os.path.join(result_dir, "accuracy.jpg")
            #plot_accuracy(np.array(val_ep)+1, [train_acc, val_accuracy_results], path)

            #model_path = os.path.join(checkpoint_dir,"trained_model.ckpt")
            #self.saver_save.save(sess,model_path)
            #print("Train model finshed. The model is saved in : ", model_path)
            print('\n----------- end to train -----------\n')        
            
            
 
    def eval_on_val_dataset(self,sess,val_datagen,val_batch_num,ep,common_ccle,pd_drug_ccle,min_max_scaler_common,min_max_scaler_labels):
    #def eval_on_val_dataset(self,sess,val_datagen,val_batch_num,ep,common_ccle,pd_drug_ccle,min_max_scaler_labels):
        
        epoch_loss_avg = AverageMeter()
        epoch_drug_cls_loss_avg = AverageMeter()
        epoch_domain_cls_loss_avg = AverageMeter()
        epoch_accuracy = AverageMeter()

        #acc aucroc pre recall f1
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
            batch_fp_ccle =make_batch_druglist(batch_ccle_data)
            batch_drug_features = make_batch_drug_features(batch_fp_ccle,self.drug_dict_gdsc)
            batch_ccle_labels = make_batch_regression1(batch_ccle_data,min_max_scaler_labels)
            #batch_ccle_labels = make_batch_regression(batch_ccle_data)
            #print(batch_fp_ccle)
            #domain         
            batch_domain_labels = np.tile([0., 1.], [self.cfg.batch_size * 2, 1])

            #batch_mnist_m_image_data = (batch_mnist_m_image_data - self.cfg.val_image_mean) /255.0
            #batch_mnist_m_domain_labels = np.ones((self.cfg.batch_size,1))
            #batch_domain_labels = np.concatenate((batch_mnist_m_domain_labels, batch_mnist_m_domain_labels), axis=0)
            val_loss, val_drug_cls_loss, val_domain_cls_loss,val_response_cls  = \
                sess.run([self.loss, self.response_cls_loss, self.domain_cls_loss, self.response_cls],
                        feed_dict={self.gene_source_input: batch_ccle_ep,
                                        self.gene_target_input: batch_ccle_ep,
                                        #self.drug_feature:batch_fp_ccle,#fp
                                        self.drug_input: batch_drug_features,
                                        self.response_labels:batch_ccle_labels,
                                        self.domain_labels: batch_domain_labels}) 
            
            #
            epoch_loss_avg.update(val_loss, 1)
            epoch_drug_cls_loss_avg.update(val_drug_cls_loss, 1)
            epoch_domain_cls_loss_avg.update(val_domain_cls_loss, 1)
            #epoch_accuracy.update(val_acc, 1)
            
                                                                                                                                                                                                                                                                                        


            #
            origin_y_batch_labels = min_max_scaler_labels.inverse_transform(batch_ccle_labels)
            origin_y_pred_batch_labels = min_max_scaler_labels.inverse_transform(val_response_cls)
            y_val_batch_labels+=list(origin_y_batch_labels)
            y_val_pred_batch_labels+=list(origin_y_pred_batch_labels)
            
            #y_val_batch_labels_argmax+=list(val_true)
            #y_val_pred_batch_labels_argmax+=list(val_pred)

        #self.writer.add_summary(make_summary('val/val_loss', epoch_loss_avg.average),global_step=ep)
        #self.writer.add_summary(make_summary('val/val_drug_cls_loss', epoch_drug_cls_loss_avg),global_step=ep)
        #self.writer.add_summary(make_summary('val/val_domain_cls_loss', epoch_domain_cls_loss_avg.average),global_step=ep)
        #self.writer.add_summary(make_summary('accuracy/val_accuracy', epoch_accuracy.average),global_step=ep)

        
        #return loss cls_loss domain_loss acc auc_roc pre recall f1
        #print(len(y_val_batch_labels),len(y_val_pred_batch_labels))
        return epoch_loss_avg.average,epoch_drug_cls_loss_avg.average,\
                   epoch_domain_cls_loss_avg.average,mean_squared_error(y_val_batch_labels,y_val_pred_batch_labels,squared=False),r2_score(y_val_batch_labels,y_val_pred_batch_labels)
