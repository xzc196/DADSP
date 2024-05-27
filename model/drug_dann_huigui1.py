import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from doctest import OutputChecker
from re import S, X
from tkinter import SEL_LAST, W
from turtle import shape
from typing_extensions import Self
from unicodedata import name
#from matplotlib import units
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

import tf_geometric as tfg


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
#闂佽法鍠愰弸濠氬箯閻戣姤鏅搁柡鍌樺€栫€氬綊鏌ㄩ悢鍛婄伄闁归鍏橀弫鎾诲棘閵堝棗顏堕梺璺ㄥ枔缁劍绗熷☉銏℃櫢闁哄倶鍊栫€氬綊鏌ㄩ悢鍛婄伄闁归鍏橀弫鎾诲棘閵堝棗顏堕梺璺ㄥ枑閺嬪骞忛敓锟�  
def corruption(x,noise_factor = 0.2):
    noise_vector = x+noise_factor*np.random.randn(x.shape)
    #np.clip闂佽法鍠愰弸濠氬箯缁岋拷,min,max闂佽法鍠愰弸濠氬箯閻戣姤鏅搁柟鎼簼閺屽洭鎮€涙ê顏堕梺璺ㄥ枑閺嬪骞忛敓锟�
    noise_vector = np.clip(noise_vector,0.,1.)
    return noise_vector
#闂佽法鍠撳▓鏇犳媼鐟欏嫬顏堕梺璺ㄥ枑閺嬪骞忕粚鏀卆rs闂佽法鍠愰弸濠氬箯閻戣姤鏅稿Δ妞捐閻撳骞忔搴＄厺闂佽法鍠撶划鍝ユ兜闁垮顏堕梺璺ㄥ枑閺嬪骞忛悜鑺ユ櫢闁哄倶鍊栫€氬綊鏌ㄩ悤鍌涘?

def extra_same_elem(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    iset = set1.intersection(set2)
    return list(iset)
#ssp
def make_SSP(drug_fingerprint):
        ssp_mat = 1. - squareform(pdist(drug_fingerprint.values, 'jaccard'))
        return pd.DataFrame(ssp_mat, index=drug_fingerprint.index, columns=drug_fingerprint.index)
#闂佽法鍠愰弸濠氬箯瀹勬澘绲縮sp闂佽法鍠愰弸濠氬箯閻戣姤鏅搁柡鍌樺€栫€氾拷
def make_batch_ssp(batch_response_data,pd_ssp):
    batch_drug_name = batch_response_data[:][:,0]
    batch_ssp = pd_ssp.loc[batch_drug_name]
    return batch_ssp.values
#闂佽法鍠愰弸濠氬箯瀹勬澘绲块梺璺ㄥ枑閺嬪骞忛悜鑺ユ櫢闁哄倶鍊栫€氬綊鏌ㄩ悢鍛婄伄闁归鍏橀弫鎾诲棘閵堝棗顏堕梺璺ㄥ枑閺嬪骞忛悜鑺ユ櫢闁跨噦鎷�?
def make_batch_ep(batch_response_data,pd_ep,min_max_scaler):
    batch_gene_name = batch_response_data[:][:,1]
    batch_gene_name=[str(i) for i in batch_gene_name]
    batch_ep = pd_ep[batch_gene_name]
    return min_max_scaler.transform(batch_ep.values.T)
#闂佽法鍠愰弸濠氬箯瀹勬澘绲块梺璺ㄥ枑閺嬪骞忛悜鑺ユ櫢闁哄倶鍊栫€氬綊鏌ㄩ悢鐑樼暦闁规彃顑嗙€氬湱绮甸敓锟�
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
#闂佸吋鍎抽崲鑼闁秴鐐婇柣鎰皺缁夊搫霉閻樹警鍤欏┑顔惧枛瀵粙宕堕埡濠傚О
def make_batch_regression(batch_response_data):
    batch_labels =batch_response_data[:][:,3]
    return batch_labels
#閻熸粎澧楃敮濠勭博閹绢喖绀岄柡宥冨妿閸ㄥジ鏌ㄩ悤鍌涘?
def make_batch_regression1(batch_response_data,min_max_scaler):
    batch_labels =batch_response_data[:][:,3]
    return min_max_scaler.transform(batch_labels.reshape([-1,1]))

#闂佸搫鍊绘晶妤呭汲閻旂厧绠叉い鏃傛櫕閵堟挳鏌ㄩ悤鍌涘?
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
#闂佸憡甯掑Λ妤呮偤濞嗘挸鐐婇柣鎰靛墯濞堝爼鏌熺拠鈥虫灈妞わ腹鏅犻幃鍫曟晸閿燂拷
def construct_graph(drug_name,drug_dict,drug_smile,smile_graph):
    A = np.zeros(shape=(96,78))
    x_A = np.array(smile_graph[drug_smile[drug_dict[drug_name]]][1])
    #print(x_A.shape)
    A[0:x_A.shape[0]] = x_A
    return tfg.Graph(
        x=A,
        edge_index=np.array(smile_graph[drug_smile[drug_dict[drug_name]]][2]).T
).convert_edge_to_directed()
#
#graphs
@tf.function()
def create_batch_graph(batch_drug_list,drug_dict,drug_smile,smile_graph):
        batch_graphs_list = []
        for drug_name in batch_drug_list:
            #print(smile_graph[drug_smile[drug_dict[drug_name]]][2])
            graph = tfg.Graph(
            x=smile_graph[drug_smile[drug_dict[drug_name]]][1],
            edge_index=smile_graph[drug_smile[drug_dict[drug_name]]][2]).convert_edge_to_directed()
            batch_graphs_list.append(graph)
        batch_graph = tfg.BatchGraph.from_graphs(batch_graphs_list)

        return batch_graph.x,batch_graph.edge_index,batch_graph.edge_weight
    

class Drug_DANN(object):
    def __init__(self,config,SAE_Newcheck,drug_dict,drug_smile,smile_graph):
        # 闂佽法鍠愰弸濠氬箯瀹勯偊娼楅梺璺ㄥ枑閺嬪骞忛悜鑺ユ櫢闁哄倶鍊栫€氬綊鏌ㄩ悢鍛婄伄闁归鍏橀弫鎾诲棘閵堝棗顏�
        self.cfg = config
        self.Newcheck = SAE_Newcheck
        self.drug_list = drug_dict
        self.drug_smiles = drug_smile
        self.smile_graph = smile_graph

        # 闂佽法鍠愰弸濠氬箯閻戣姤鏅搁柡鍌樺€栫€氬綊鏌ㄩ悢鍛婄伄闁归鍏橀弫鎾剁矓缁嬫寧顫嶅ù鍏肩懄鐎氬綊鏌ㄩ悤鍌涘?

        self.grl_lambd = tf.placeholder(tf.float32, [])                         # GRL
        self.learning_rate = tf.placeholder(tf.float32, [])                     #
        
        self.response_labels = tf.placeholder(tf.float32, name ='ic50') #闂佹悶鍎抽崑娑氱礊婵犲洤鍐€闁搞儮鏅╅崝顕€鏌ㄩ悤鍌涘?
        self.domain_labels = tf.placeholder(tf.float32, shape=(None, 2)) #闂佺硶鏅濋崰搴ㄥ垂鎼达絿灏电紓浣诡焽閸ㄥジ鏌ㄩ悤鍌涘?

        #闁搞儻鎷�
        self.graph_x = tf.placeholder(tf.float32,shape = (1536,44),name='1')
        self.graph_edge_index = tf.placeholder(tf.float32,shape =(2,None),name='2')
        self.graph_edge_weights = tf.placeholder(tf.float32,name='3')
        self.drug_weights = tf.Variable(tf.random.truncated_normal([44, 16]))
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

        # 闂佽法鍠愰弸濠氬箯閻戣姤鏅搁柦妯侯槺缁ㄥ潡鏌ㄩ悢鍛婄伄闁归銆綾c闂佽法鍠愰弸濠氬箯缁屽〗c_roc,f1,precsicon,recall
        #self.y_true= tf.argmax(self.response_labels, 1)
        #self.y_pred= tf.argmax(self.response_cls, 1)
        #correct_label_pred = tf.equal(tf.argmax(self.response_labels, 1), tf.argmax(self.response_cls, 1))
        #self.acc = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))

        # 闂佽法鍠愰弸濠氬箯閻戣姤鏅搁柡鍌樺€栫€氱懓螣閿熺姵鏅搁梺鏉跨仛閸炲骞忛悜鑺ユ櫢闁哄倶鍊栫€氬綊鏌ㄩ悢鍛婄伄闁归鍏橀弫鎾诲棘閵堝棗顏堕梺璺ㄥ枑閺嬪骞忛悜鑺ユ櫢闁哄倶鍊栫€氬綊鏌ㄩ悤鍌涘?
        self.saver_save = tf.train.Saver(max_to_keep=100)  #

        # 闂佽法鍠愰弸濠氬箯閻戣姤鏅搁柡鍌樺€栫€氬湱鈧冻缂氱弧鍕煥閻斿憡鐏柟鍑ゆ嫹
        self.global_step = tf.Variable(tf.constant(0),trainable=False)
        #self.process = self.global_step / self.cfg.epoch

        # 闂佽法鍠愰弸濠氬箯瀹勯偊娼楅梺璺ㄥ枑閺嬪骞忛悜鑺ユ櫢闁煎瓨鐭槐顕€骞忛悜鑺ユ櫢闁哄倶鍊栫€氾拷
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
        
        # 闂佽法鍠愰弸濠氬箯閻戣姤鏅搁柡鍌樺€栫€氱ANN婵☆垽绻濋弫鎾诲棘閵堝棗顏堕梺璺ㄥ枑閺嬪骞忛悜鑺ユ櫢闁哄倶鍊栫€氾拷 K.layers.Input(shape = 闂佽法鍠愰弸濠氬箯閿燂拷16759闂佽法鍠愰弸濠氬箯閻戣姤鏅搁柡鍌樺€栫€氾拷 婵犙勫姍閺佹捇寮妶鍡楊伓闂佽法鍠愰弸濠氬箯妞嬪孩绐楅梺璺ㄥ枑閺嬪骞忛悜鑺ユ櫢闁哄倶鍊栫€氬綊骞忔總鍛婃櫢闁哄倶鍊栫€氾拷 闂佽法鍠愬鍐儌閸涘﹤顏堕梺璺ㄥ枑閺嬪骞忕粚濂眛ch*2闂佽法鍠愰弸濠氬箯閿燂拷16017闂佽法鍠愰弸濠氬箯閿燂拷
        self.gene_source_input = K.layers.Input(shape=self.cfg.gene_input_shape,batch_size =16,name="source_gene_input")
        #print(self.gene_source_input.shape)
        self.gene_target_input = K.layers.Input(shape=self.cfg.gene_input_shape,batch_size =16,name="target_gene_input")
        #print(self.gene_target_input.shape)
        self.gene_input = K.layers.Concatenate(axis=0,name="gene_input")([self.gene_source_input,self.gene_target_input])
        # 闂佽法鍠愰弸濠氬箯閻戣姤鏅搁柡鍌樺€栫€氬綊鏌ㄩ悢鍛婄伄闁瑰嘲鍢茶ぐ锟�
        print(self.gene_input.shape)
        share_feature = self.featur_extractor(self.gene_input)
        print(share_feature.shape)
        # 闂佽法鍠愰弸濠氬箯閻戣姤鏅稿Δ妤勬〃缁鳖噣骞忛悜鑺ユ櫢閻炴稒顨堝▍銏ゅ箯閻戣姤鏅搁柡鍌樺€栫€氬綊鏌ㄩ悢鍛婄伄闁归鍏橀弫鎾诲棘閵堝棗顏跺☉鎾跺劋缁噣鏌ㄩ悢鍛婄伄闁归鍏橀弫鎾诲棘閵堝棗顏堕梺璺ㄥ枑閺嬪骞忛悜鑺ユ櫢闁哄倶鍊栫€氬綊鏌ㄩ悢鍛婄伄闁归鍏橀弫鎾诲棘閵堝棗顏堕柣鈺婂櫍閺佹捇寮妶鍡楊伓闂佽法鍠愰弸濠氬箯閻戣姤鏅搁柡鍌樺€栫€氬綊鏌ㄩ悢鍛婄伄闁归鍏橀弫鎾诲棘閵堝棗顏堕梺璺ㄥ枑閺嬪骞忛敓锟�
        source_feature,target_feature = \
            K.layers.Lambda(tf.split, arguments={'axis': 0, 'num_or_size_splits': 2})(share_feature)
        source_feature = K.layers.Lambda(lambda x:x,name="source_feature")(source_feature)
        print(source_feature.shape)
        print(source_feature)
        # 闂佽法鍠愰弸濠氬箯瀹勬澘绲块梺璺ㄥ枑閺嬪骞忔搴姰闂佽法鍠愰弸濠氬箯閻戣姤鏅搁柡鍌樺€栫€氬綊鏌ㄩ悢鍛婄伄闁归鍏橀弫鎾诲棘閵堝棗顏堕梺璺ㄥ枑閺嬪骞忛悜鑺ユ櫢闁哄倶鍊栫€氬綊鏌ㄩ悢鍛婄伄闁归鍏橀弫鎾诲棘閵堝棗顏堕梺璺ㄥ枑閺嬪骞忛悜鑺ユ櫢闁跨噦鎷�?
        self.drug_feature = self.build_drug_gcn_feature()
        print(self.drug_feature.shape)
        self.response_cls = self.build_response_classify_model(source_feature)
        #print()
        
        self.domain_cls = self.build_domain_classify_model(share_feature)
        
        
#闂佽法鍠愰弸濠氬箯閻戣姤鏅搁柡鍌樺€栫€氬綊鏌ㄩ悢鍛婄伄闁瑰嘲鍢茶ぐ鍥煥閻斿憡鐏柟椋庡厴閺佹捇骞戦悿顖ｅ晣闁归鍏橀弫鎾诲棘閵堝棗顏�
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
#闂佹悶鍎抽崑娑氱礊婵犲洦鏅搁柨鐕傛嫹?
    def build_response_classify_model(self,gene_classify_feature):
        """
        闂佽法鍠愰弸濠氬箯閻戣姤鏅搁悷娆愬笚閹苯顕欏ú顏呮櫢闁哄倶鍊栫€氬湱绮垫ィ鍐╂櫢闁哄倶鍊栫€氬綊鏌ㄩ悢鍛婄伄闁归鍏橀弫鎾诲棘閵堝棗顏舵俊顖ょ節閺佹捇鏌婇悽鍨暠闁绘艾鐡ㄧ€氬綊鏌ㄩ悢鍛婄伄闁瑰嚖鎷�
        :param gene_classify_feature: 闂佽法鍠愰弸濠氬箯閻戣姤鏅搁柡鍌樺€栫€氬綊鏌ㄩ悢鍛婄伄闁归鍏橀弫鎾诲棘閵堝棗顏堕梺璺ㄥ枑閺嬪骞忛悜鑺ユ櫢闁哄倶鍊栫€氬綊鏌ㄩ悢鍛婄伄闁归鍏橀弫鎾绘晸閿燂拷?
        :return:
        """
        
            # 闂佽法鍠撻悺鏂款嚈濞差亝鏅搁柡鍌樺€栫€氬湱绮垫ィ鍐╂櫢闁哄倶鍊栫€氬綊鏌ㄩ悢鍛婄伄闁归鍏橀弫鎾诲棘閵堝棗顏�
            
        #x = K.layers.Lambda(lambda x:x,name="response_classify_feature")(gene_classify_feature)
        #print(x.shape)
        #闂佽法鍠愰弸濠氬箯閻ゎ垰绁╅梺璺ㄥ枑閺嬪骞忛悜鑺ユ櫢闁哄倶鍊栫€氬綊鏌ㄩ悢鍛婄伄闁归鍏橀弫鎾诲棘閵堝棗顏堕梺璺ㄥ枑閺嬪骞忛悜鑺ユ櫢闁哄倶鍊栫€氬綊鏌ㄩ悢鍛婄伄闁归鍏橀弫鎾诲棘閵堝棗顏堕梺璺ㄥ枑閺嬪骞忛悜鑺ユ櫢闁哄倶鍊栫€氬綊骞忔總鍛婃櫢闁哄倶鍊栫€氾拷512+256
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

#闂佺硶鏅濋崰搴ㄥ垂鎼达絿灏甸悹鍥у级閻濓拷
    def build_domain_classify_model(self,domain_classify_feature):
        """
        闂佽法鍠愰弸濠氬箯閻戣姤鏅搁悷娆愬笚閹苯顕欏ú顏呮櫢闁哄倶鍊栫€氬綊鏌ㄩ悢鍛婄伄闁归鍏橀弫鎾诲棘閵堝棗顏堕梺璺ㄥ枑閺嬪骞忛柨瀣槚闂佽法鍠愰弸濠氬箯閻戣姤鏅搁柨鐕傛嫹?
        :param domain_classify_feature: 闂佽法鍠愰弸濠氬箯閻戣姤鏅搁柡鍌樺€栫€氬綊鏌ㄩ悢鍛婄伄闁归鍏橀弫鎾诲棘閵堝棗顏堕梺璺ㄥ枑閺嬪骞忛悜鑺ユ櫢闁哄倶鍊栫€氬綊鏌ㄩ悤鍌涘?
        :return:
        """
        # 闂佽法鍠撻悺鏂款嚈濞差亝鏅搁柡鍌樺€栫€氬綊鏌ㄩ悢鍛婄伄闁归鍏橀弫鎾诲棘閵堝棗顏堕梺璺ㄥ櫐閹凤拷?
    
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
    #闂佸憡甯掑Λ妤呮偤濞嗘挸鐐婇柛鎾楀喚妫楅悗鐢割暒閻掞箒銇愭笟鈧畷锝夊冀瑜庨悵锟�
    def build_drug_gcn_feature(self):
        
        Output = tfg.nn.gcn(
                    self.graph_x,
                    self.graph_edge_index,
                    self.graph_edge_weights,
                    self.drug_weights  # GCN Weight
                    # GCN use caches to avoid re-computing of the normed edge information
                    )
        Output = tf.reshape(Output,[16,-1])
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
        
        

#閻犱緡鍙冮弫鎾诲棘閵堝棗顏�
    def train(self,train_source_datagen,train_target_datagen,val_datagen,train_iter_num,val_iter_num,pd_drug_ccle,pd_drug_gdsc,common_ccle,common_gdsc,min_max_scaler_common,min_max_scaler_labels):    
    #def train(self,train_source_datagen,train_target_datagen,val_datagen,train_iter_num,val_iter_num,pd_drug_ccle,pd_drug_gdsc,common_ccle,common_gdsc,min_max_scaler_labels):
        # 闂佽法鍠愰弸濠氬箯瀹勯偊娼楅梺璺ㄥ枑閺嬪骞忛悜鑺ユ櫢闁哄倶鍊栫€氬綊鏌ㄩ悢鍛婄伄闁瑰嘲鍢茬€氭瑩鏌ㄩ悢鎯板闁哄矈鍨遍崑銏ゅ礄閵堝棗顏堕梺璺ㄥ櫐閹凤拷?
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

        # 闂佽法鍠愰弸濠氬箯瀹勯偊娼楅梺璺ㄥ枑閺嬪骞忛悿顖ｅ敳闂佽法鍠愰弸濠氬箯閻戣姤鏅搁柡鍌樺€栫€氳寰勯柆宥嗘櫢闂佹澘鐏氶幏婵嬪箯閻戣姤鏅搁柡鍌樺€栫€氬綊鏌ㄩ悢鍛婄伄闁归鍏橀弫鎾诲棘閵堝棗顏�
        train_loss_results = []                     # 闂佽法鍠愰弸濠氬箯閻戣姤鏅搁柡鍌樺€栫€氬湱鎷嬮銏℃櫢闁哄倶鍊栫€氱oss闁稿⿵鎷�
        train_gene_cls_loss_results = []           # 闂佽法鍠愰弸濠氬箯閻戣姤鏅搁柡鍌樺€栫€氬湱鎷嬮銏℃櫢闁哄倶鍊栫€氬綊鎳￠鐐存櫢闁哄倶鍊栫€氬綊鏌ㄩ悢鍛婄伄闁归鍏橀弫鎾跺緤椤ょs闁稿⿵鎷�
        train_domain_cls_loss_results = []          # 闂佽法鍠愰弸濠氬箯閻戣姤鏅搁柡鍌樺€栫€氬湱鎷嬮銏℃櫢闁哄倶鍊栫€氬綊鏌ㄩ悢鍛婄伄闁归鍏橀弫鎾诲棘閵堝棗顏堕梺璺ㄥ枔閹峰嵀ss闁稿⿵鎷�
        train_accuracy_results = []                 # 闂佽法鍠愰弸濠氬箯閻戣姤鏅搁柡鍌樺€栫€氬湱鎷嬮銏℃櫢闁哄倶鍊栫€氱ccuracy闁稿⿵鎷�

        # 闂佽法鍠愰弸濠氬箯瀹勯偊娼楅梺璺ㄥ枑閺嬪骞忛悜鑺ユ櫢闁哄倶鍊栫€氬湱鎷犳笟鈧弫鎾诲棘閵堝棗顏跺鍫曚憾閺佹捇鏌婇崹顐ｅ珪闁归鍏橀弫鎾诲棘閵堝棗顏堕梺璺ㄥ枑閺嬪骞忛悜鑺ユ櫢濞寸姴顑戠槐婵嬫煥閻斿憡鐏柟椋庢焿閻﹀鏌ㄩ悢鍛婄伄闁圭兘顥撶痪褔顢旈幎鑺ユ櫢闁跨噦鎷�?
        val_ep = []
        val_loss_results = []                     # 闂佽法鍠愰弸濠氬箯閻戣姤鏅搁柡鍌樺€栫€氬綊鏌ㄩ悢鍛婄伄闁归鏌夐惁濉磑ss闁稿⿵鎷�
        val_drug_cls_loss_results = []           # 闂佽法鍠愰弸濠氬箯閻戣姤鏅搁柡鍌樺€栫€氬綊鏌ㄩ悢鍛婄伄闁归鏌夐惁澶愭嚒椤栫偞鏅搁柡鍌樺€栫€氬綊鏌ㄩ悢鍛婄伄闁归鍏橀弫鎾跺緤椤ょs闁稿⿵鎷�
        val_domain_cls_loss_results = []          # 闂佽法鍠愰弸濠氬箯閻戣姤鏅搁柡鍌樺€栫€氬綊鏌ㄩ悢鍛婄伄闁归鏌夐惁澶愭煥閻斿憡鐏柟椋庡厴閺佹捇寮妶鍡楊伓闂佽法鍠撻幏鍗璼s闁稿⿵鎷�
        val_accuracy_results = []                 # 闂佽法鍠愰弸濠氬箯閻戣姤鏅搁柡鍌樺€栫€氬綊鏌ㄩ悢鍛婄伄闁归鏌夐惁濉curacy闁稿⿵鎷�
        val_acc_max = 10                           # 闂佽法鍠愰弸濠氬箯閻戣姤鏅搁柡鍌樺€栫€氬綊鏌ㄩ悢娲绘晭闁靛牆绻戠€氬綊鏌ㄩ悢鍛婄伄闁归鍏橀弫鎾绘晸閿燂拷?
        test_max_r = 0


        
        #闂佽法鍠愰弸濠氬箯閻戣姤鏅搁柡鍌樺€栫€氬綊骞愰崶顒佹櫢闁哄倶鍊栫€氾拷:auc_roc,f1
        #auc_roc = metrics.roc_auc_score(test_y,prodict_prob_y)
        
        y_batch_labels = [] #婵烇絽娲︾换鍌炴偤閵娾晛鏋侀柣妤€鐗嗙粊锕傛煛瀹ュ懏宸濇い鎺炴嫹
        y_pred_batch_labels = [] #婵烇絽娲︾换鍌炴偤閵婏讣绱ｉ柛鏇ㄥ墰閵堟挳鏌″鍛窛妞ゆ帪鎷�
#         domain_batch_labels =[]
#         domain_pred_lables =[]
        op_num = '{0}'.format(0)

        with tf.Session() as sess:
            # 闂佽法鍠愰弸濠氬箯瀹勯偊娼楅梺璺ㄥ枑閺嬪骞忛悜鑺ユ櫢闁哄倶鍊栫€氬綊鏌ㄩ悢鍛婄伄闁瑰嚖鎷�
            sess.run(tf.global_variables_initializer())

            # 闂佽法鍠愰弸濠氬箯閻戣姤鏅搁柡鍌樺€栫€氳锛愰崟顕呭敳闂佽法鍠愰弸濠氬箯闁垮瑔渚€鏌ㄩ悢鍛婄伄闁瑰嚖鎷�
#             if pre_model_path is not None:              # pre_model_path闂佽法鍠嶉懠搴ｅ枈婢跺顏堕柛褉鍋撻柛鎰懇閺佹捇寮妶鍡楊伓.ckpt
#                 saver_restore = tf.train.import_meta_graph(pre_model_path+".meta")
#                 saver_restore.restore(sess,pre_model_path)
#                 print("restore model from : %s" % (pre_model_path))

            self.merged = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter(log_dir, sess.graph)
            #self.writer1 = tf.summary.FileWriter(os.path.join("./tf_dir"), sess.graph)

            print('\n----------- start to train -----------\n')
            #闁稿繈鍔戦弫鎾舵偘濡ゅ嫷鍤嬮柟椋庡厴閺佹捇寮妶鍡楊伓
            total_global_step = self.cfg.epoch * train_iter_num
            for ep in np.arange(2):
                # if(ep>1):
                #         op_num = '{0}'.format(1)           
                # 闂佺儵鍋撻崝宀勫垂閸偅鍙忛悗锝庝簻椤曆囨煕閹达絽袚闁哄棛鍠栧畷姘攽閸♀晜缍忛梺鍛婄墬濞兼籍ss
                epoch_loss_avg = AverageMeter()
                epoch_drug_cls_loss_avg = AverageMeter()
                epoch_domain_cls_loss_avg = AverageMeter()
                epoch_accuracy = AverageMeter()

               
                print('Epoch {}/{}'.format(ep+1, 2))
                #闂佽法鍠愰弸濠氬箯閻戣姤鏅搁柡鍌樺€栫€氬綊鏌ㄩ悢鍛婄伄闁归鍏橀弫鎾舵喆閹烘繄鍘絙atch+batch闂佽法鍠愰弸濠氬箯缁岋拷1,0],[0,1]]
                batch_domain_labels = np.vstack([np.tile([1., 0.], [self.cfg.batch_size, 1]),
                                           np.tile([0., 1.], [self.cfg.batch_size, 1])])
                #婵絽绻橀弫鎾诲棘閵堝棗顏秂poch
                for i in np.arange(1,train_iter_num+1):
                    #闂佽桨鑳舵晶妤€鐣垫笟鈧弻鍫ュΩ閵夈儳鈧拷
                    sess1 = tf.Session()
                    print('batch{}/{}'.format(i,train_iter_num+1))
                    batch_gdsc_data = train_source_datagen.__next__()#train_source_datagen.next_batch()
                    batch_ccle_data = train_target_datagen.__next__()#train_target_datagen.next_batch()

                    #闂佽偐鐡旈崹铏櫠閸ф绀嗛柛鈩冾殘閹藉秹鏌熺粙鎸庡窛婵顨婂顐︽偋閸繄銈�
                    #batch_fp_gdsc=make_batch_fp(batch_gdsc_data,pd_drug_gdsc)
                    batch_fp_gdsc =make_batch_druglist(batch_gdsc_data)
                    
                    batch_graph_x,batch_graph_egde_index,batch_edge_weight= create_batch_graph(batch_fp_gdsc,self.drug_list,self.drug_smiles,self.smile_graph)
                    
                    batch_graph_x,batch_graph_egde_index,batch_edge_weight = sess1.run([batch_graph_x,batch_graph_egde_index,batch_edge_weight])
                    sess1.close()
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
                    op,train_loss,train_response_cls_loss,train_domain_cls_loss,y_pred_response_batch = \
                        sess.run([self.train_op['0'],self.loss,self.response_cls_loss,self.domain_cls_loss,self.response_cls],
                                  feed_dict={self.gene_source_input:batch_gdsc_ep,
                                             self.gene_target_input:batch_ccle_ep,
                                             self.response_labels:batch_gdsc_labels,
                                             self.domain_labels:batch_domain_labels,
                                             #self.drug_feature:batch_fp_gdsc,#闂佸憡甯掑Λ妤呮偤濞嗘挸绠伴柛銉㈡櫆閻繘鏌℃担鍝勵暭鐎规搫鎷�
                                             self.graph_x:batch_graph_x,
                                             self.graph_edge_index:batch_graph_egde_index,
                                             self.graph_edge_weights:batch_edge_weight,
                                             self.learning_rate:leanring_rate,
                                             self.grl_lambd:grl_lambda})
                    
                    
                    #print(batch_gdsc_labels)
                    #print(y_pred_batch_labels)
                    #闂佽法鍠愰弸濠氬箯閻戣姤鏅搁柡鍌樺€栫€氬綊骞愰崶顒佹櫢闁哄倶鍊栫€氬綊宕欓崱娑欐櫢闁哄倶鍊栫€氬綊鏌ㄩ悢鍛婄伄闁归鍏橀弫鎾诲棘閵堝棗顏�
                    #闂佸憡鐟ョ粔瀵哥礊婵犲啰鈻旈柍褜鍓熷畷鐘诲冀閵娧冪伇闂佽法鍣﹂幏锟�?
                    origin_y_batch_labels = min_max_scaler_labels.inverse_transform(batch_gdsc_labels)
                    origin_y_pred_batch_labels = min_max_scaler_labels.inverse_transform(y_pred_response_batch)
                    y_batch_labels+= list(origin_y_batch_labels)#gdsc_true_labels
                    y_pred_batch_labels+= list(origin_y_pred_batch_labels)#gdsc_pred_labels
                    
                    
                    #self.writer.add_summary(make_summary('learning_rate', leanring_rate),global_step=global_step)
                    #self.writer1.add_summary(make_summary('learning_rate', leanring_rate), global_step=global_step)

                    # 闂佽法鍠愰弸濠氬箯閻戣姤鏅搁柡鍌樺€栫€氬湱鎷嬮銏℃櫢闁哄倶鍊栫€氬綊鏌ㄩ悢鍛婄伄闁瑰嘲鍢查妵鎴︽煥閻斿憡鐏柟椋庢焿椤斿嫰鏌ㄩ悢鍛婄伄闁归鍏橀弫鎾诲棘閵堝棗顏堕梺璺ㄥ枑閺嬪骞忛敓锟�
                    epoch_loss_avg.update(train_loss,1)
                    epoch_drug_cls_loss_avg.update(train_response_cls_loss,1)
                    epoch_domain_cls_loss_avg.update(train_domain_cls_loss,1)
                    #epoch_accuracy.update(train_acc,1)

#                     # 闂佽法鍠愰弸濠氬箯閻戣姤鏅搁梺鍓у閺嬪骞忛悜鑺ユ櫢闁哄倶鍊栫€氬綊鏌ㄩ悢鍛婄伄闁瑰嚖鎷�
#                     progbar.update(i, [('train_drug_cls_loss', train_response_cls_loss),
#                                        ('train_domain_cls_loss', train_domain_cls_loss),
#                                        ('train_loss', train_loss),
#                                        ("train_acc",train_acc)])

                # 闂佽法鍠愰弸濠氬箯閻戣姤鏅搁柡鍌樺€栫€氬綊鏌ㄩ悢鍛婄伄闁归鍏橀弫鎾诲棘閵堝棗顏堕梺璺ㄥ枔缁瓕顧傞柟宄板槻閸ㄣ劑顢曢浣割伓闂佽法鍠曢、婊呭枈婢跺顏堕梺璺ㄥ枑閺嬪骞忛悜鑺ユ櫢闁哄倶鍊栫€氬綊鏌ㄩ悢鍛婄伄闁归鏌夐銊╂煥閻斿憡鐏柟椋庢焿缁楀倿鏌ㄩ悤鍌涘?
                train_loss_results.append(epoch_loss_avg.average)
                train_gene_cls_loss_results.append(epoch_drug_cls_loss_avg.average)
                train_domain_cls_loss_results.append(epoch_domain_cls_loss_avg.average)
                train_accuracy_results.append(epoch_accuracy.average)

                self.writer.add_summary(make_summary('train/train_loss', epoch_loss_avg.average),global_step=ep+1)
                self.writer.add_summary(make_summary('train/train_drug_cls_loss', epoch_drug_cls_loss_avg.average),
                                   global_step=ep+1)
                self.writer.add_summary(make_summary('train/train_domain_cls_loss', epoch_domain_cls_loss_avg.average),
                                   global_step=ep+1)
                self.writer.add_summary(make_summary('accuracy/train_accuracy', epoch_accuracy.average),global_step=ep+1)

                #闁荤姴娲ょ€氼亪鎮抽鐐茬闁搞儯鍔庨崹锟�
                #print(y_batch_labels)
                #print(y_pred_batch_labels)
                print('Epoch {}/{} train_r2 {} train_rmse {} train_drug_cls_loss {} train_domain_cls_loss {} total_loss {}'.format(ep+1, self.cfg.epoch,r2_score(y_batch_labels,y_pred_batch_labels),mean_squared_error(y_batch_labels,y_pred_batch_labels,squared=False),epoch_drug_cls_loss_avg.average,epoch_domain_cls_loss_avg.average,epoch_loss_avg.average))
                
                #闂佽法鍠愰弸濠氬箯閻ゎ垳妲堥梺璺ㄥ枑閺嬪骞忛悜鑺ユ櫢闁哄倶鍊栫€氾拷
                
                
                
                
                interval=1
                if (ep+1) % interval == 0:
                    # 闂佽法鍠愰弸濠氬箯閻戣姤鏅搁柡鍌樺€栫€氱懓螣閿熺姵鏅搁柡鍌樺€栫€氬綊鏌ㄩ悢鍛婄伄闁归鍏橀弫鎾诲棘閵堝棗顏堕悹鍥︾窔閺佹捇寮妶鍡楊伓闂佽法鍠曠欢婵堝枈婢跺顏堕梺璺ㄥ枑閺嬪骞忛悜鑺ユ櫢闁哄倶鍊栫€氾拷
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
                    if val_rmse < val_acc_max:              # 闂佽法鍠愰弸濠氬箯閻ゎ垳妲堥梺璺ㄥ枑閺嬪骞忛悜鑺ユ櫢濡ゆ缂氶幓顏堝礆娴煎瓨鏅搁柡鍌樺€栫€氬綊宕滃澶嬫櫢闁哄倶鍊栫€氬綊鎽堕崪浣稿壒闂佽法鍠愰弸濠氬箯閻戣姤鏅稿〒姘ュ劵缂嶅洭骞忛悜鑺ユ櫢闁跨噦鎷�?
                        val_acc_max = val_rmse
                        #self.saver_save.save(sess,os.path.join(checkpoint_dir,str+".ckpt"))
                    print(val_acc_max)

                    #闂佸搫鍊绘晶妤冪矈鐎靛憡瀚氶柡鍥ュ灩閸斻儵鏌ㄩ悤鍌涘?
                    # r,nrmse,test_loss,test_drug_cls_loss,test_domain_cls_loss = self.test_dataset(sess,pd_17,pd_AZD6244,pd_Nilotinib,pd_Nutlin_3,min_max_scaler_common,min_max_scaler_labels,common_ccle,pd_drug_ccle)
                    # str1 =  "Epoch{:03d}_val_drug_cls_loss{}_val_domain_cls_loss{}" \
                    #        "_test_p{:.3%}_test_nrmse{:.3%}".format(ep+1,test_drug_cls_loss,test_domain_cls_loss,r,nrmse)
                    # print(str1)                                          
                    # if abs(r) > test_max_r:              # 闂佽法鍠愰弸濠氬箯閻ゎ垳妲堥梺璺ㄥ枑閺嬪骞忛悜鑺ユ櫢濡ゆ缂氶幓顏堝礆娴煎瓨鏅搁柡鍌樺€栫€氬綊宕滃澶嬫櫢闁哄倶鍊栫€氬綊鎽堕崪浣稿壒闂佽法鍠愰弸濠氬箯閻戣姤鏅稿〒姘ュ劵缂嶅洭骞忛悜鑺ユ櫢闁跨噦鎷�?
                    #     test_max_r = r
                    #     #self.saver_save.save(sess,os.path.join(checkpoint_dir,str+".ckpt"))
                    # print(r)

            # 闂佽法鍠愰弸濠氬箯閻戣姤鏅搁柡鍌樺€栫€氬湱鎷嬮銏℃櫢闁哄倶鍊栫€氬綊鏌ㄩ悢鍛婄伄闁归鍏橀弫鎾诲棘閵堝棗顏堕悹鍥︾窔閺佹捇寮妶鍡楊伓闂佽法鍣﹂幏锟�?
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

            # 闂佽法鍠愰弸濠氬箯閻戣姤鏅搁柡鍌樺€栫€氬綊鏌ㄩ悢鍛婄伄闁归鍏橀弫鎾剁矓閸濄儺鏆滈柟鐑芥敱鑶╅梺璺ㄥ枑閺嬪骞忛敓锟�
            #model_path = os.path.join(checkpoint_dir,"trained_model.ckpt")
            #self.saver_save.save(sess,model_path)
            #print("Train model finshed. The model is saved in : ", model_path)
            print('\n----------- end to train -----------\n')        
            
            
#闂佽法鍠愰弸濠氬箯閻ゎ垳妲堥梺璺ㄥ枑閺嬪骞忛悜鑺ユ櫢闁哄倶鍊栫€氾拷    
    def eval_on_val_dataset(self,sess,val_datagen,val_batch_num,ep,common_ccle,pd_drug_ccle,min_max_scaler_common,min_max_scaler_labels):
    #def eval_on_val_dataset(self,sess,val_datagen,val_batch_num,ep,common_ccle,pd_drug_ccle,min_max_scaler_labels):
        """
        闂佽法鍠愰弸濠氬箯閻戣姤鏅搁柡鍌樺€栫€氬綊鏌ㄩ悢鍛婄伄闁归鍏橀弫鎾诲棘閵堝棗顏舵俊顖ょ節閺佹捇寮妶鍡楊伓闂佽法鍠愰弸濠氬箯閻戣姤鏅搁柡鍌樺€栫€氬湱鎷犳笟鈧弫鎾诲棘閵堝棗顏堕梺璺ㄥ枙缁舵繄鍠婃径瀣伓闂佽法鍠愰弸濠氬箯閻戣姤鏅搁柡澶嬪濞堟垿鎮€涙ê顏堕梺璺ㄥ枑閺嬪骞忛敓锟�
        :param val_datagen: 闂佽法鍠愰弸濠氬箯閻ゎ垳妲堥梺璺ㄥ枑閺嬪骞忛悜鑺ユ櫢闁哄倶鍊栫€氬綊鏌ㄩ悢绋跨ス缁绢參鏀辩€氬綊鏌ㄩ悢鍛婄伄闁归鍏橀弫鎾诲棘閵堝棗顏堕梺璺ㄥ枑閺嬪骞忛敓锟�
        :param val_batch_num: 闂佽法鍠愰弸濠氬箯閻ゎ垳妲堥梺璺ㄥ枑閺嬪骞忛悜鑺ユ櫢闁哄倶鍊栫€氬綊鏌ㄩ悢绋跨ス缁绢參鏀辩€氬綊鏌ㄩ悢鍛婄伄闁归鍏橀弫鎾诲棘閵堝棗顏堕梺璺ㄥ枑閺嬪骞忛悜鑺ユ櫢闁哄倶鍊栫€氾拷
        """
        epoch_loss_avg = AverageMeter()
        epoch_drug_cls_loss_avg = AverageMeter()
        epoch_domain_cls_loss_avg = AverageMeter()
        epoch_accuracy = AverageMeter()

        #闂佽法鍠愰弸濠氬箯閻戣姤鏅告俊妤佹⒐閸炲骞忓畡鏉挎珯acc aucroc pre recall f1
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
            batch_graph= create_batch_graph(batch_fp_ccle,self.drug_list,self.drug_smiles,self.smile_graph)
            batch_ccle_labels = make_batch_regression1(batch_ccle_data,min_max_scaler_labels)
            #batch_ccle_labels = make_batch_regression(batch_ccle_data)
            #print(batch_fp_ccle)
            #domain         
            batch_domain_labels = np.tile([0., 1.], [self.cfg.batch_size * 2, 1])

            #batch_mnist_m_image_data = (batch_mnist_m_image_data - self.cfg.val_image_mean) /255.0
            #batch_mnist_m_domain_labels = np.ones((self.cfg.batch_size,1))
            # 闂佽法鍠愰弸濠氬箯閻戣姤鏅搁柡鍌樺€栫€氬湱鎷犳笟鈧弫鎾绘⒓閹偊鍟囬柟宄板槻瑜把囨煥閻斿憡鐏柟椋庡厴閺佹捇寮妶鍡楊伓闁烩晩鍣ｉ弫鎾诲棘閵堝棗顏堕梺璺ㄥ枑閺嬪骞忛悜鑺ユ櫢闁哄倶鍊栫€氬綊鏌ㄩ悢绋跨ス缁绢參鏀辩€氬綊鏌ㄩ悢鍛婄伄闁归鍏橀弫鎾舵喆閹哄棙娅旈柟椋庡厴閺佹捇寮妶鍡楊伓闁煎厖绶氶弫鎾诲棘閵堝棗顏堕梺璺ㄥ櫐閹凤拷?
            #batch_domain_labels = np.concatenate((batch_mnist_m_domain_labels, batch_mnist_m_domain_labels), axis=0)
            # 闂佽法鍠愰弸濠氬箯閻戣姤鏅搁柡鍌樺€栫€氱懓螣閿熺姵鏅搁柡鍌樺€栫€氬綊鏌ㄩ悢鍛婄伄闁归鍏橀弫鎾诲棘閵堝棗顏堕悹鍥︾窔閺佹捇寮妶鍡楊伓闂佽法鍠愰弸濠氬箯閻戣姤鏅搁柡鍌樺€栫€氬綊鏌ㄩ悢娲绘晭闂傚牃鏅滅€氬綊鏌ㄩ悢鍛婄伄闁瑰嘲鍢查埀顒婃嫹
            val_loss, val_drug_cls_loss, val_domain_cls_loss,val_response_cls  = \
                sess.run([self.loss, self.response_cls_loss, self.domain_cls_loss, self.response_cls],
                        feed_dict={self.gene_source_input: batch_ccle_ep,
                                        self.gene_target_input: batch_ccle_ep,
                                        #self.drug_feature:batch_fp_ccle,#fp
                                        self.graph_x:batch_graph.x,
                                        self.graph_edge_index:batch_graph.edge_index,
                                        self.graph_edge_weights:batch_graph.edge_weight,
                                        self.response_labels:batch_ccle_labels,
                                        self.domain_labels: batch_domain_labels}) 
            
            # 闂佽法鍠愰弸濠氬箯閻戣姤鏅搁柡鍌樺€栫€氬綊鏌ㄩ悢鍛婄伄闁瑰嘲鍢查妵鎴︽煥閻旀椿鍤戠紒顕€绠栭弫鎾搭殰閾忚鏆滈柟宄板槻闁解晠鏌ㄩ悢鍛婄伄闁瑰嘲鍢查埀顒婃嫹
            epoch_loss_avg.update(val_loss, 1)
            epoch_drug_cls_loss_avg.update(val_drug_cls_loss, 1)
            epoch_domain_cls_loss_avg.update(val_domain_cls_loss, 1)
            #epoch_accuracy.update(val_acc, 1)
            



            #闂佽法鍠愰弸濠氬箯閻戣姤鏅搁柡鍌樺€栫€氬綊骞愰崶顒佹櫢闁哄倶鍊栫€氾拷
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
