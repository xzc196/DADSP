'''
Author: your name
Date: 2021-12-07 17:58:02
LastEditTime: 2021-12-16 22:30:52
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: \Drug_DaNN\model\drug_huigui.py
'''
import numpy as np
import tensorflow as tf
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
#import matplotlib.pyplot as plt
from tensorflow import keras as K
import datetime
from config.config import config
import os

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.variables import trainable_variables
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
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from scipy.stats import pearsonr


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
#���������ṩ��������  
def corruption(x,noise_factor = 0.2):
    noise_vector = x+noise_factor*np.random.randn(x.shape)
    #np.clip��x,min,max���ضϺ���
    noise_vector = np.clip(noise_vector,0.,1.)
    return noise_vector
#�Զ���kears���Ȩ�س�ʼ������?

def extra_same_elem(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    iset = set1.intersection(set2)
    return list(iset)
#ssp
def make_SSP(drug_fingerprint):
        ssp_mat = 1. - squareform(pdist(drug_fingerprint.values, 'jaccard'))
        return pd.DataFrame(ssp_mat, index=drug_fingerprint.index, columns=drug_fingerprint.index)
#��ȡssp����
def make_batch_ssp(batch_response_data,pd_ssp):
    batch_drug_name = batch_response_data[:][:,0]
    batch_ssp = pd_ssp.loc[batch_drug_name]
    return batch_ssp.values
#��ȡ�����������?
def make_batch_ep(batch_response_data,pd_ep,min_max_scaler):
    batch_gene_name = batch_response_data[:][:,1]
    batch_gene_name=[str(i) for i in batch_gene_name]
    batch_ep = pd_ep[batch_gene_name]
    return min_max_scaler.transform(batch_ep.values.T)

def make_batch_ep1(batch_response_data,pd_ep):
    batch_gene_name = batch_response_data[:][:,1]
    batch_gene_name=[str(i) for i in batch_gene_name]
    batch_ep = pd_ep[batch_gene_name]
    return batch_ep.values.T
#��ȡ�����Ա�ǩ
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
#获得回归任务标签
def make_batch_regression(batch_response_data):
    batch_labels =batch_response_data[:][:,3]
    return batch_labels
#def make_common_ccle_ep(common_ccle,batch_response_data,):

#归一化标�?
def make_batch_regression1(batch_response_data,min_max_scaler):
    batch_labels =batch_response_data[:][:,3]
    return min_max_scaler.transform(batch_labels.reshape([-1,1]))

#新数据测�?
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
def guassian_kernel(source_feature, target_feature,batch_size = 16, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    source_feature：输入的源域特征（shape=batch_size*feature_dim）；
    target_feature：输入的目标域特征（shape=batch_size*feature_dim）；
    kernel_mul：和带宽有关的参数；
    kernel_num：高斯核的数量；
    fix_sigma：如果不同高斯核函数的带宽相同，则该值为 fix_sigma�?

    '''
    #n_samples = source_feature.shape[0] + target_feature.shape[0]#源域与目标域的总样本量�?2*batch_size�?
    n_samples = 2*batch_size
    print(n_samples)
    total = tf.concat([source_feature, target_feature], axis=0)#输入的全部样本，[2*batch,f_dim]
    print(total.shape)
    total0 = tf.tile(tf.expand_dims(total, axis=0), [n_samples, 1, 1])#[n_samples,2*batch,f_dim]
    print(total0.shape)
    total1 = tf.tile(tf.expand_dims(total, axis=1), [1, n_samples, 1])#[2* batch,n_samaples,f_dim]
    print(total.shape)
    L2_distance = tf.reduce_sum(((total0-total1)**2), axis=-1)#计算L2距离[2*batch,2*batch]
    if fix_sigma:
        bandwidth = fix_sigma#带宽
    else:
        bandwidth = tf.reduce_sum(L2_distance) / (int(n_samples)**2-int(n_samples))
    bandwidth /= kernel_mul ** int((kernel_num // 2))
    bandwidth_list = [bandwidth * (kernel_mul**int(i)) for i in range(kernel_num)]
    kernel_val = [tf.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]#核值列表，有几个核就有几个元素
    print('jieshu')
    return tf.reduce_sum(kernel_val, axis=0)

def DAN_loss(source_feature, target_feature, kernel_mul=2.0, kernel_num=5, fix_sigma=None, linear=None,batch_size = 16):
    batch_size = batch_size
    kernels = guassian_kernel(source_feature, target_feature,batch_size = batch_size,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    print('计算�?')
    if linear:
        loss = 0
        for i in range(batch_size):
            s1, s2 = i, (i+1)%int(batch_size)
            t1, t2 = s1+int(batch_size), s2+int(batch_size)
            loss += kernels[s1, s2] + kernels[t1, t2]
            loss -= kernels[s1, t2] + kernels[s2, t1]
        DAN_loss = loss / float(int(batch_size))
    else:
        loss1 = 0
        for s1 in range(batch_size):
            for s2 in range(s1+1, batch_size):
                t1, t2 = s1+int(batch_size), s2+int(batch_size)
                loss1 += kernels[s1, s2] + kernels[t1, t2]
        loss1 = loss1 / float(int(batch_size) * (int(batch_size) - 1) / 2)
    
        loss2 = 0
        for s1 in range(batch_size):
            for s2 in range(batch_size):
                t1, t2 = s1+int(batch_size), s2+int(batch_size)
                loss2 -= kernels[s1, t2] + kernels[s2, t1]
        loss2 = loss2 / float(int(batch_size) * int(batch_size))
        DAN_loss = loss1 + loss2
    return DAN_loss

def DAN_loss1(source_feature, target_feature, kernel_mul=2.0, kernel_num=5, fix_sigma=None,batch_size =16):
    batch_size = batch_size
    kernels = guassian_kernel(source_feature, target_feature,batch_size = batch_size,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = tf.reduce_mean(XX + YY - XY -YX)
    return loss
def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0/(fan_in + fan_out))
    high = constant * np.sqrt(6.0/(fan_in + fan_out))
    # 使用均匀分布
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

class Drug_regerssion(object):
    def __init__(self,config,SAE_Newcheck,list_dim) -> None:
        super().__init__()
        tf.reset_default_graph()
        self.cfg = config
        self.Newcheck = SAE_Newcheck
        #self.grl_lambd = tf.placeholder(tf.float32, [])                         # GRL�����?
        self.learning_rate = tf.placeholder(tf.float32, []) 
        
        self.response_labels = tf.placeholder(tf.float32, name ='ic50') #回归标签�?
        self.domain_labels = tf.placeholder(tf.float32, shape=(None, 2)) #域分类标�?
        self.drug_feature = tf.placeholder(tf.float32,shape = (None,256))#ҩ������
        self.batch_size = 16
        #用来存储特征提取�?
        self._X_domain = {}
        self._W_domain = {}
        self._b_domain = {}
        #model
        self.build_DANN()
        #回归损失
        self.response_cls_loss =  tf.reduce_mean(tf.square(self.response_labels-self.response_cls))
        self.loss = self.response_cls_loss
        self.saver_save = tf.train.Saver(max_to_keep=100) 
        
        self.global_step = tf.Variable(tf.constant(0),trainable=False)
        self.optimizer =tf.train.AdamOptimizer(self.learning_rate)
        #self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum=self.cfg.momentum_rate)
        self.train_op3 = self.optimizer.minimize(self.loss,global_step=self.global_step,var_list =[self.w1[0],self.w1[1],self.w2[0],self.w2[1],self.w3[0],self.w3[1],self.w4[0],self.w4[1]])#冻结特征层，训练分类�?
        self.train_op2 = self.optimizer.minimize(self.loss,global_step=self.global_step,var_list =[self.w1[0],self.w1[1],self.w2[0],self.w2[1],self.w3[0],self.w3[1],self.w4[0],self.w4[1],self._W_domain['1'],self._W_domain['2'],self._W_domain['3'],self._b_domain['1'],self._b_domain['2'],self._b_domain['3']])#全部
        self.train_op1 = self.optimizer.minimize(self.loss,global_step=self.global_step,var_list =[self._W_domain['1'],self._W_domain['2'],self._W_domain['3'],self._b_domain['1'],self._b_domain['2'],self._b_domain['3']])#冻结分类器，训练特征�?

        #编码器部�?
        self.N = len(list_dim)-1
        self.input_dim = list_dim[0]
        
        #为每一个目标编码器定义权重和偏�?
        self._W = {}
        self._b = {}
        self._X = {}
        self._X['0'] = tf.placeholder('float',[None,list_dim[0]])
        #self._X = tf.placeholder('float',[None,list_dim[0]])
        #定义每一层的权重和占位符
        for i in range(self.N):
            layer = '{0}'.format(i+1)
            print('autoencoder layer{0}:{1}--> {2}'.format(layer,list_dim[i],list_dim[i+1]))
            
            #E_weight/bias
            #w1= tf.convert_to_tensor(self.Newcheck.get_tensor('W_Encoder'+layer),tf.float32)
            #self._W['E'+layer] = tf.Variable(w1,name='W_Encoder'+layer)
            self._W['E'+layer] = self._W_domain[layer]
            print(self._W['E'+layer].shape)
            #self._b['E'+layer] = tf.Variable(np.zeros(list_dim[i+1]).astype(np.float32),name='b_Encoder'+layer)
            self._b['E'+layer] = self._b_domain[layer]
            
            #layer_dim
            self._X[layer] = tf.placeholder('float',[None,list_dim[i+1]])
            
            #D_weight/bias :共享权重（解码器权重取编码器权重的转置）
            #w2= tf.convert_to_tensor(self.Newcheck.get_tensor('W_Dncoder'+layer),tf.float32)
            self._W['D'+layer] = tf.get_variable('w_d'+layer, shape=[list_dim[i+1],list_dim[i]], initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
            print(self._W['D'+layer].shape)
            
            self._b['D'+layer] = tf.Variable(np.zeros(list_dim[i]).astype(np.float32),name='b_Dncoder'+layer)

        #op,loss:为每一个编码器定义重构损失，以及瓶颈层
        self.train_op = {}
        self.output = {}
        self.loss_mmd ={}
        self.loss_r ={}
        for i in range(self.N):
            now_layer ='{0}'.format(i+1)
            prev_layer ='{0}'.format(i)
            print(layer)
            #opt,self.per_mmd_loss = self.pretrain(self._X1[prev_layer],self._X2[prev_layer],now_layer)
            opt,loss_m,loss_r = self.pretrain(self._X_domain[prev_layer],self._X[prev_layer],now_layer)
            self.loss_mmd[now_layer] = loss_m
            self.loss_r[now_layer] =loss_r
            self.train_op[now_layer] = opt
            self.output[now_layer] = self.one_pass(self._X[prev_layer],self._W['E'+now_layer],self._b['E'+now_layer])#瓶颈�?
            
        #整个自编码器
        self.y = self.encoder(self._X['0'])    
        self.r = self.decoder(self.y)
        error = self._X['0'] - self.r
        self._loss_1 =tf.reduce_mean(tf.pow(error,2))
        self._MMD_loss = DAN_loss1(self.y,self._X_domain['2'])#最后一层的mmd loss
        self._loss = self._loss_1+self._MMD_loss+self.loss#全局微调
        optimizer1  = tf.train.AdamOptimizer(self.learning_rate)
        self._opt = optimizer1.minimize(self._loss)
        print("abs")
        
    def build_DANN(self):
        #  K.layers.Input(shape =16759)
        self.gene_input = K.layers.Input(shape=self.cfg.gene_input_shape,name="gene_input")
       
        print(self.gene_input.shape)
        share_feature = self.featur_extractor(self.gene_input,"gene_feature")
        print(share_feature.shape)
        
        self.response_cls = self.build_response_classify_model(share_feature)

    
    #sae
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

    def pretrain(self,x1,x2,layer):
        
        y = tf.nn.sigmoid(tf.matmul(x2,self._W['E'+layer])+self._b['E'+layer])
        #print(y.shape)
        r = tf.nn.sigmoid(tf.matmul(y,self._W['D'+layer])+self._b['D'+layer])
        #print(r.shape)
        #重构误差
        error =x2-r
        loss_r = tf.reduce_mean(tf.pow(error,2))+tf.reduce_mean(tf.pow(error,2))
        #print(y1.shape)
        print('计算mmd')
        loss_mmd = DAN_loss1(x1,y,batch_size=self.batch_size)
        print('b')
        loss = loss_mmd+loss_r+self.loss#每一步优化的loss为重�?+mmd+分类�?
        #var_list:最小化参数列表
        opt = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(loss,var_list=[self._W['E'+layer],self._b['E'+layer],self._b['D'+layer],self._W['D'+layer],self._b['D'+layer],self._W_domain[layer],self._b_domain[layer]])
        return opt,loss_mmd,loss_r
    
    def one_pass(self,X,W,b):
        h = tf.nn.sigmoid(tf.matmul(X,W)+b )
        return h
    #整个堆叠自编�?
    def encoder(self,X):
        x = X
        for i in range(self.N):
            layer = '{0}'.format(i+1)
            hiddenE = tf.nn.sigmoid(tf.matmul(x,self._W['E'+layer])+self._b['E'+layer])
            x = hiddenE#上一隐层作为输入
        return x
    def decoder(self,X):
        x = X
        for i in range(self.N,0,-1):
            layer = '{0}'.format(i)
            hiddenD = tf.nn.sigmoid(tf.matmul(x,self._W['D'+layer])+self._b['D'+layer])
            x = hiddenD
        return x
        
    def featur_extractor(self,gene_input,name):
   
        layer1 = K.layers.Dense(2048,kernel_initializer=self.my_init1,
                                bias_initializer = self.my_init2, activation='sigmoid')
        x = layer1(gene_input)
        self._X_domain['0'] = x
        self._W_domain['1'] = layer1.weights[0]
        self._b_domain['1'] = layer1.weights[1]
        layer2 = K.layers.Dense(1024,kernel_initializer=self.my_init3,
                                bias_initializer = self.my_init4, activation='sigmoid')
        x = layer2(x)
        #bn
        self._X_domain['1'] = x
        self._W_domain['2'] = layer2.weights[0]
        self._b_domain['2'] = layer2.weights[1]
        layer3 = K.layers.Dense(512,kernel_initializer=self.my_init5,
                                bias_initializer = self.my_init6, activation='sigmoid')
        x =layer3(x)
        self._X_domain['2'] = x
        self._W_domain['3'] = layer3.weights[0]
        self._b_domain['3'] = layer3.weights[1]
        return x
    #regression
    def build_response_classify_model(self,gene_classify_feature):
        """
        :param gene_classify_feature: 
        :return:
        """
        # ���ǩ
        x = K.layers.Lambda(lambda x:x,name="response_classify_feature")(gene_classify_feature)
        #512+256
        x = K.layers.Concatenate(axis=1,name="gene_drug_input")([gene_classify_feature,self.drug_feature])

        layer1 = K.layers.Dense(256,kernel_initializer=K.initializers.he_normal(seed=None),
                                bias_initializer = K.initializers.Constant(value=0.01),activation='relu')
        #print(layer1.kernel,layer1.bias)
        x = layer1(x)
        self.w1 = layer1.weights
        layer2 = K.layers.Dense(128,kernel_initializer=K.initializers.he_normal(seed=None),
                                bias_initializer = K.initializers.Constant(value=0.01),activation='relu')
        x = layer2(x)
        #x = K.layers.Dropout(0.5)(x)
        self.w2 = layer2.weights
        layer3 = K.layers.Dense(64,kernel_initializer=K.initializers.he_normal(seed=None),
                                bias_initializer = K.initializers.Constant(value=0.01),activation='relu')
        x = layer3(x)
        self.w3 = layer3.weights
        #回归任务，最后一层激活？sigmod:回归�?0-1�?
        #set learning phase
        #x = K.layers.Dropout(rate = 0.5)(x)
        layer4 = K.layers.Dense(1,kernel_initializer=K.initializers.he_normal(seed=None),
                                bias_initializer = K.initializers.Constant(value=0.01))
        x = layer4(x)
        self.w4 = layer4.weights
        return x
    def train(self,train_source_datagen,train_target_datagen,val_datagen,train_iter_num,val_iter_num,pd_drug_ccle,pd_drug_gdsc,common_ccle,common_gdsc,min_max_scaler_common,min_max_scaler_labels,list_dim):
        time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")


        self.cfg.save_config(time)

        # ��ʼ��ѵ����ʧ�;�������
        train_loss_results = []                     # ����ѵ��lossֵ
        train_gene_cls_loss_results = []           # ����ѵ��ҩ�����lossֵ        
        train_accuracy_results = []                 # ����ѵ��accuracyֵ

        # ��ʼ����֤��ʧ�;������飬��֤��󾫶�?
        val_ep = []
        val_loss_results = []                     # ������֤lossֵ
        val_drug_cls_loss_results = []           # ������֤ҩ�����lossֵ
                # ������֤�����lossֵ
        val_accuracy_results = []                 # ������֤accuracyֵ
        val_acc_max = 10                           # �����֤����?
        test_max_r = 0

        # _17_AAG= 'data\_17-AAG.csv'
        # AZD6244= 'data\AZD6244.csv'
        # Nilotinib= r'data\Nilotinib.csv'
        # Nutlin_3= r'data\Nutlin-3.csv'

        # pd_17 = pd.read_csv(_17_AAG)
        # pd_AZD6244 = pd.read_csv(AZD6244)
        # pd_Nilotinib = pd.read_csv(Nilotinib)
        # pd_Nutlin_3 = pd.read_csv(Nutlin_3)
        
        y_batch_labels = [] #保存数据标签
        y_pred_batch_labels = [] #保存预测标签
        
        #now_time = datetime.datetime.now().strftime('%Y-%m-%d')
        min_loss = 1
        test_loss = []
        train_loss = []
        X = {}
        X_linshi ={}#临时存放数据
        #初始�?
        
        X['0'] =common_ccle
        N = len(list_dim)-1
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())


            print('\n----------- start to train -----------\n')

            #第一轮训练为训练分类�?
            total_global_step = self.cfg.epoch * train_iter_num
            
            for ep in np.arange(self.cfg.epoch):
                # 用初始化函数初始化loss
                epoch_loss_avg = AverageMeter()
                epoch_drug_cls_loss_avg = AverageMeter()
                epoch_accuracy = AverageMeter()

                print('Epoch {}/{}'.format(ep+1, self.cfg.epoch))
                for i in np.arange(1,train_iter_num+1):
                    #数据部分
                    
                    batch_gdsc_data = train_source_datagen.__next__()#train_source_datagen.next_batch()

                    #药物分子指纹数据
                    batch_fp_gdsc=make_batch_fp(batch_gdsc_data,pd_drug_gdsc)
                    #print(batch_fp_gdsc)
                    #基因表达数据归一�?
                    batch_gdsc_ep =make_batch_ep(batch_gdsc_data,common_gdsc,min_max_scaler_common)
                    #labels为回归任�?/?
                    batch_gdsc_labels = make_batch_regression1(batch_gdsc_data,min_max_scaler_labels)
                    #batch_gdsc_labels = make_batch_regression(batch_gdsc_data)
                    #print(batch_gdsc_data)

                    # global_step = (ep-1)*train_iter_num + i
                    # process = global_step * 1.0 / total_global_step
                    # leanring_rate = learning_rate_schedule(process,self.cfg.init_learning_rate)
                    # grl_lambda = grl_lambda_schedule(process)

                    op,train_loss,train_response_cls_loss,y_pred_response_batch = \
                        sess.run([self.train_op2,self.loss,self.response_cls_loss,self.response_cls],
                                  feed_dict={self.gene_input:batch_gdsc_ep,
                                             self.response_labels:batch_gdsc_labels,
                                             self.drug_feature:batch_fp_gdsc,#分子指纹数据
                                             self.learning_rate:self.cfg.init_learning_rate
                                             #self.grl_lambd:grl_lambda
                                             })
                    #反归一化标�?
                    
                    origin_y_batch_labels = min_max_scaler_labels.inverse_transform(batch_gdsc_labels)
                    origin_y_pred_batch_labels = min_max_scaler_labels.inverse_transform(y_pred_response_batch)

                    y_batch_labels+= list(origin_y_batch_labels)#gdsc_true_labels
                    y_pred_batch_labels+= list(origin_y_pred_batch_labels)#gdsc_pred_labels
                    #print(y_batch_labels)

                    epoch_loss_avg.update(train_loss,1)
                    epoch_drug_cls_loss_avg.update(train_response_cls_loss,1)

                
                #train_loss_results.append(epoch_loss_avg.average)
                #train_gene_cls_loss_results.append(epoch_drug_cls_loss_avg.average)
                
                #train_accuracy_results.append(epoch_accuracy.average)
                
                print('Epoch {}/{} train_r2 {} train_rmse {} train_drug_cls_loss {}  total_loss {}'.format(ep+1, self.cfg.epoch,r2_score(y_batch_labels,y_pred_batch_labels),mean_squared_error(y_batch_labels,y_pred_batch_labels,squared=False),epoch_drug_cls_loss_avg.average,epoch_loss_avg.average))
                
                
                
            #第二轮训�? 固定分类器，训练特征提取�?
            
            for i in range(self.N):
                layer = '{0}'.format(i)
                layer_1 = '{0}'.format(i+1)
                Xin = X[layer]
                
                print('pretraining Layer',i+1)
                for e in range(2):
               
                    loss_mmd =0
                    loss_r = 0
                    #训练轮次为gdsc的train_num
                    for j in range(train_iter_num):
                        batch_gdsc_data = train_source_datagen.__next__()#train_source_datagen.next_batch()
                        batch_fp_gdsc=make_batch_fp(batch_gdsc_data,pd_drug_gdsc)
                        batch_gdsc_ep =make_batch_ep(batch_gdsc_data,common_gdsc,min_max_scaler_common)
                        batch_gdsc_labels = make_batch_regression1(batch_gdsc_data,min_max_scaler_labels)
                   
                      
                        #ep_ccle 
                        if i==0 :
                            batch_ccle_ep = make_batch_ep(batch_gdsc_data,Xin,min_max_scaler_common)
                        else:
                            batch_ccle_ep = make_batch_ep1(batch_gdsc_data,Xin)
                        
                        op1,p_loss_mmd,p_loss_r = sess.run([self.train_op[layer_1],self.loss_mmd[layer_1],self.loss_r[layer_1]],feed_dict={
                        self._X[layer]:batch_ccle_ep,
                        self.gene_input:batch_gdsc_ep,
                        self.response_labels:batch_gdsc_labels,
                        self.drug_feature:batch_fp_gdsc,#分子指纹数据
                        self.learning_rate:self.cfg.init_learning_rate
                        })
                    loss_mmd+=p_loss_mmd
                    loss_r +=p_loss_r
                    print(loss_mmd,loss_r)
                X_linshi[layer_1] = sess.run(self.output[layer_1],feed_dict={self._X[layer]:Xin.values.T})
                X[layer_1] =pd.DataFrame(X_linshi[layer_1].T,columns=common_ccle.columns)
                
                print('Pretraining Finished')

            #训练3
            for ep in np.arange(self.cfg.epoch):
                # 用初始化函数初始化loss
                epoch_loss_avg = AverageMeter()
                epoch_drug_cls_loss_avg = AverageMeter()
                epoch_accuracy = AverageMeter()
                print('Epoch {}/{}'.format(ep+1, self.cfg.epoch))
                for i in np.arange(1,train_iter_num+1):
                    #数据部分
                    
                    batch_gdsc_data = train_source_datagen.__next__()#train_source_datagen.next_batch()

                    #药物分子指纹数据
                    batch_fp_gdsc=make_batch_fp(batch_gdsc_data,pd_drug_gdsc)
                    #print(batch_fp_gdsc)
                    #基因表达数据归一�?
                    batch_gdsc_ep =make_batch_ep(batch_gdsc_data,common_gdsc,min_max_scaler_common)
                    #labels为回归任�?/?
                    batch_gdsc_labels = make_batch_regression1(batch_gdsc_data,min_max_scaler_labels)
                    #batch_gdsc_labels = make_batch_regression(batch_gdsc_data)
                    #print(batch_gdsc_data)

                  

                    op,train_loss,train_response_cls_loss,y_pred_response_batch = \
                        sess.run([self.train_op3,self.loss,self.response_cls_loss,self.response_cls],
                                  feed_dict={self.gene_input:batch_gdsc_ep,
                                             self.response_labels:batch_gdsc_labels,
                                             self.drug_feature:batch_fp_gdsc,#分子指纹数据
                                             self.learning_rate:self.cfg.init_learning_rate
                                             
                                             })
                    #反归一化标�?
                    origin_y_batch_labels = min_max_scaler_labels.inverse_transform(batch_gdsc_labels)
                    origin_y_pred_batch_labels = min_max_scaler_labels.inverse_transform(y_pred_response_batch)
                    y_batch_labels+= list(origin_y_batch_labels)#gdsc_true_labels
                    y_pred_batch_labels+= list(origin_y_pred_batch_labels)#gdsc_pred_labels
                    
                    epoch_loss_avg.update(train_loss,1)
                    epoch_drug_cls_loss_avg.update(train_response_cls_loss,1)

                
                #train_loss_results.append(epoch_loss_avg.average)
                #train_gene_cls_loss_results.append(epoch_drug_cls_loss_avg.average)
                
                #train_accuracy_results.append(epoch_accuracy.average)
                
                print('Epoch {}/{} train_r2 {} train_rmse {} train_drug_cls_loss {}  total_loss {}'.format(ep+1, self.cfg.epoch,r2_score(y_batch_labels,y_pred_batch_labels),mean_squared_error(y_batch_labels,y_pred_batch_labels,squared=False),epoch_drug_cls_loss_avg.average,epoch_loss_avg.average))
                
                #测试阶段
                interval=1
                if (ep+1) % interval == 0:
                    # ����ģ������֤���ϵ�����
                    val_ep.append(ep)
                    val_loss, val_drug_cls_loss, \
                        val_rmse,val_r2 = self.eval_on_val_dataset(sess,val_datagen,val_iter_num,ep+1,common_ccle,pd_drug_ccle,min_max_scaler_common,min_max_scaler_labels)
                        #val_rmse,val_r2 = self.eval_on_val_dataset(sess,val_datagen,val_iter_num,ep+1,common_ccle,pd_drug_ccle,min_max_scaler_labels)
                    
                    val_loss_results.append(val_loss)
                    val_drug_cls_loss_results.append(val_drug_cls_loss)
                    # val_domain_cls_loss_results.append(val_domain_cls_loss)
                    #val_accuracy_results.append(val_accuracy)
                    
                    str =  "Epoch{:03d}_val_image_cls_loss{}_val_loss{}" \
                           "_val_rmse{:.3%}_val_r2{:.3%}".format(ep+1,val_drug_cls_loss,val_loss,val_rmse,val_r2)
                    print(str)                                          
                    if val_rmse < val_acc_max:              # ��֤���ȴﵽ��ǰ��󣬱���ģ��?
                        val_acc_max = val_rmse
                        #self.saver_save.save(sess,os.path.join(checkpoint_dir,str+".ckpt"))
                    print(val_acc_max)

 
            print('\n----------- end to train -----------\n')        
            
            
#��֤����    
    def eval_on_val_dataset(self,sess,val_datagen,val_batch_num,ep,common_ccle,pd_drug_ccle,min_max_scaler_common,min_max_scaler_labels):
    #def eval_on_val_dataset(self,sess,val_datagen,val_batch_num,ep,common_ccle,pd_drug_ccle,min_max_scaler_labels):
 
        epoch_loss_avg = AverageMeter()
        epoch_drug_cls_loss_avg = AverageMeter()
        #epoch_domain_cls_loss_avg = AverageMeter()
        epoch_accuracy = AverageMeter()

        #���۱�׼acc aucroc pre recall f1
        y_val_batch_labels = []
        y_val_pred_batch_labels = []
        y_val_batch_labels_argmax = []
        y_val_pred_batch_labels_argmax =[]


        for i in np.arange(1, val_batch_num + 1):

            batch_ccle_data = val_datagen.__next__()#val_datagen.next_batch()
            batch_ccle_ep = make_batch_ep(batch_ccle_data,common_ccle,min_max_scaler_common)
            batch_fp_ccle=make_batch_fp(batch_ccle_data,pd_drug_ccle)
            batch_ccle_labels = make_batch_regression1(batch_ccle_data,min_max_scaler_labels)         
            #batch_domain_labels = np.tile([0., 1.], [self.cfg.batch_size * 2, 1])


            val_loss, val_drug_cls_loss,val_response_cls  = \
                sess.run([self.loss, self.response_cls_loss, self.response_cls],
                        feed_dict={self.gene_input: batch_ccle_ep,
                                   self.drug_feature:batch_fp_ccle,#fp
                                   self.response_labels:batch_ccle_labels
                                   }) 
            
            
            epoch_loss_avg.update(val_loss, 1)
            epoch_drug_cls_loss_avg.update(val_drug_cls_loss, 1)
            
            origin_y_batch_labels = min_max_scaler_labels.inverse_transform(batch_ccle_labels)
            origin_y_pred_batch_labels = min_max_scaler_labels.inverse_transform(val_response_cls)
            y_val_batch_labels+=list(origin_y_batch_labels)
            y_val_pred_batch_labels+=list(origin_y_pred_batch_labels)
            
        #return loss cls_loss domain_loss acc auc_roc pre recall f1
        #print(len(y_val_batch_labels),len(y_val_pred_batch_labels))
        return epoch_loss_avg.average,epoch_drug_cls_loss_avg.average,\
                   mean_squared_error(y_val_batch_labels,y_val_pred_batch_labels,squared=False),r2_score(y_val_batch_labels,y_val_pred_batch_labels)












