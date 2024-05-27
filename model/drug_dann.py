
import numpy as np
import tensorflow as tf
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
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


#���躯��
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
#���������ṩ��������  
def corruption(x,noise_factor = 0.2):
    noise_vector = x+noise_factor*np.random.randn(x.shape)
    #np.clip��x,min,max���ضϺ���
    noise_vector = np.clip(noise_vector,0.,1.)
    return noise_vector
#�Զ���kears���Ȩ�س�ʼ������

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
#��ȡ�����������
def make_batch_ep(batch_response_data,pd_ep,min_max_scaler):
    batch_gene_name = batch_response_data[:][:,1]
    batch_gene_name=[str(i) for i in batch_gene_name]
    batch_ep = pd_ep[batch_gene_name]
    return min_max_scaler.transform(batch_ep.values.T)
#��ȡ�����Ա�ǩ
def make_batch_labels(batch_response_data):
    batch_onehot_labels = []
    batch_labels = batch_response_data[:][:,2]
    for i in range(batch_labels.shape[0]):
        if batch_labels[i] =='Sensitivity':
            batch_onehot_labels.append([1,0])
        else:
            batch_onehot_labels.append([0,1])
            
    return np.array(batch_onehot_labels)


#model

class Drug_DANN(object):
    def __init__(self,config,SAE_Newcheck):
        # ��ʼ��������
        self.cfg = config
        self.Newcheck = SAE_Newcheck

        # �������ռλ��
        self.grl_lambd = tf.placeholder(tf.float32, [])                         # GRL�����
        self.learning_rate = tf.placeholder(tf.float32, [])                     # ѧϰ��
        
        self.response_labels = tf.placeholder(tf.float32, shape=(None, 2)) #�����ǩ
        self.domain_labels = tf.placeholder(tf.float32, shape=(None, 2)) #���ǩ
        self.drug_feature = tf.placeholder(tf.float32,shape = (None,222))#ҩ������
        
        #����DAnn
        self.build_DANN()
        
        self.response_cls_loss =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.response_labels,
                                                                          logits=self.response_cls))
        self.domain_cls_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.domain_labels,
                                                                        logits=self.domain_cls))
        self.loss = self.response_cls_loss+self.domain_cls_loss

        # ���徫��acc��auc_roc,f1,precsicon,recall
        self.y_true= tf.argmax(self.response_labels, 1)
        self.y_pred= tf.argmax(self.response_cls, 1)
        correct_label_pred = tf.equal(tf.argmax(self.response_labels, 1), tf.argmax(self.response_cls, 1))
        self.acc = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))

        # ����ģ�ͱ������������
        self.saver_save = tf.train.Saver(max_to_keep=100)  # ������󱣴�������Ϊ������

        # ����ѧϰ��
        self.global_step = tf.Variable(tf.constant(0),trainable=False)
        #self.process = self.global_step / self.cfg.epoch

        # ��ʼ���Ż���
        #self.optimizer = MomentumOptimizer(self.learning_rate, momentum=self.cfg.momentum_rate)
        #self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum=self.cfg.momentum_rate)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        #var_list = [v.name() for v in tf.trainable_variables()]
        self.train_op = self.optimizer.minimize(self.loss,global_step=self.global_step)
        
        
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
        ���Ǵ����������ĺ���
        :return:
        """
        # ����DANNģ������ K.layers.Input(shape = ��16017���� Դ����Ŀ����ƴ�� �ܹ���batch*2��16017��
        self.gene_source_input = K.layers.Input(shape=self.cfg.gene_input_shape,name="source_gene_input")
        #print(self.gene_source_input.shape)
        self.gene_target_input = K.layers.Input(shape=self.cfg.gene_input_shape,name="target_gene_input")
        #print(self.gene_target_input.shape)
        self.gene_input = K.layers.Concatenate(axis=0,name="gene_input")([self.gene_source_input,self.gene_target_input])
        # ������ȡ
        print(self.gene_input.shape)
        share_feature = self.featur_extractor(self.gene_input,"gene_feature")
        print(share_feature.shape)
        # ���Ȼ��ֹ�������ΪԴ������������Ŀ������������
        source_feature,target_feature = \
            K.layers.Lambda(tf.split, arguments={'axis': 0, 'num_or_size_splits': 2})(share_feature)
        source_feature = K.layers.Lambda(lambda x:x,name="source_feature")(source_feature)
        print(source_feature.shape)
        # ��ȡ��ǩ�������������������
        self.response_cls = self.build_response_classify_model(source_feature)
        
        self.domain_cls = self.build_domain_classify_model(share_feature)
        
        
#������ȡ���ݶ���
    def featur_extractor(self,gene_input,name):
        """
        ����������ȡ����Ĺ����������ѵ��Ա������ı���������
        :param gene_input: ��������
        :param name: �����������
        :return:
        """
        #self.SAE_W/b:Ԥѵ��ģ�͵�Ȩ��
        #��ȡԤѵ��ģ�͵�Ȩ��
        print(self.Newcheck.get_tensor('W_Encoder1').shape)
        print(self.Newcheck.get_tensor('b_Encoder1').shape)
        x = K.layers.Dense(1500,kernel_initializer=self.my_init1,
                                bias_initializer = self.my_init2, activation='sigmoid')(gene_input)
        
        x = K.layers.Dense(1000,kernel_initializer=self.my_init3,
                                bias_initializer = self.my_init4, activation='sigmoid')(x)
        
        x = K.layers.Dense(500,kernel_initializer=self.my_init5,
                                bias_initializer = self.my_init6, activation='sigmoid')(x)
        
        return x
#��ǩ������      
    def build_response_classify_model(self,gene_classify_feature):
        """
        ���Ǵ��ǩ������ģ�͵ĺ���
        :param gene_classify_feature: ���������������
        :return:
        """
        # ���ǩ������
        x = K.layers.Lambda(lambda x:x,name="response_classify_feature")(gene_classify_feature)
        #��ҩ��������������������ƴ��;500+222
        x = K.layers.Concatenate(axis=1,name="gene_drug_input")([gene_classify_feature,self.drug_feature])
        x = K.layers.Dense(256,kernel_initializer=K.initializers.he_normal(seed=None),
                                bias_initializer = K.initializers.Constant(value=0.1))(x)
        
        x = K.layers.Dense(128,kernel_initializer=K.initializers.he_normal(seed=None),
                                bias_initializer = K.initializers.Constant(value=0.1))(x)
        #x = K.layers.Dropout(0.5)(x)
        x = K.layers.Dense(2,kernel_initializer=K.initializers.he_normal(seed=None),
                                bias_initializer = K.initializers.Constant(value=0.1), activation='softmax',
                           name = "response_classify_pred")(x)
        return x

#���б���
    def build_domain_classify_model(self,domain_classify_feature):
        """
        ���Ǵ��������ĺ���
        :param domain_classify_feature: �������������
        :return:
        """
        # ��������
        x = GRL(domain_classify_feature,self.grl_lambd)
        
        x = K.layers.Dense(256,kernel_initializer=K.initializers.he_normal(seed=None),
                                bias_initializer = K.initializers.Constant(value=0.1))(x)
        
        x = K.layers.Dense(128,kernel_initializer=K.initializers.he_normal(seed=None),
                                bias_initializer = K.initializers.Constant(value=0.1))(x)
        #x = K.layers.Dropout(0.5)(x)
        x = K.layers.Dense(2,kernel_initializer=K.initializers.he_normal(seed=None),
                                bias_initializer = K.initializers.Constant(value=0.1), activation='softmax'
                           ,name="domain_classify_pred")(x)
        return x


#ѵ��
    def train(self,train_source_datagen,train_target_datagen,val_datagen,train_iter_num,val_iter_num,pd_ssp_ccle,pd_ssp_gdsc,common_ccle,common_gdsc,min_max_scaler_common):
        """
        ����DANN��ѵ������
        :param train_source_datagen: Դ��ѵ�����ݼ�������
        :param train_target_datagen: Ŀ����ѵ�����ݼ�������
        :param val_datagen: ��֤���ݼ�������
        :param interval: ��֤���
        :param train_iter_num: ÿ��epoch��ѵ������
        :param val_iter_num: ÿ����֤���̵���֤����
        :param pre_model_path: Ԥѵ��ģ�͵�ַ,��ѵ��ģ��Ϊckpt�ļ���ע���ļ�·��ֻ�赽.ckpt���ɡ�
        
        """
        # ��ʼ������ļ�Ŀ¼·��
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

        # ��ʼ��ѵ����ʧ�;�������
        train_loss_results = []                     # ����ѵ��lossֵ
        train_gene_cls_loss_results = []           # ����ѵ��ҩ�����lossֵ
        train_domain_cls_loss_results = []          # ����ѵ�������lossֵ
        train_accuracy_results = []                 # ����ѵ��accuracyֵ

        # ��ʼ����֤��ʧ�;������飬��֤��󾫶�
        val_ep = []
        val_loss_results = []                     # ������֤lossֵ
        val_drug_cls_loss_results = []           # ������֤ҩ�����lossֵ
        val_domain_cls_loss_results = []          # ������֤�����lossֵ
        val_accuracy_results = []                 # ������֤accuracyֵ
        val_acc_max = 0                           # �����֤����
        
        
        #����ָ��:auc_roc,f1
        #auc_roc = metrics.roc_auc_score(test_y,prodict_prob_y)
        
        y_batch_labels = []
        y_pred_batch_labels = []
#         domain_batch_labels =[]
#         domain_pred_lables =[]

        with tf.Session() as sess:
            # ��ʼ������
            sess.run(tf.global_variables_initializer())

            # ����Ԥѵ��ģ��
#             if pre_model_path is not None:              # pre_model_path�ĵ�ַд��.ckpt
#                 saver_restore = tf.train.import_meta_graph(pre_model_path+".meta")
#                 saver_restore.restore(sess,pre_model_path)
#                 print("restore model from : %s" % (pre_model_path))

            self.merged = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter(log_dir, sess.graph)
            #self.writer1 = tf.summary.FileWriter(os.path.join("./tf_dir"), sess.graph)

            print('\n----------- start to train -----------\n')
            #ȫ�ֲ���
            total_global_step = self.cfg.epoch * train_iter_num
            for ep in np.arange(self.cfg.epoch):
                # ��ʼ��ÿ�ε�����ѵ����ʧ�뾫��ƽ��ָ����  ��Ϊ0
                epoch_loss_avg = AverageMeter()
                epoch_drug_cls_loss_avg = AverageMeter()
                epoch_domain_cls_loss_avg = AverageMeter()
                epoch_accuracy = AverageMeter()

               
                print('Epoch {}/{}'.format(ep+1, self.cfg.epoch))
                #�������ǩ[batch+batch��[1,0],[0,1]]
                batch_domain_labels = np.vstack([np.tile([1., 0.], [self.cfg.batch_size, 1]),
                                           np.tile([0., 1.], [self.cfg.batch_size, 1])])
                #ÿ��epoch
                for i in np.arange(1,train_iter_num+1):
                    # ��ȡС�������ݼ�����ͼ���ǩ�����ǩ
                    
                    batch_gdsc_data = train_source_datagen.__next__()#train_source_datagen.next_batch()
                    batch_ccle_data = train_target_datagen.__next__()#train_target_datagen.next_batch()
                    batch_ssp_gdsc=make_batch_ssp(batch_gdsc_data,pd_ssp_gdsc)
                    

                    

                    batch_ccle_ep = make_batch_ep(batch_gdsc_data,common_ccle,min_max_scaler_common)
                    batch_gdsc_ep =make_batch_ep(batch_gdsc_data,common_gdsc,min_max_scaler_common)
                    
                    batch_gdsc_labels = make_batch_labels(batch_gdsc_data)
                    
                    # ����ѧϰ�ʺ�GRL��Ĳ���lambda
                    global_step = (ep-1)*train_iter_num + i
                    process = global_step * 1.0 / total_global_step
                    leanring_rate = learning_rate_schedule(process,self.cfg.init_learning_rate)
                    grl_lambda = grl_lambda_schedule(process)

                    # ǰ�򴫲�,������ʧ�����ݶ�
                    op,train_loss,train_response_cls_loss,train_domain_cls_loss,train_acc,y_pred_response_batch = \
                        sess.run([self.train_op,self.loss,self.response_cls_loss,self.domain_cls_loss,self.acc,self.response_cls],
                                  feed_dict={self.gene_source_input:batch_gdsc_ep,
                                             self.gene_target_input:batch_ccle_ep,
                                             self.response_labels:batch_gdsc_labels,
                                             self.domain_labels:batch_domain_labels,
                                             self.drug_feature:batch_ssp_gdsc,
                                             self.learning_rate:leanring_rate,
                                             self.grl_lambd:grl_lambda})
                    
                    
                    
                    #����ָ��׼������
                    y_batch_labels+= list(batch_gdsc_labels)#gdsc_true_labels
                    y_pred_batch_labels+= list(y_pred_response_batch)#gdsc_pred_labels
                    
                    
                    self.writer.add_summary(make_summary('learning_rate', leanring_rate),global_step=global_step)
                    #self.writer1.add_summary(make_summary('learning_rate', leanring_rate), global_step=global_step)

                    # ����ѵ����ʧ��ѵ������
                    epoch_loss_avg.update(train_loss,1)
                    epoch_drug_cls_loss_avg.update(train_response_cls_loss,1)
                    epoch_domain_cls_loss_avg.update(train_domain_cls_loss,1)
                    epoch_accuracy.update(train_acc,1)

#                     # ���½�����
#                     progbar.update(i, [('train_drug_cls_loss', train_response_cls_loss),
#                                        ('train_domain_cls_loss', train_domain_cls_loss),
#                                        ('train_loss', train_loss),
#                                        ("train_acc",train_acc)])

                # ���������ʧ�뾫��ֵ�������ڿ��ӻ�
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

                print('Epoch {}/{} train_acc {} train_auc {} train_drug_cls_loss {} train_domain_cls_loss {} total_loss {}'.format(ep+1, self.cfg.epoch,epoch_accuracy.average,metrics.roc_auc_score(y_batch_labels,y_pred_batch_labels),epoch_drug_cls_loss_avg.average,epoch_domain_cls_loss_avg.average,epoch_loss_avg.average))
                
                #��֤����
                
                
                
                
                interval=1
                if (ep+1) % interval == 0:
                    # ����ģ������֤���ϵ�����
                    val_ep.append(ep)
                    val_loss, val_drug_cls_loss,val_domain_cls_loss, \
                        val_accuracy,val_roc,val_pre,val_recall,val_f1_core = self.eval_on_val_dataset(sess,val_datagen,val_iter_num,ep+1,common_ccle,pd_ssp_gdsc)
                    
                    val_loss_results.append(val_loss)
                    val_drug_cls_loss_results.append(val_drug_cls_loss)
                    val_domain_cls_loss_results.append(val_domain_cls_loss)
                    val_accuracy_results.append(val_accuracy)
                    
                    str =  "Epoch{:03d}_val_image_cls_loss{:.3f}_val_domain_cls_loss{:.3f}_val_loss{:.3f}" \
                           "_val_accuracy{:.3%}_val_roc{:.3%}_val_pre{:.3%}_val_recall{:.3%}_val_f1_core{:.3%}".format(ep+1,val_drug_cls_loss,val_domain_cls_loss,val_loss,val_accuracy,val_roc,val_pre,val_recall,val_f1_core)
                    print(str)                                          
                    if val_accuracy > val_acc_max:              # ��֤���ȴﵽ��ǰ��󣬱���ģ��
                        val_acc_max = val_accuracy
                        #self.saver_save.save(sess,os.path.join(checkpoint_dir,str+".ckpt"))
                    print(val_acc_max)

                        
            # ����ѵ������֤���
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

            # �������յ�ģ��
            #model_path = os.path.join(checkpoint_dir,"trained_model.ckpt")
            #self.saver_save.save(sess,model_path)
            #print("Train model finshed. The model is saved in : ", model_path)
            print('\n----------- end to train -----------\n')        
            
            
#��֤����    
    def eval_on_val_dataset(self,sess,val_datagen,val_batch_num,ep,common_ccle,pd_ssp_gdsc):
        """
        ��������ģ������֤���ϵ����ܵĺ���
        :param val_datagen: ��֤�����ݼ�������
        :param val_batch_num: ��֤�����ݼ���������
        """
        epoch_loss_avg = AverageMeter()
        epoch_drug_cls_loss_avg = AverageMeter()
        epoch_domain_cls_loss_avg = AverageMeter()
        epoch_accuracy = AverageMeter()

        #���۱�׼acc aucroc pre recall f1
        y_val_batch_labels = []
        y_val_pred_batch_labels = []
        y_val_batch_labels_argmax = []
        y_val_pred_batch_labels_argmax =[]


        for i in np.arange(1, val_batch_num + 1):

            batch_ccle_data = val_datagen.__next__()#val_datagen.next_batch()

            min_max_scaler = MinMaxScaler()
            min_max_scaler.fit_transform(common_ccle.values.T)
            batch_ccle_ep = make_batch_ep(batch_ccle_data,common_ccle,min_max_scaler)

            batch_ssp_ccle=make_batch_ssp(batch_ccle_data,pd_ssp_gdsc)
            batch_ccle_labels = make_batch_labels(batch_ccle_data)
            
            #ֻ����Ŀ�������� 

            #���ǩ           
            batch_domain_labels = np.tile([0., 1.], [self.cfg.batch_size * 2, 1])

            #batch_mnist_m_image_data = (batch_mnist_m_image_data - self.cfg.val_image_mean) /255.0
            #batch_mnist_m_domain_labels = np.ones((self.cfg.batch_size,1))
            # ����֤�׶�ֻ����Ŀ�������ݼ����ǩ���в���
            #batch_domain_labels = np.concatenate((batch_mnist_m_domain_labels, batch_mnist_m_domain_labels), axis=0)
            # ����ģ������֤�������ָ���ֵ
            val_loss, val_drug_cls_loss, val_domain_cls_loss, val_acc ,val_response_cls ,val_true,val_pred = \
                sess.run([self.loss, self.response_cls_loss, self.domain_cls_loss, self.acc,self.response_cls,self.y_true,self.y_pred],
                        feed_dict={self.gene_source_input: batch_ccle_ep,
                                        self.gene_target_input: batch_ccle_ep,
                                        self.drug_feature:batch_ssp_ccle,
                                        self.response_labels:batch_ccle_labels,
                                        self.domain_labels: batch_domain_labels}) 
            
            # ������ʧ�뾫�ȵ�ƽ��ֵ
            epoch_loss_avg.update(val_loss, 1)
            epoch_drug_cls_loss_avg.update(val_drug_cls_loss, 1)
            epoch_domain_cls_loss_avg.update(val_domain_cls_loss, 1)
            epoch_accuracy.update(val_acc, 1)

            #����ָ��
            y_val_batch_labels+=list(batch_ccle_labels)
            y_val_pred_batch_labels+=list(val_response_cls)
            
            y_val_batch_labels_argmax+=list(val_true)
            y_val_pred_batch_labels_argmax+=list(val_pred)

        #self.writer.add_summary(make_summary('val/val_loss', epoch_loss_avg.average),global_step=ep)
        #self.writer.add_summary(make_summary('val/val_drug_cls_loss', epoch_drug_cls_loss_avg),global_step=ep)
        #self.writer.add_summary(make_summary('val/val_domain_cls_loss', epoch_domain_cls_loss_avg.average),global_step=ep)
        #self.writer.add_summary(make_summary('accuracy/val_accuracy', epoch_accuracy.average),global_step=ep)

        
        #return loss cls_loss domain_loss acc auc_roc pre recall f1
        #print(len(y_val_batch_labels),len(y_val_pred_batch_labels))
        return epoch_loss_avg.average,epoch_drug_cls_loss_avg.average,\
                   epoch_domain_cls_loss_avg.average,epoch_accuracy.average,metrics.roc_auc_score(y_val_batch_labels,y_val_pred_batch_labels),metrics.precision_score(y_val_batch_labels_argmax,y_val_pred_batch_labels_argmax),metrics.recall_score(y_val_batch_labels_argmax,y_val_pred_batch_labels_argmax),metrics.f1_score(y_val_batch_labels_argmax,y_val_pred_batch_labels_argmax)