# -*- coding:utf-8 -*-
#
#        Author : TangHanYi
#        E-mail : thydeyx@163.com
#   Create Date : 2018-08-02 12时20分23秒
# Last modified : 2018-08-02 16时22分44秒
#     File Name : keras_seq2seq.py
#          Desc :


import keras
import os
import json
from keras import backend as K
from keras.datasets import imdb
from keras import regularizers
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Activation
from keras.layers import Embedding, Flatten
from keras.layers import LSTM, TimeDistributed, Bidirectional
from keras.models import Model
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import multi_gpu_model
import tensorflow as tf
from keras.callbacks import EarlyStopping, TensorBoard, Callback
from keras.backend.tensorflow_backend import set_session
import numpy as np

class TrainVaildTensorBoard(TensorBoard):

    def __init__(self, log_dir='./logs', x_train=[], y_train=[], **kwargs):
        self.traing_log_dir = os.path.join(log_dir, 'training')
        super(TrainVaildTensorBoard, self).__init__(**kwargs)
        self.val_log_dir = os.path.join(log_dir, 'val')
        self.batch_count = -1
        self.t_loss = 0.0
        self.t_acc = 0.0

    def set_model(self, model):
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        self.train_writer = tf.summary.FileWriter(self.traing_log_dir)
        super(TrainVaildTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):

        logs = logs or {}
        print('\n', logs)
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        self.batch_count += 1
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            if 'loss' in name:
                self.val_writer.add_summary(summary, self.batch_count)
            else:
                self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        for name, value in logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            if 'loss' in name:
                self.train_writer.add_summary(summary, self.batch_count)
            else:
                self.train_writer.add_summary(summary, epoch)
        self.train_writer.flush()

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        val_data = self.validation_data
        try:
            batch = logs['batch']
            loss = logs['loss']
            acc = logs['acc']
            if batch % 10 == 0 and batch != 0:
                t_loss = self.t_loss / 10.0
                t_acc = self.t_acc / 10.0
                self.t_loss = 0.0
                self.t_acc = 0.0
                self.batch_count +=1

                """
                y_pred = tf.convert_to_tensor(self.model.predict(val_data[0]), np.float32)
                y_true = tf.convert_to_tensor(val_data[1], np.float32)
                val_loss = K.categorical_crossentropy(y_true, y_pred)
                #print(np.asarray(val_loss, np.float32))
                loss_list = self.sess.run(val_loss)
                val_loss = np.sum(loss_list) / len(loss_list)
                print(' - val_loss', val_loss)
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = val_loss
                summary_value.tag = 'loss'
                self.val_writer.add_summary(summary, self.batch_count)
                self.val_writer.flush()
                #batch_logs = {'loss':t_loss, 'acc':t_acc}
                """
                batch_logs = {'loss':t_loss}

                for name, value in batch_logs.items():
                    summary = tf.Summary()
                    summary_value = summary.value.add()
                    summary_value.simple_value = value
                    summary_value.tag = name
                    self.train_writer.add_summary(summary, self.batch_count)
                self.train_writer.flush()                
            else:
                self.t_loss += loss
                self.t_acc += acc

        except Exception as e:
            print(e)
                
    def on_train_end(self, logs=None):
        #super(TrainVaildTensorBoard, self).on_train_end(logs)
        self.val_writer.close()
        self.train_writer.close()


class Solution:

    def __init__(self): 
        config = tf.ConfigProto()
        #config.gpu_options.per_process_gpu_memory_fraction = 0.5
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        self.log_filepath = '../data/keras_log'
        es = EarlyStopping(monitor='val_acc', patience=5)
        #tb_cb = keras.callbacks.TensorBoard(log_dir=self.log_filepath, histogram_freq=0) 
        tb_cb = TrainVaildTensorBoard(log_dir=self.log_filepath, histogram_freq=0) 
        #tb_sb = TrainValidTensorBoardCallback()
        #self.clkb = [es, tb_cb]
        self.clkb = [tb_cb]

    def read_data(self):
    
        self.vocab_size = len(self.word_dict)
        self.sent_len_num = 200
        """
        with open('../data/sentence_train.data', 'r') as inf:
            flag = False
            for line in inf:
                line = line.strip()
                if flag == False:
                    self.xtrain = json.loads(line)
                else:
                    y_train = json.loads(line)
        """
        self.xtrain = self.train_x
        y_train = self.train_y
        self.xtrain = pad_sequences(self.xtrain, maxlen=self.sent_len_num, padding='post')
        self.vocab_size += 1

        print('feature length:', self.sent_len_num)
        print('vocab size:', self.vocab_size)

        self.ytrain = np.zeros((len(self.xtrain), self.vocab_size), dtype=np.int32)
        for i in range(len(y_train)):
            self.ytrain[i, y_train[i]] = 1

        self.ytrain = np.array(self.ytrain)
        self.xtrain = np.array(self.xtrain)
        print(self.xtrain.shape)
        print(self.ytrain.shape)

    def process_data(self):

        self.sent_len_num = 0
        word_dict = {}
        with open('../data/ch_word_dict', 'w') as word_file:
            with open('../data/pku_training.utf8', 'r') as inf:
                for line in inf:
                    line = line.strip().split()
                    if len(line) > 200:
                        continue
                    self.sent_len_num = max(self.sent_len_num, len(line))
                    for word in line:
                        t = word_dict.setdefault(word, 0)
                        word_dict[word] = t + 1
            sorted_word_list = sorted(word_dict.items(), key = lambda d:d[1], reverse=True)
            for index, (word, num) in enumerate(sorted_word_list):
                print(word + '\t' + str(index + 1), file=word_file, end='\n')
        print('sentence len:', self.sent_len_num)

    def gene_feature(self):

        self.word_dict = {}
        self.idx2word = {}
        self.idx2word[0] = 'EOF'
        with open('../data/ch_word_dict', 'r') as inf:
            for line in inf:
                line = line.strip().split('\t')
                if len(line) != 2:
                    continue
                self.word_dict[line[0]] = int(line[1])

        self.train_x = []
        self.train_y = []
        count = 0
        with open('../data/sentence_train.data', 'w') as outf:
            with open('../data/pku_training.utf8', 'r') as inf:
                for line in inf:
                    line = line.strip().split()
                    l = len(line)
                    if l > 200:
                        continue
                    count += 1
                    if count % 10 == 0:
                        print(count)
                    #if count > 12:
                        #break
                    tmp = []
                    for i in range(l):
                        self.train_y.append(self.word_dict[line[i]])
                        for j in range(l):
                            if i != j:
                                tmp.append(self.word_dict[line[j]])
                        self.train_x.append(tmp[:])
            #print(json.dumps(self.train_x), file=outf, end='\n')
            #print(json.dumps(self.train_y), file=outf, end='\n')
                
    def function_model(self):
        vector_input = Input(shape=(200,), dtype='int32', name='vector_input')
        embedding = Embedding(self.vocab_size, output_dim=128)(vector_input)
        lstm = Bidirectional(LSTM(256, return_sequences=True), merge_mode='concat')(embedding)
        dropout1 = Dropout(0.5)(lstm)
        lstm1 = LSTM(128, return_sequences=False)(dropout1)
        #lstm1 = Bidirectional(LSTM(1024, kernel_regularizer=regularizers.l2(0.01)))(dropout1)
        #lstm = LSTM(1024)(embedding)
        dropout = Dropout(0.5)(lstm1)
        #dropout = Flatten()(dropout)
        output = Dense(self.vocab_size, activation='softmax')(dropout)
        #output = Dense(self.vocab_size, kernel_regularizer=regularizers.l2(0.01), activation='softmax')(dropout)
        parallel_model = Model(inputs=vector_input, outputs=output)
        #print(parallel_model.predict(self.xtrain[:10]).shape)
        parallel_model = multi_gpu_model(parallel_model, gpus=2)
        #parallel_model = multi_gpu_model(parallel_model, gpus=4)
        #parallel_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        #parallel_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        parallel_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        #parallel_model.fit(self.xtrain, self.ytrain, batch_size=128, epochs=20, validation_data=(self.xtest, self.ytest), callbacks=self.clkb, verbose=1)
        #parallel_model.fit(self.xtrain, self.ytrain, batch_size=256, epochs=10, validation_data=(self.xtest, self.ytest), callbacks=self.clkb)
        #parallel_model.fit(self.xtrain, self.ytrain, batch_size=256, epochs=10, callbacks=self.clkb, verbose=1)
        parallel_model.fit(self.xtrain, self.ytrain, batch_size=1024, epochs=100)
        #parallel_model.fit(self.xtrain, self.ytrain, batch_size=512, epochs=30, validation_split=0.2, callbacks=self.clkb, verbose=1)
        parallel_model.save('large_model.h5')

    def gene_x_feature(self, sen):
        
        s = sen.split()
        print(s)
        l = len(s)
        tmp = []
        x_feature = []
        y_feature = []
        for i in range(l):
            y_feature.append(self.word_dict[s[i]])
            for j in range(l):
                if i != j:
                    tmp.append(self.word_dict[s[j]])
            x_feature.append(tmp[:])
        x_feature = pad_sequences(x_feature, maxlen=200, padding='post')
        x_feature = np.array(x_feature)
        y_feature = np.array(y_feature)
        return x_feature, y_feature

    def predict(self):
        self.word_dict = {}
        self.idx2word = {}
        self.idx2word[0] = 'EOF'
        with open('../data/ch_word_dict', 'r') as inf:
            for line in inf:
                line = line.strip().split('\t')
                if len(line) != 2:
                    continue
                self.word_dict[line[0]] = int(line[1])
        model = load_model('my_model.h5')
        sentence = '北京 的 外交 工作 取得 了 重要 成果'
        sentence = '志 已 诚挚 的 问候 和 良好 的 祝愿'
        x, y = self.gene_x_feature(sentence)
        print(x.shape)
        print(y.shape)
        print(x, y)

        y_pred = model.predict(x)
        
        print(y_pred)
        y_pred = tf.convert_to_tensor(y_pred, np.float32)
        y_true = tf.convert_to_tensor(y, np.float32)
        val_loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        #print(np.asarray(val_loss, np.float32))
        with tf.Session() as sess:
            loss_list = sess.run(val_loss)
            print(loss_list)

    def run(self):
        self.gene_feature()
        self.read_data()
        self.function_model()
        #self.predict()
        #self.process_data()

if __name__ == "__main__":
    s = Solution()
    s.run()
