import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
import re
import random 
from bs4 import BeautifulSoup
import sys
import os
import nltk
import tensorflow as tf

import keras
from keras.models import load_model
from keras.utils import plot_model
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from nltk import word_tokenize, sent_tokenize
import pudb
from nltk import tokenize

os.environ['KERAS_BACKEND']='tensorflow'


MAX_SENT_LENGTH = 100
MAX_SENTS = 100
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2


with K.tf.device('/gpu:0'):
  
    data_dictionary = {}
    
    def getData():
        data_train = pd.read_csv('~/team2_dir/new_model/JK/DATA/train_test_Keras.txt', sep='\t', encoding='Latin-1', error_bad_lines=False)     
        user_file = pd.read_csv('~/team2_dir/new_model/JK/DATA/userList.txt', sep='\t', encoding='Latin-1', error_bad_lines=False)
        user_file.user= user_file.user.astype(str)
        drug_file = pd.read_csv('~/team2_dir/new_model/JK/DATA/drugList.txt', sep='\t', encoding='Latin-1', error_bad_lines=False)
        drug_file.drug= drug_file.drug.astype(str)
         
        user_rand = random.randint(0,user_file.shape[0])
        drug_rand = random.randint(0,drug_file.shape[0])
         
        user_list=[]
        for idx in range(user_file.shape[0]):
            text = user_file.user[idx]
            user_list.append(text)  

        random_user = str(user_list[user_rand])
           
        drug_list=[]
        for idx in range(drug_file.shape[0]):
            text = drug_file.drug[idx]
            drug_list.append(text)
         
        random_drug = str(drug_list[drug_rand])
        
        random_review = []
        random_sentiment = []
        random_review = data_train.review[drug_rand]
        random_sentiment = data_train.sentiment[drug_rand]
        
        return random_user, random_drug, random_review, random_sentiment
        
        
    def execute_model(data_dic):
        def clean_str(string):
            string = re.sub(r"\\", "", string)    
            string = re.sub(r"\'", "", string)    
            string = re.sub(r"\"", "", string)  
            string = re.sub(r"\n", "", string)
            string = re.sub(r"<sssss>", "", string)  
            return string.strip().lower()
    
    
        user_file = pd.read_csv('~/team2_dir/new_model/JK/DATA/userList.txt', sep='\t', encoding='Latin-1', error_bad_lines=False)
        user_file.user= user_file.user.astype(str)
        drug_file = pd.read_csv('~/team2_dir/new_model/JK/DATA/drugList.txt', sep='\t', encoding='Latin-1', error_bad_lines=False)
        drug_file.drug= drug_file.drug.astype(str)
         
        user_list=[]
        for idx in range(user_file.shape[0]):
            text = user_file.user[idx]
            user_list.append(text)

        user_length = user_file.shape[0]
        
        user_dict = []
        for i, u in enumerate(user_list):
            user_dict.append([u,i]) 
        user_dict = dict(user_dict)
        

        i_user_list=[]
        for u in user_list:
            i_user_list.append(user_dict[u])

        
        drug_list=[]
        for idx in range(drug_file.shape[0]):
            text = drug_file.drug[idx]
            drug_list.append(text)
            
        drug_length = drug_file.shape[0]


        drug_dict = [] 
        for i, d in enumerate(drug_list):
            drug_dict.append([d,i])   
        drug_dict = dict(drug_dict)
        
        
        i_drug_list=[]
        for d in drug_list:
            i_drug_list.append(int(drug_dict[d]))    

    
        reviews = []
        labels = []
        texts = []
        users = []
        drugs = []

        text = data_dic['test_review']
        text = clean_str(text)
        texts.append(text)
        sentences = tokenize.sent_tokenize(text) 
        reviews.append(sentences)
           
        labels.append(int(data_dic['test_sentiment'])-1)
        users.append(data_dic['test_user'])
        drugs.append(data_dic['test_drug'])
        
        
        i_users=[]
        for u in users:
            i_users.append(user_dict[str(u)])
            
        i_drugs=[]
        for d in drugs:
            i_drugs.append(drug_dict[d])
            
    
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
        tokenizer.fit_on_texts(texts)

        data = np.zeros((len(texts), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
        
        
        for i, sentences in enumerate(reviews):
            for j, sent in enumerate(sentences):
                if j< MAX_SENTS:
                    wordTokens = text_to_word_sequence(sent)
                    k=0
                    for _, word in enumerate(wordTokens):
                        if k<MAX_SENT_LENGTH and tokenizer.word_index[word]<MAX_NB_WORDS:
                            data[i,j,k] = tokenizer.word_index[word]
                            k=k+1      

        word_index = tokenizer.word_index
        index_word = {v:k for k,v in tokenizer.word_index.items()}    
        labels = to_categorical(np.asarray(labels))
 
        GLOVE_DIR = "/home/team2/team2_dir/new_model/theano-Keras_model"
        embeddings_index = {}
        f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        

    # building Hierachical Attention network + UPA
        class AttLayer(Layer):
            def __init__(self, **kwargs):   
                super(AttLayer, self).__init__(**kwargs)
    
            def build(self, input_shape):
                assert len(input_shape)==3
                self.W = K.variable(np.random.random(input_shape[-1], ))
                self.Wu = K.variable(np.random.rand(EMBEDDING_DIM, 100))
                self.Wd = K.variable(np.random.rand(EMBEDDING_DIM, 100))
                self.b = K.variable(np.zeros(EMBEDDING_DIM,))
                
                self.trainable_weights = [self.W, self.Wu, self.Wd, self.b]
                super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!
    
            def call(self, x, mask=None):
                if(K.int_shape(x)[1]==100):
                    attenu = K.dot(ulookup, self.Wu)
                    attend = K.dot(dlookup, self.Wd)
                else:
                    attenu = K.dot(ulookup2, self.Wu)
                    attend = K.dot(dlookup2, self.Wd)
                eij = K.tanh(K.squeeze(K.dot(x, K.expand_dims(self.W)), axis=-1) + attenu + attend + self.b)        
                ai = K.exp(eij)
                results = tf.expand_dims(K.sum(ai, axis=1), 1)
                weights = ai/results
                mask = tf.expand_dims(weights,2)
                weighted_input = x * mask
                return tf.reduce_sum(weighted_input, 1)
            
    
            def compute_output_shape(self, input_shape):
                return (input_shape[0], input_shape[-1])

        embedding_matrix = np.random.rand(29007,EMBEDDING_DIM)
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
                
        embedding_layer = Embedding(29007,
                                    EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=MAX_SENT_LENGTH,
                                    trainable=True)
        
        # Need Embedding Initialization
        fixed_user_input = Input(tensor = K.variable(i_user_list))
        embedding_user = Embedding(user_length, EMBEDDING_DIM, trainable=True)(fixed_user_input)
        embedding_user = tf.squeeze(embedding_user)
    
        fixed_drug_input = Input(tensor = K.variable(i_drug_list))
        embedding_drug = Embedding(drug_length, EMBEDDING_DIM, trainable=True)(fixed_drug_input)
        embedding_drug = tf.squeeze(embedding_drug)
        
        user_list_input= Input(shape=(1,), dtype='int32', name='user_list_input')
        drug_list_input= Input(shape=(1,), dtype='int32', name='drug_list_input')
    
        ulookup = tf.nn.embedding_lookup(params=embedding_user, ids=user_list_input)
        ulookup = tf.squeeze(ulookup, axis=1)
    
        dlookup = tf.nn.embedding_lookup(params=embedding_drug, ids=drug_list_input)
        dlookup = tf.squeeze(dlookup, axis=1)
    
        user_list_input2 = K.repeat_elements(user_list_input, MAX_SENTS, axis=0)
        drug_list_input2 = K.repeat_elements(drug_list_input, MAX_SENTS, axis=0)
    
        ulookup2 = tf.nn.embedding_lookup(params=embedding_user, ids=user_list_input2)
        ulookup2 = tf.squeeze(ulookup2, axis=1)
    
        dlookup2 = tf.nn.embedding_lookup(params=embedding_drug, ids=drug_list_input2)
        dlookup2 = tf.squeeze(dlookup2, axis=1)
    
        sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
       
        embedded_sequences = embedding_layer(sentence_input)
        l_lstm = Bidirectional(GRU(100, dropout=0.2, return_sequences=True))(embedded_sequences)
        l_dense = TimeDistributed(Dense(200, activation='relu'))(l_lstm)
        l_att = AttLayer()(l_dense)
        sentEncoder = Model(sentence_input, l_att)
        
        review_input = Input(shape=(MAX_SENTS,MAX_SENT_LENGTH), dtype='int32', name='review_input')
        # pudb.set_trace()
        review_encoder = TimeDistributed(sentEncoder)(review_input)
        l_lstm_sent = Bidirectional(GRU(100, dropout=0.2, return_sequences=True))(review_encoder)
        l_dense_sent = TimeDistributed(Dense(200, activation='relu'))(l_lstm_sent)
        l_att_sent = AttLayer()(l_dense_sent)
    
       
        preds = Dense(10, activation='softmax', name='output')(l_att_sent)
        
        
        model = Model(inputs=[review_input, user_list_input,drug_list_input], outputs=preds)

        model.load_weights('trained_weights.h5')
        
        
        for i in range(1):
    
            xtest = data[i].reshape(1,100,100)
            ylabel = labels[i]
            ypred = model.predict(x={'review_input': data, 'user_list_input': np.array(i_users), 'drug_list_input': np.array(i_drugs)})[i]
            sent = " ".join([index_word[x_] for x_ in xtest[0].reshape(10000) if x_!=0])
         
            print( '\nOriginal score:',np.argmax(ylabel)+1, ', Predict score:', np.argmax(ypred)+1,'\n', sent)
            
        return np.argmax(ylabel)+1, np.argmax(ypred)+1
     
     
    # Running code inside team2_demo.py         
    #data_dictionary['test_user'], data_dictionary['test_drug'], data_dictionary['test_review'], data_dictionary['test_sentiment'] = getData()  
    #print(data_dictionary)    

    #result = execute_model(data_dictionary)
    
    import gc; gc.collect()
        


