import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



class DialectClassifier(tf.keras.Model):
    

    def __init__(self,voc_len,max_sen_len,embed_dim,hidden_dim,no_classes,isPreTrainedEmbed=False,pretrainedWeights = None):
        super().__init__()
                
        if isPreTrainedEmbed:
            weights=tf.keras.initializers.Constant(pretrainedWeights)
            self.embed_layer = Embedding(input_dim=voc_len,output_dim = 
                                         embed_dim,input_length = max_sen_len ,mask_zero = True,
                                         embeddings_initializer = weights,trainable=False)
        else:
            self.embed_layer = Embedding(input_dim=voc_len,output_dim = embed_dim,input_length = 
                                         max_sen_len ,mask_zero = True)
        self.lstm_layer = Bidirectional(LSTM(units = hidden_dim,dropout = 0.4))
        self.classifier = tf.keras.models.Sequential([
            Dense(units = hidden_dim,activation = "relu"),
            BatchNormalization(),
            Dense(units = no_classes,activation = "softmax")
        ])

    def call(self, inputs):
        embeddings = self.embed_layer(inputs)
        #print(embeddings.shape)

        hiddens = self.lstm_layer(embeddings)
        #print(hiddens.shape)

        preds = self.classifier(hiddens)
        #print(preds.shape)
        
        return preds

