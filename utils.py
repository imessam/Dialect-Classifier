import numpy as np
import pandas as pd
import requests
import sklearn
from time import time
from sklearn.metrics import accuracy_score





def readCSV(filename,asNumpy = False,dtype = None):
    
    file = pd.read_csv(filename)
    
    if asNumpy:
        file = np.array(file,dtype=dtype)
        
    return file


def saveCSV(numpy_arr,filename,isArabic = False):
    
    encoding = None
    
    if isArabic:
        encoding ='utf-8-sig'
    
    pd.DataFrame(numpy_arr).to_csv(filename,index=False,encoding=encoding)
    

def get_tweets(ids,ids_batch_size = 1000):
    
    tweets = []
    url = 'https://recruitment.aimtechnologies.co/ai-tasks'
    no_batches = int(ids.shape[0]/ids_batch_size)
    
    for batch in range(no_batches+1):
        begin = batch*ids_batch_size
        if batch == no_batches:         
            end = ids.shape[0]
        else:
            end = (batch*ids_batch_size)+ids_batch_size
    
        json = ids[begin:end].tolist()
        req = requests.post(url = url, json=json)
        
        while(req.status_code != 200):
            print("no response",req.status_code)
            req = requests.post(url = url, json=json)
            
        print(f"completed batch {batch+1}/{no_batches+1}")        
        resp = list(map(req.json().get,json))
        
        tweets += resp
        
    return np.array(tweets)



def labelsEncoder(labels):
    
    labels2idx = {}
    idx2labels = {}
    count = 0
    no_classes = 0
    
    labelsEncoded = np.zeros((labels.shape[0]),dtype = np.int32)
    
    for i in range(labels.shape[0]):
        if labels[i] not in labels2idx.keys():
            labels2idx[labels[i]] = count
            idx2labels[count] = labels[i]
            count += 1
            
        labelsEncoded[i] = labels2idx[labels[i]]
    
    no_classes = count
    
    return labelsEncoded,no_classes,labels2idx,idx2labels



def oneHotEncoder(labels,no_classes):
    
    oneHotLabels = np.zeros((labels.shape[0],no_classes))
    
    for i in range(labels.shape[0]):
        oneHotLabels[i,labels[i]] = 1
    
    return oneHotLabels



def getStats(data):
    
    values, counts = np.unique(data, return_counts=True)
    
    for i in range(values.shape[0]):
        print(values[i]," : ",counts[i])


def splitData(X,Y,masks=None,splitPercent = 0.7,isMask = True):
    
    dataSize = X.shape[0]
    
    trainSize = int(splitPercent*dataSize)
    
    X_train,Y_train = X[:trainSize],Y[:trainSize]
    X_test,Y_test = X[trainSize:],Y[trainSize:]
    
    if isMask:
        
        masks_train = masks[:trainSize]
        masks_test = masks[trainSize:]        
    
        return X_train,X_test,Y_train,Y_test,masks_train,masks_test
    
    else:
        return X_train,X_test,Y_train,Y_test



def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}
    
    start = time() # Get start time
    learner = learner.fit(X_train[:sample_size],y_train[:sample_size])
    end = time() # Get end time
    
    results['train_time'] = end-start
        

    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:sample_size])
    end = time() # Get end time
    
    results['pred_time'] = end-start
            
    results['acc_train'] = accuracy_score(y_train[:sample_size],predictions_train)
        
    results['acc_test'] = accuracy_score(y_test,predictions_test)
       
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
        
    return learner,results
    