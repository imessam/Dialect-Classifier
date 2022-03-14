from flask import Flask,render_template,request
import numpy as np
import pickle
import tensorflow as tf
from utils import *
from processor import PreProcessor

app = Flask(__name__)
preprocessor = PreProcessor(w2vPath = "weights/full_grams_cbow_100_twitter.mdl",isRemove = True,isLoad = True)
preTrainedEmbed = preprocessor.loadPreTrainedEmbeddings()
print(preTrainedEmbed.shape)

idx2labels = pickle.load(open("idx2labels.pkl","rb"))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict/', methods=['GET','POST'])
def predict():
    
    if request.method == "POST":
        #get form data
        ara_text = request.form.get('ara_text')
        
        #call preprocessDataAndPredict and pass inputs
        try:
            prediction = preprocessDataAndPredict(ara_text)
            
            #pass prediction to template
            return render_template('predict.html', prediction = prediction)
   
        except ValueError:
            return "Please Enter valid values"
  
        pass
    pass
def preprocessDataAndPredict(text):
    
    #keep all inputs in array
    test_data = [text]
    print(test_data)
    
    #convert value data into numpy array
    test_data = np.array(test_data,dtype="str")
    
    #reshape array
    
    test_data =preprocessor.cleanTweets(test_data,isRemove=True)
    tweetsTokens,masks = preprocessor.tokenizeTweets(test_data)   
    sentVecs = preprocessor.sentToVec(tweetsTokens,preTrainedEmbed)

    #load trained model
    pca = pickle.load(open("models/pca.pkl", "rb" ))
    svm = pickle.load(open("models/svm.pkl", "rb" ))
    rnn  = tf.keras.models.load_model("models/dialectClassifierRNN")
        
    sentVecsRed = pca.transform(sentVecs)

    #predict
    predictions = []
    predictions.append(svm.predict(sentVecsRed)[0])
    predictions.append(idx2labels[np.argmax(rnn.predict(tweetsTokens))])
           
    return predictions
    
    pass

if __name__ == '__main__':
    app.run(debug=True)