import numpy as np
import regex as re
import matplotlib.pyplot as plt
import random
import gensim
import pickle


class PreProcessor:
    
    def __init__(self,w2vPath,isRemove=False,isLoad = False):
        
        
        self.word2vec = None
        print('Loading Word2vec')

        self.word2vec = gensim.models.Word2Vec.load(w2vPath).wv
        
        if isLoad:
            data =  pickle.load(open( "data.p", "rb" ))
            self.VOC = data["len_voc"]
            self.LEN_VOC = data["len_voc"]
            self.MAX_SEN_LEN = data["max_sen_len"]
            self.word2idx = data["word2idx"]
            self.idx2word = data["idx2word"]
            self.wordCount = data["wordCount"]
                       
        else:
            self.VOC = ["<PAD>","<UNK>"]
            self.LEN_VOC = 1
            self.MAX_SEN_LEN = 0

            self.word2idx = {}
            self.idx2word = {}
            self.wordCount = {}


            self.word2idx["<PAD>"] = 0
            self.idx2word[0] = "<PAD>"
            self.wordCount["<PAD>"] = 0

            self.word2idx["<UNK>"] = 1
            self.idx2word[1] = "<UNK>"
            self.wordCount["<UNK>"] = 1

            if not isRemove:
                self.word2idx["@USER"] = 1
                self.word2idx["@URL"] = 2
                self.word2idx["@HASH"] = 3
                self.word2idx["@EMOJI"] = 4

                self.idx2word[1] = "@USER"
                self.idx2word[2] = "@URL"
                self.idx2word[3] = "@HASH"
                self.idx2word[4] = "@EMOJI"
                self.LEN_VOC += 4
       
        self.isRemove = isRemove
    
    
    def identifyMentions(self,tweet,isRemove = False):
        tweet+=" "
        sub = " @USER "
        if isRemove:
            sub = " "
        return re.sub("@([\w]+)\s",sub,tweet)[:-1]


    def identifyURL(self,tweet,isRemove = False):
        sub = " @URL "
        if isRemove:
            sub = " "
        return re.sub("(http[s]*://[\w/.:\-\=]+)|(www.[\w/.:\-\=]+)",sub,tweet)


    def identifyHashTag(self,tweet,isRemove = False):
        sub = " @HASH "
        if isRemove:
            sub = " "
        return re.sub("#([\w _]+)",sub,tweet)
    
    
    def identifyEmojis(self,tweet,isRemove = False):
        sub = " @EMOJI "
        if isRemove:
            sub = " "
        regrex_pattern =  re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002500-\U00002BEF"  # chinese char
            u"\U00002702-\U000027B0"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u"\U00010000-\U0010ffff"
            u"\u2640-\u2642" 
            u"\u2600-\u2B55"
            u"\u200d"
            u"\u23cf"
            u"\u23e9"
            u"\u231a"
            u"\ufe0f"  # dingbats
            u"\u3030"
                          "]+", re.UNICODE)
        return regrex_pattern.sub(sub,tweet)
    

    def removePunctuations(self,tweet):
        puncs = [".",",","\'","?","!","_","-","+","*","،","/","\\","؟","=","$","&"]
        newTweet = ""
        for i in range(len(tweet)):
            if tweet[i] not in puncs:
                newTweet+=tweet[i]
        return newTweet

    def removeEmpty(self,tweet):
        cleaned_tweet = ""
        words = tweet.split(" ")
        
        for word in words:
            if len(word)>0:
                cleaned_tweet += word+" "
                
        return cleaned_tweet[:-1]
    
    def clean_str(self,tweet):
        search = ["أ","إ","آ","ة","_","-","/",".","،"," و "," يا ",'"',"ـ","'","ى","\\",'\n', '\t','&quot;','?','؟','!']
        replace = ["ا","ا","ا","ه"," "," ","","",""," و"," يا","","","","ي","",' ', ' ',' ',' ? ',' ؟ ',' ! ']

        #remove tashkeel
        p_tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
        cleanedTweet = re.sub(p_tashkeel,"", tweet)

        #remove longation
        p_longation = re.compile(r'(.)\1+')
        subst = r"\1\1"
        cleanedTweet = re.sub(p_longation, subst, cleanedTweet)

        cleanedTweet = cleanedTweet.replace('وو', 'و')
        cleanedTweet = cleanedTweet.replace('يي', 'ي')
        cleanedTweet = cleanedTweet.replace('اا', 'ا')

        for i in range(0, len(search)):
            cleanedTweet = cleanedTweet.replace(search[i], replace[i])

        #trim    
        cleanedTweet = cleanedTweet.strip()

        return cleanedTweet

    
    def cleanTweet(self,tweet,isRemove = False):
        
        
        cleaned_tweet = self.identifyMentions(tweet.lower(),isRemove)
        cleaned_tweet = self.identifyURL(cleaned_tweet,isRemove)
        cleaned_tweet = self.identifyHashTag(cleaned_tweet,isRemove)
        cleaned_tweet = self.identifyEmojis(cleaned_tweet,isRemove)
        cleaned_tweet = self.clean_str(cleaned_tweet)
        cleaned_tweet = self.removePunctuations(cleaned_tweet)
        cleaned_tweet = self.removeEmpty(cleaned_tweet)
        

        return cleaned_tweet
    
    def cleanTweets(self,tweets,isRemove = False):
        
        cleanedTweets = np.copy(tweets)
        
        for i,tweet in enumerate(tweets):
            
            cleanedTweet = self.cleanTweet(tweet,self.isRemove)
            cleanedTweets[i] = cleanedTweet
            
        return cleanedTweets

    
    def setMaxLen(self,tweets):
        
        for i,tweet in enumerate(tweets):

            self.MAX_SEN_LEN = max(self.MAX_SEN_LEN,len(tweet.split(" ")))        
    
    
    def padTweet(self,tweet):
        
        paddedTweet = tweet
        
        for _ in range(self.MAX_SEN_LEN-len(tweet.split(" "))):
            paddedTweet += " <PAD>"
    
        return paddedTweet
         
    
    def processTweet(self,tweet):

        words = tweet.split(" ")

        for word in words:
            if word not in self.word2idx:
                self.LEN_VOC += 1
                self.word2idx[word] = self.LEN_VOC
                self.idx2word[self.LEN_VOC] = word
                self.VOC.append(word)
                self.wordCount[word] = 0
            self.wordCount[word] += 1
            
            

    def buildVocabulary(self,tweets,isSave = False):
        
        self.setMaxLen(tweets)

        for i,tweet in enumerate(tweets):
            
            paddedTweet = self.padTweet(tweet)
            self.processTweet(paddedTweet)
        
        if isSave:
            data = {"voc":self.VOC,"len_voc":self.LEN_VOC,"max_sen_len":self.MAX_SEN_LEN,
                    "word2idx":self.word2idx,"idx2word":self.idx2word,"wordCount":self.wordCount}
            pickle.dump(data, open( "data.p", "wb" ) )



    def tokenizeTweet(self,tweet):

        words = tweet.split(" ")
        tokens = []
        mask = []

        for word in words:
            if word not in self.word2idx:
                word = "<UNK>"
            tokens.append(self.word2idx[word])
            if self.word2idx[word] == 0 :
                mask.append(0)
            else:
                mask.append(1)

        return tokens,mask
        
    
    def tokenizeTweets(self,tweets):
        
        tweetsTokens = np.zeros((tweets.shape[0],self.MAX_SEN_LEN),dtype=np.int32())
        tweetsMasks = np.zeros((tweets.shape[0],self.MAX_SEN_LEN),dtype=np.bool_())
        
        for i in range(tweets.shape[0]):
            paddedTweet = self.padTweet(tweets[i])
            tokens,mask = self.tokenizeTweet(paddedTweet)
            tweetsTokens[i] = tokens
            tweetsMasks[i] = mask
            
        return tweetsTokens,tweetsMasks
    
    def loadPreTrainedEmbeddings(self):
                            
        embedding_matrix = np.zeros((self.LEN_VOC+1, self.word2vec.vector_size))

        for i in self.idx2word:
            if self.idx2word[i] in self.word2vec:
                embedding_matrix[i] = self.word2vec[self.idx2word[i]]
            elif self.wordCount[self.idx2word[i]] >= 5:
                embedding_matrix[i] = np.random.uniform(-0.25, 0.25, self.word2vec.vector_size)
            else:
                pass

        return embedding_matrix
    
  
    def sentToVec(self,tweetsTokens,embeddings):
        
        sentVecs = np.zeros((tweetsTokens.shape[0],self.word2vec.vector_size))
            
        for i in range(tweetsTokens.shape[0]):
            for j in range(tweetsTokens.shape[1]):
                embedding = embeddings[tweetsTokens[i,j]]
                sentVecs[i] += embedding
            sentVecs[i] = sentVecs[i] / np.sqrt(sentVecs[i].dot(sentVecs[i]))
            
        return sentVecs
                
                
        
