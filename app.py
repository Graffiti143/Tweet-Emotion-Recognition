#Emotion pks
import preprocessor as p
import numpy as np 
import pandas as pd 
import emoji
import keras
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.layers import BatchNormalization
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU,SimpleRNN
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import transformers
from transformers import TFAutoModel, AutoTokenizer
from tqdm.notebook import tqdm
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors
from tqdm import tqdm

#Saving and Loding Module pks
from tensorflow.keras.models import model_from_json

#Tweets Extration pks
import tweepy
from tweepy import OAuthHandler
import pandas as pd

#Flask pks
from flask import Flask, redirect, url_for, request, render_template
import matplotlib.pyplot as plt


app = Flask(__name__)

#####################################################

def working(q,c):
    data = pd.read_csv("datasets/text_emotion.csv")
    misspell_data = pd.read_csv("datasets/aspell.txt",sep=":",names=["correction","misspell"])
    misspell_data.misspell = misspell_data.misspell.str.strip()
    misspell_data.misspell = misspell_data.misspell.str.split(" ")
    misspell_data = misspell_data.explode("misspell").reset_index(drop=True)
    misspell_data.drop_duplicates("misspell",inplace=True)
    miss_corr = dict(zip(misspell_data.misspell, misspell_data.correction))

    def misspelled_correction(val):
        for x in val.split(): 
            if x in miss_corr.keys(): 
                val = val.replace(x, miss_corr[x]) 
        return val

    contractions = pd.read_csv("datasets/contractions.csv")
    cont_dic = dict(zip(contractions.Contraction, contractions.Meaning))
    def cont_to_meaning(val): 
  
        for x in val.split(): 
            if x in cont_dic.keys(): 
                val = val.replace(x, cont_dic[x]) 
        return val

    p.set_options(p.OPT.MENTION, p.OPT.URL, p.OPT.HASHTAG)

    def punctuation(val): 
        punctuations = '''()-[]{};:'"\,<>./@#$%^&_~'''
        for x in val.lower(): 
            if x in punctuations: 
                val = val.replace(x, " ") 
        return val
    
    def clean_text(val):
        val = misspelled_correction(val)
        val = cont_to_meaning(val)
        val = p.clean(val)
        val = ' '.join(punctuation(emoji.demojize(val)).split())

        return val    

    sent_to_id  = {"empty":0, "sadness":1,"enthusiasm":2,"neutral":3,"worry":4,
                        "surprise":5,"love":6,"fun":7,"hate":8,"happiness":9,"boredom":10,"relief":11,"anger":12}

    data["clean_content"]=data['content'].apply(lambda x : clean_text(x))
    data = data[data.clean_content != ""]
    data["sentiment_id"] = data['sentiment'].map(sent_to_id)

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(data.sentiment_id)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    Y = onehot_encoder.fit_transform(integer_encoded)
    X_train, X_test, y_train, y_test = train_test_split(data.clean_content,Y, random_state=1995, test_size=0.2, shuffle=True)

    token = text.Tokenizer(num_words=None)
    max_len = 160
    token.fit_on_texts(list(X_train) + list(X_test))
    X_train_pad = sequence.pad_sequences(token.texts_to_sequences(X_train), maxlen=max_len)
    X_test_pad = sequence.pad_sequences(token.texts_to_sequences(X_test), maxlen=max_len)

    def tmp(r):
        rr = r.sort_values('percentage',ascending=False,ignore_index=True)
        return rr['sentiment'][0]

    def get_sentiment(model,text):
        text = clean_text(text)
        twt = token.texts_to_sequences([text])
        twt = sequence.pad_sequences(twt, maxlen=max_len, dtype='int32')
        sentiment = model.predict(twt,batch_size=1,verbose = 2)
        sent = np.round(np.dot(sentiment,100).tolist(),0)[0]
        result = pd.DataFrame([sent_to_id.keys(),sent]).T
        result.columns = ["sentiment","percentage"]
        result=result[result.percentage !=0]
        return tmp(result)  

    json_file = open('datasets/emotion1.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("datasets/model1.h5")

    access_token = '868158714-mthcwpcduyPODo1iYcBx0dYpdJjieE5bPZDyC5S8'
    access_token_secret = 'XH2nB56BFkWJLWhMv3KIpQzer9VRJOZsnOCioYP7UCC0B'
    consumer_key = 'utJ8k3max3uLccjZfwN8v7E4Z'
    consumer_secret = 'zwrFCfafBuNYC1Erq41SP28vL5otNYK77FNn4dcTjUSespwhsu'

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth,wait_on_rate_limit=True)

    #inputs
    # text_query = input("Enter the Tweet or Search word : ")
    # count = int(input("Count : "))
    text_query = q
    count = c
    td = pd.DataFrame()

    try:
        tweets = tweepy.Cursor(api.search_tweets,q=text_query,lang='en').items(count) #, since='2021-06-07'
        tweets_list = [tweet.text for tweet in tweets]
        tweets_df = pd.DataFrame(tweets_list)
        td = tweets_df
        # print(tweets_df.head())
    
    except BaseException as e:
        print('failed on_status,',str(e))

    td["clean_content"]=td[0].apply(lambda x : clean_text(x))
    td = td[td.clean_content != ""]
    td['emotion'] = td.clean_content.apply(lambda x : get_sentiment(loaded_model,x))

    #MAJOR EMOTION
    me = td['emotion'].mode()[0]

    # EMOTION GRAPH
    fig, ax = plt.subplots()
    td['emotion'].value_counts().plot(ax=ax, kind='pie',title=' ')
    plt.savefig("static/output.jpg")

    return me


#####################################################

@app.route('/result',methods = ['POST'])
def result():
# Here load the module and take inputs and prdiect output
    lst=[x for x in request.form.values()]
    me = working(lst[0],int(lst[1]))

    return render_template('index.html',me=me,op=True)
    
    

@app.route('/',methods = ['POST', 'GET'])
def index():
# Here home page with textboxes for taking inputs 
    return render_template('index.html')
    




if __name__ == '__main__':
    app.run(debug = True)

