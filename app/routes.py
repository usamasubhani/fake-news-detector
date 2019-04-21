from flask import render_template, request, url_for
from app import app
import numpy as np
from app.funcs import Data, Train
from sklearn.metrics.pairwise import cosine_similarity
#import tensorflow as tf
from tensorflow import get_default_graph
from tensorflow.keras.models import load_model
#from tensorflow.keras import backend as K
import time


loaded = False
#def load_model():
 #   global model
  #  model = load_model('Model.h5')
    # this is key : save the graph after loading the model
   # global graph
    #graph = tf.get_default_graph()
# Dataset directories
#load_model()


@app.route('/')
@app.route('/index')
def Home():
    return render_template('index.html')

@app.route('/form')
def Form():
    return render_template('form.html', title='Form')



@app.route('/features', methods=['POST'])
def Features():
    input = request.form
    h_bow = bow_vectorizer.transform([input['head']]).toarray()
    h_tf = tfreq_vectorizer.transform(h_bow).toarray()[0].reshape(1, -1)
    h_tfidf = tfidf_vectorizer.transform([input['head']])


    b_bow = bow_vectorizer.transform([input['body']]).toarray()
    b_tf = tfreq_vectorizer.transform(b_bow).toarray()[0].reshape(1, -1)
    b_tfidf = tfidf_vectorizer.transform([input['body']])

    tfidf_cos = cosine_similarity(h_tfidf.toarray().reshape(1, -1), b_tfidf.toarray().reshape(1, -1))[0].reshape(1,1)
    feat_vec = np.squeeze(np.c_[h_tf, b_tf, tfidf_cos])
    with graph.as_default():
        stance = model.predict_classes(np.array([feat_vec]))
        probabilities = model.predict(np.array([feat_vec]))
    label_ref_rev = {0: 'agree', 1: 'disagree', 2: 'discuss', 3: 'unrelated'}

    import pandas as pd
    weights = np.asarray(h_tfidf.mean(axis=0)).ravel().tolist()
    h_weights_df = pd.DataFrame({'Term': tfidf_vectorizer.get_feature_names(), 'Weight': weights})
    h_weights_df.sort_values(by=['Weight'], ascending=False)
    h_weights_df = h_weights_df[h_weights_df.Weight != 0]
    # h_weights_df.set_index('Weight', inplace=True)

    weights = np.asarray(b_tfidf.mean(axis=0)).ravel().tolist()
    b_weights_df = pd.DataFrame({'Term': tfidf_vectorizer.get_feature_names(), 'Weight': weights})
    b_weights_df.sort_values(by=['Weight'], ascending=False)
    b_weights_df = b_weights_df[b_weights_df.Weight != 0]

    s=[]
    for i in range(0,4):
        s.append(label_ref_rev[i])
    # probabilities = pd.DataFrame({'Stance': probabilities[0,:], 'Probability': probabilities[1, :].to})
    p = np.asarray(probabilities).ravel().tolist()
    probabilities = pd.DataFrame({'Stance': s, 'Probability': p})
    np.set_printoptions(precision=4)

    return render_template('features.html', result = [label_ref_rev[ stance[0] ], h_weights_df, b_weights_df, probabilities ])

@app.route('/ajax/index')
def ajax_index():
    global loaded
    if (loaded):
        return '<p>Model is Loaded!</p>'
    file_train_instances = "Dataset/train_stances.csv"
    file_train_bodies = "Dataset/train_bodies.csv"
    file_test_instances = "Dataset/test_stances_unlabeled.csv"
    file_test_bodies = "Dataset/test_bodies.csv"
    # Load Data
    raw_train = Data(file_train_instances, file_train_bodies)
    raw_test = Data(file_test_instances, file_test_bodies)

    global bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer
    bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer \
    = Train(raw_train, raw_test, lim_unigram=5000)
    del raw_train
    del raw_test
    global model
    model = load_model('Model.h5')
    loaded = True
    global graph
    graph = get_default_graph()
    return '<p>Model Loaded!</p>'

@app.route('/ajax/stance')
def ajax_stance():
    time.sleep(5)
    return '<h1> Stance: </h1>'

