from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import numpy as np
import json
import random
import glob

def create_dataset(base_topics):
    data = []
    for i in base_topics:
        for j in glob.glob(f'./20news-18828/{i}/*'):
            with open(j, 'r', encoding='cp1252') as f:
                data.append(f.read())
    with open('data.json', 'w', encoding='utf8') as f:
        f.write(json.dumps(data))

def load_data():
    corpus = []
    with open('data.json', 'r', encoding='utf8') as f:
        corpus = json.loads(f.read())
        random.shuffle(corpus)
    return corpus

def build_LDA(data, vectorizer, n_features, n_components):
    tf = vectorizer.fit_transform(data)
    model = LatentDirichletAllocation(n_components=n_components, random_state=1).fit(tf)
    return model

def build_LSA(data, vectorizer, n_features, n_components, algorithm='arpack'):
    tf = vectorizer.fit_transform(data)
    model = TruncatedSVD(n_components=n_components, random_state=1, algorithm=algorithm).fit(tf)
    return model

def get_model_topics(model, vectorizer, topics, n_top_words=10):
    word_dict = {}
    feature_names = vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        word_dict[topics[topic_idx]] = top_features
    return pd.DataFrame(word_dict)

if __name__=='__main__':
    n_features = 1000
    n_components = 6
    n_top_words = 20
    topics = ['comp.sys.ibm.pc.hardware', 'soc.religion.christian', 'talk.politics.mideast', 'rec.sport.baseball', 'sci.space', 'talk.politics.guns']

    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=0.05, max_features=n_features, stop_words='english', ngram_range=(1, 2))
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=0.05, max_features=n_features, stop_words='english', ngram_range=(1, 2))
    # Create and load data
    create_dataset(base_topics=topics)
    corpus = load_data()
    # Build models
    lda = build_LDA(data=corpus, vectorizer=tf_vectorizer,    n_features=n_features, n_components=n_components)
    lsa = build_LSA(data=corpus, vectorizer=tfidf_vectorizer, n_features=n_features, n_components=n_components, algorithm='arpack')
    # Print results
    lda_topics = get_model_topics(model=lda, vectorizer=tf_vectorizer,    topics=topics, n_top_words=10)
    lsa_topics = get_model_topics(model=lsa, vectorizer=tfidf_vectorizer, topics=topics, n_top_words=10)
    print(lda_topics)
    print(lsa_topics)