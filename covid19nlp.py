#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import json
import glob
from scipy import stats
import matplotlib.pyplot as plt
import spacy
from sklearn.metrics import mean_squared_error
from math import sqrt
from operator import itemgetter
from yellowbrick.text import TSNEVisualizer


# # Download data from Kaggle and load

# In[2]:


root = 'C:/Users/brian/Documents/GitHub/COVID19_RESEARCH/data'
meta_path = os.path.join(root, 'metadata.csv')

meta_df = pd.read_csv(meta_path)
meta_df.head()


# # Clean the data for processing

# In[3]:


meta_df.shape


# In[4]:


meta_df = meta_df[['cord_uid', 'title', 'abstract', 'publish_time', 'authors', 'url']]
meta_df.isnull().sum()


# In[5]:


meta_df = meta_df.dropna(subset=['title', 'abstract'])
meta_df.shape


# In[6]:


check_duplicates = ['cord_uid','title', 'abstract', 'authors']

for head in check_duplicates:
    meta_df = meta_df.drop(meta_df[meta_df[f'{head}'].duplicated()].index)

#Check duplicates are empty
[(meta_df[meta_df[f'{head}'].duplicated()]) for head in check_duplicates]


# In[7]:


meta_df.shape


# In[8]:


text = meta_df


# In[9]:


index, value = zip(*[(index, len(val)) for index, val in enumerate(text['abstract'])])
docs_length= pd.Series(value, index)                      


# In[10]:


doc_length = docs_length.plot(kind = 'hist', title = 'Histogram: Abstract Length', colormap='jet', figsize =(20,5))
doc_length.set_xlabel("Abstract Word Count")
doc_length.set_ylabel("Frequency")


# In[11]:


z_scores = np.abs(stats.zscore(docs_length))
filtered_entries = (z_scores <= 4)
outliers = (z_scores > 4)
outliers_text = text[outliers]
docs_length = docs_length[filtered_entries]


# In[12]:


doc_length = docs_length.plot(kind = 'hist', title = 'Histogram: Abstract Length, minus outliers', colormap='jet', figsize =(20,5))
doc_length.set_xlabel("Abstract Word Count")
doc_length.set_ylabel("Frequency")


# In[13]:


text = text[filtered_entries]
print(len(text))


# # Lemmatization/Stop word Reduction

# In[14]:


nlp = spacy.load('en_core_web_lg', disable=['tagger','parser','ner'])


# In[15]:


def tokenizer(sentence):
    return [word.lemma_ for word in nlp(sentence) if not (word.like_num or word.is_stop or len(word)==1)]


# # Vectorize

# In[16]:


from sklearn.feature_extraction.text import CountVectorizer


# In[17]:


vec = CountVectorizer(tokenizer = tokenizer, max_df=0.80, min_df=3)


# In[ ]:


cv = vec.fit_transform(text['abstract'])


# # Latent Dirichlet Allocation

# In[ ]:


from sklearn.decomposition import LatentDirichletAllocation


# In[ ]:


n_components = 20
lda = LatentDirichletAllocation(n_components=n_components,random_state=42)


# In[ ]:


lda = lda.fit(cv)
lda_doc_dist = lda.transform(cv)


# In[ ]:


def plot_doc(doc):
    index, value = zip(*[(index, val) for index, val in enumerate(lda_doc_dist[doc])])
    doc_topic_plot = pd.Series(value, index)
    return doc_topic_plot


# In[ ]:


doc = 42
topic_dist_chart = plot_doc(doc).plot(kind = 'bar', title = 'Distribution of Topics over Document', colormap='jet', figsize =(20,5))
topic_dist_chart.set_xlabel("Topic Number")
topic_dist_chart.set_ylabel("Topic Proportion")


# In[ ]:


text['topic_dist'] = lda_doc_dist.tolist()


# In[ ]:


text.to_csv('data/covid_papers.csv')


# # to similar texts for existing or non-existing texts in the data
# 

# In[ ]:


new_text = ['this is a sample abstract for the coronavirus text data.']


# In[ ]:


def similar_text(text_to_compare, n_articles = 10):
    store_vals = list()
    loc =0
    if(isinstance(text_to_compare, list) and len(text_to_compare) is 1):
        new_vec = vec.transform(text_to_compare)
        topic_dist = list(lda.transform(new_vec)[0])
    elif(isinstance(text_to_compare, int)):
        topic_dist = text.loc[text_to_compare, 'topic_dist']
    else:
        raise ValueError
    
    for i in range(len(text)):
        loc +=1
        
        if(i in text.index):
            store_vals.append((i, (sqrt(mean_squared_error(topic_dist, text.loc[i, 'topic_dist'])))))
    most_similar = sorted(store_vals, key=itemgetter(1))
    return [text.loc[i[0]] for i in most_similar[1:n_articles+1]]


# In[ ]:


similar_text(1)


# In[ ]:


text['topic'] = lda_doc_dist.argmax(axis=1)


# In[ ]:


def paper_by_topic(topic):
    print("---------There are " + str(len(text[text['topic'] == topic])) + " Articles in this topic cluster.---------")
    return text[text['topic'] == topic]['title']

for i in text['topic'].unique():
    paper_by_topic(i)


# In[ ]:


count_by_topic = text['topic'].value_counts()


# In[ ]:


topic_chart = count_by_topic.sort_index().plot(kind = 'bar', title = 'Number of Papers by Topic Cluster', colormap='jet', figsize =(20,5))
topic_chart.set_xlabel("Topic Number")
topic_chart.set_ylabel("Papers (Qty)")


# # Visualizations

# In[161]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vec = TfidfVectorizer(tokenizer = tokenizer, max_df=0.80, min_df=3)
X = tfidf_vec.fit_transform(text['abstract'])


# In[162]:


from sklearn.cluster import KMeans


# In[163]:


k_means = KMeans(n_clusters = 20)


# In[164]:


k_means.fit(X)


# In[168]:


kmeans = k_means.transform(X)


# In[174]:


y_kmeans = k_means.predict(X)


# In[187]:


x = X.todense()


# In[408]:


#tsne = TSNEVisualizer()


# In[410]:


#tsne.fit(cv, text['topic'])


# In[ ]:




