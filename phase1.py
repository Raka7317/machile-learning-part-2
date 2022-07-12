#!/usr/bin/env python
# coding: utf-8

# In[167]:


import numpy as np
import pandas as pd


# In[168]:


movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')


# In[169]:


movies.head(1)


# In[170]:


credits.head(1)


# In[171]:


movies=movies.merge(credits,on='title')


# In[172]:


movies.head(2)


# In[173]:


movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[174]:


movies.head()


# In[175]:


movies.isnull().sum()


# In[176]:


movies.dropna(inplace=True)


# In[177]:


movies.isnull().sum()


# In[178]:


movies.duplicated().sum()


# In[179]:


import ast


# In[180]:


def convert(text):
    l=[]
    for i in ast.literal_eval(text):
        l.append(i['name'])
    return l


# In[181]:


movies['genres'] = movies['genres'].apply(convert)
movies.head()


# In[182]:


movies['keywords'] = movies['keywords'].apply(convert)
movies.head()


# In[183]:


def convert2(text):
    l=[]
    count=0
    for i  in ast.literal_eval(text):
        if count!=3:
            l.append(i['name'])
            count+=1
        else:
            break
            
    return l


# In[184]:


movies['cast']=movies['cast'].apply(convert2)


# In[185]:


movies.head()


# In[186]:


def fetchdirector(obj):
    l=[]
    for  i in ast.literal_eval(obj):
        if i['job']=='Director':
            l.append(i['name'])
            break
    return l


# In[187]:


movies['crew']=movies['crew'].apply(fetchdirector)


# In[188]:


movies.head()


# In[189]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[190]:


movies.head()


# In[191]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keyowrds']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[192]:


movies.head()


# In[193]:


movies['tag']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']


# In[194]:


movies.head()


# In[195]:


new_df=movies[['movie_id','title','tag']]


# In[196]:


new_df.head()


# In[197]:



new_df['tag']=new_df['tag'].apply(lambda x:" ".join(x))


# In[198]:


new_df.head(3)


# In[199]:


new_df['tag']=new_df['tag'].apply(lambda x:x.lower())


# In[200]:


new_df.head(3)


# In[201]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')
    


# In[202]:


vectors=cv.fit_transform(new_df['tag']).toarray()


# In[203]:


vectors


# In[204]:


get_ipython().system('pip install nltk')


# In[205]:


import nltk


# In[206]:


from nltk.stem.porter import PorterStemmer


# In[207]:


ps=PorterStemmer()


# In[208]:


def stem(text):
    y=[]
    for i  in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[209]:


ps.stem('danced')


# In[210]:


new_df['tag']=new_df['tag'].apply(stem)


# In[211]:


from sklearn.metrics.pairwise import cosine_similarity


# In[212]:


similarity=cosine_similarity(vectors)


# In[213]:


def recommend(movie):
       movie_index=new_df[new_df['title']==movie].index[0]
       distances=similarity[movie_index]
       movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
       
       for i in movies_list:
           print(new_df.iloc[i[0]].title)


# In[214]:


recommend('Avatar')


# In[215]:


import pickle


# In[216]:


pickle.dump(new_df,open('movies.pkl','wb'))


# In[217]:


pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb))


# In[ ]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:





# In[ ]:





# In[ ]:




