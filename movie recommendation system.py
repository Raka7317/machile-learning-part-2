#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


movies.head(1)


# In[4]:


credits.head(1)


# In[5]:


movies=movies.merge(credits,on='title')


# In[6]:


movies.head(2)


# In[7]:


movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[8]:


movies.head()


# In[9]:


movies.isnull().sum()


# In[10]:


movies.dropna(inplace=True)


# In[11]:


movies.isnull().sum()


# In[12]:


movies.duplicated().sum()


# In[13]:


import ast


# In[14]:


def convert(text):
    l=[]
    for i in ast.literal_eval(text):
        l.append(i['name'])
    return l


# In[15]:


movies['genres'] = movies['genres'].apply(convert)
movies.head()


# In[16]:


movies['keywords'] = movies['keywords'].apply(convert)
movies.head()


# In[17]:


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


# In[18]:


movies['cast']=movies['cast'].apply(convert2)


# In[19]:


movies.head()


# In[20]:


def fetchdirector(obj):
    l=[]
    for  i in ast.literal_eval(obj):
        if i['job']=='Director':
            l.append(i['name'])
            break
    return l


# In[21]:


movies['crew']=movies['crew'].apply(fetchdirector)


# In[22]:


movies.head()


# In[23]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[24]:


movies.head()


# In[25]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keyowrds']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[26]:


movies.head()


# In[27]:


movies['tag']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']


# In[28]:


movies.head()


# In[29]:


new_df=movies[['movie_id','title','tag']]


# In[30]:


new_df.head()


# In[31]:



new_df['tag']=new_df['tag'].apply(lambda x:" ".join(x))


# In[32]:


new_df.head(3)


# In[33]:


new_df['tag']=new_df['tag'].apply(lambda x:x.lower())


# In[34]:


new_df.head(3)


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')
    


# In[ ]:


vectors=cv.fit_transform(new_df['tag']).toarray()


# In[ ]:


vectors


# In[ ]:


get_ipython().system('pip install nltk')


# In[ ]:


import nltk


# In[ ]:


from nltk.stem.porter import PorterStemmer


# In[ ]:


ps=PorterStemmer()


# In[ ]:


def stem(text):
    y=[]
    for i  in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[ ]:


ps.stem('danced')


# In[ ]:


new_df['tag']=new_df['tag'].apply(stem)


# In[ ]:


from sklearn.metrics.pairwise import cosine_similarity


# In[ ]:


similarity=cosine_similarity(vectors)


# In[ ]:


def recommend(movie):
       movie_index=new_df[new_df['title']==movie].index[0]
       distances=similarity[movie_index]
       movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
       
       for i in movies_list:
           print(new_df.iloc[i[0]].title)


# In[ ]:


recommend('Avatar')


# In[ ]:


import pickle


# In[ ]:


pickle.dump(new_df,open('movies.pkl','wb'))


# In[ ]:


pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb))


# In[ ]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:





# In[ ]:





# In[ ]:




