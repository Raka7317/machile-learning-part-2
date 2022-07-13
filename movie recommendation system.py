

import numpy as np
import pandas as pd

movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')

movies.head(1)

credits.head(1)

movies=movies.merge(credits,on='title')
movies.head(2)
movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]

movies.head()
movies.isnull().sum()
movies.dropna(inplace=True)
movies.isnull().sum()
movies.duplicated().sum()
import ast
def convert(text):
    l=[]
    for i in ast.literal_eval(text):
        l.append(i['name'])
    return l





movies['genres'] = movies['genres'].apply(convert)
movies.head()




movies['keywords'] = movies['keywords'].apply(convert)
movies.head()





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




movies['cast']=movies['cast'].apply(convert2)




movies.head()




def fetchdirector(obj):
    l=[]
    for  i in ast.literal_eval(obj):
        if i['job']=='Director':
            l.append(i['name'])
            break
    return l





movies['crew']=movies['crew'].apply(fetchdirector)





movies.head()





movies['overview']=movies['overview'].apply(lambda x:x.split())




movies.head()





movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keyowrds']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])





movies.head()





movies['tag']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']




movies.head()





new_df=movies[['movie_id','title','tag']]




new_df.head()






new_df['tag']=new_df['tag'].apply(lambda x:" ".join(x))




new_df.head(3)




new_df['tag']=new_df['tag'].apply(lambda x:x.lower())





new_df.head(3)





from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')
    



vectors=cv.fit_transform(new_df['tag']).toarray()





vectors





get_ipython().system('pip install nltk')




import nltk




from nltk.stem.porter import PorterStemmer





ps=PorterStemmer()





def stem(text):
    y=[]
    for i  in text.split():
        y.append(ps.stem(i))
    return " ".join(y)





ps.stem('danced')





new_df['tag']=new_df['tag'].apply(stem)





from sklearn.metrics.pairwise import cosine_similarity





similarity=cosine_similarity(vectors)




def recommend(movie):
       movie_index=new_df[new_df['title']==movie].index[0]
       distances=similarity[movie_index]
       movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
       
       for i in movies_list:
           print(new_df.iloc[i[0]].title)




recommend('Avatar')




import pickle




pickle.dump(new_df,open('movies.pkl','wb'))





pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb))




pickle.dump(similarity,open('similarity.pkl','wb'))














