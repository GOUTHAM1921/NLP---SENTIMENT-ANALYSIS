# %%
# Import Libraries
import re

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import string
import spacy

import wordcloud
from wordcloud import WordCloud, STOPWORDS

%matplotlib inline

import warnings
warnings.filterwarnings('ignore')


# %%
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# %%
conda install -c conda-forge wordcloud

# %%
"""
# Extracting Product reviews from Amazon
"""

# %%
from bs4 import BeautifulSoup as bs
import requests



pages=input("pls enter number of pages:")
review_title=[]



for i in range(1,int(pages)+1):
    url="https://www.amazon.in/OnePlus-Nord-Mirror-128GB-Storage/product-reviews/B09RG132Q5/ref=cm_cr_arp_d_paging_btm_next_2?pageNumber="+str(i)
    page=requests.get(url)
    soup=bs(page.content,'html.parser')
    review = soup.find_all('span',class_="review-text-content")
    
    for i in range(0,len(review)):
        review_title.append(review[i].get_text())
    
print("Total Reviews for the OnePlus CE2 Smart Phone:",len(review_title))

# %%
import pandas as pd

df=pd.DataFrame(review_title,columns=["Reviews"])
df.head()

# %%
df.shape

# %%
df.describe()

# %%
df.duplicated().sum()  #checking for duplicate reviews

# %%
"""
#### Removing duplicate Reviews
"""

# %%
df1=df.drop_duplicates().reset_index(drop=True) # droping duplicate reviews
df1.shape

# %%
"""
#### Checking for no of words in each review
"""

# %%
df1['Reviews_length'] = df1['Reviews'].str.split().apply(len)  #checking the length of the reviews 
df1.head()

# %%
"""
#### Checking for Avg Length of the reviews written by the consumers
"""

# %%
print(df1['Reviews_length'].mean())  #checking for avg length of reviews

df1['Reviews_length'].plot(bins=20, kind='hist',figsize = (6,4))

# %%
"""
#### Checking for no of stop words in each review
"""

# %%
stop = stopwords.words('english')

df1['Stopwords'] =df1['Reviews'].apply(lambda x: len([x for x in x.split() if x in stop]))
df1[['Reviews','Stopwords']].head()

# %%
"""
# Text pre-processing
"""

# %%
"""
#### Removing the all characters except Alphabets
"""

# %%
df1.Reviews=df.Reviews.apply(lambda x:re.sub('[^a-zA-Z ]',"",x))
df1.head()

# %%
"""
#### Converting the reviews into Lower case 
"""

# %%
df1.Reviews=df1.Reviews.apply(lambda x:x.lower())  #converting the reviews into lower case
df1.head()

# %%
"""
#### Removing STOP WORDS
"""

# %%
stop_words=stopwords.words('english')
sw_list=[
        "day","look","one","loaded","dont",
        "hai","call","day","samsung","im","writing","review","could","r","its","mp","also","if","ur","u",
        "im","n","dat","frm","itsif","dis","wit","w","c",'os','gb','k','etc','g','mm','vooc','oppo','hz','bit','would',"phone",
        "week","per","first","hr","come","say","user","aa","aaj","aap","aaya","aayi","ab","abhi"]
stop_words.extend(sw_list)

# %%
df1['Reviews'] = df1['Reviews'].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))

#Review Length after removing special characters and stop words
df1['Reviews_length_2'] = df1['Reviews'].str.split().apply(len) 
df1.head()

# %%
"""
#### Removing common words
"""

# %%
freq = pd.Series(' '.join(df1['Reviews']).split()).value_counts()[:10]
freq

# %%
freq = list(freq.index)
freq

# %%
df1['Reviews'] = df1['Reviews'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

#Review Length after removing some common words
df1['Reviews_length_3'] = df1['Reviews'].str.split().apply(len)
df1[['Reviews','Reviews_length','Reviews_length_2','Reviews_length_3']].head()

# %%
"""
#### Checking for most frequent words in the reviews using wordcloud
"""

# %%
sen_df=[Reviews.strip() for Reviews in df1.Reviews]
sentn=' '.join(sen_df)

# %%
import nltk
from nltk.tokenize import word_tokenize

rev_token=word_tokenize(sentn)

# %%
from wordcloud import WordCloud, STOPWORDS
def plot_cloud(wordcloud):
    plt.figure(figsize=(40,30))
    plt.imshow(wordcloud)
    plt.axis('on')
    

wordcloud=WordCloud(width=4000,height=3000,background_color='black',max_words=100,
                   colormap='RdYlGn',contour_color='red',contour_width=5,stopwords=STOPWORDS).generate(sentn)
plot_cloud(wordcloud)

# %%
"""
# Sentiment Analysis
"""

# %%
affin=pd.read_csv("Afinn.csv",encoding='cp1252')
affinity_scores=affin.set_index('word')['value'].to_dict()

# %%
!pip install -U pip setuptools wheel
!pip install -U spacy
!python -m spacy download en_core_web_sm

# %%
# Custom function: score each word in a sentence in lemmatised form, but calculate the score for the whole original sentence
nlp=spacy.load('en_core_web_sm')
sentiment_lexicon=affinity_scores

def calculate_sentiment(text):
    sent_score=0
    if text:
        sentence=nlp(text)
        for word in sentence:
            sent_score+=sentiment_lexicon.get(word.lemma_,0)
    return sent_score

# %%
df1['Sentiment_Value']=df1['Reviews'].apply(calculate_sentiment)
df1['Sentiment_Value'].head()

# %%
y=[]
for x in df1["Sentiment_Value"]:
    if x<0:
        y.append("Negative")
    elif x>0:
        y.append("Positive")
    else:
        y.append("Neutral")
        

# %%
df1["Category"]=pd.DataFrame(y)
df1["Category"].value_counts()

# %%
# negative sentiment score of the whole review
df1[df1['Sentiment_Value']<0].head()

# %%
# positive sentiment score of the whole review
df1[df1['Sentiment_Value']>0].head()

# %%
# Neutral sentiment score of the whole review
df1[df1['Sentiment_Value']==0].head()

# %%
# Plotting the sentiment value for whole review
import seaborn as sns
plt.figure(figsize=(10,6))
sns.distplot(df1['Sentiment_Value'])

# %%
df1.Category.value_counts()
sns.countplot(x='Category', data=df1, palette='hls')

# %%
y=[]
for x in df1["Sentiment_Value"]:
    if x<0:
        y.append(0)
    elif x>0:
        y.append(1)
    else:
        y.append(2)


# %%
#Adding Response attribute to data frame
df1["Response"]=pd.DataFrame(y)

df2=df1.loc[:,["Reviews","Sentiment_Value","Category","Response"]]
df2.head()

# %%
"""
## Stemming
"""

# %%
from nltk.stem import PorterStemmer
ps = PorterStemmer()

df2.Reviews=df2.Reviews.apply(lambda x:x.split())
df2.Reviews=df2.Reviews.apply(lambda x:[ps.stem(word) for word in x])
df2.Reviews=df2.Reviews.apply(lambda x:" ".join(x))

df2.head()

# %%
"""
## Feauture Extraction using Count Vectorizer
"""

# %%
X=df2["Reviews"]
Y=df2["Response"]

# %%
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

#cv=CountVectorizer()
vect = TfidfVectorizer()

x=vect.fit_transform(X.values).toarray()
x=pd.DataFrame(x,columns=vect.get_feature_names())
x.head()

# %%
"""
## Model Building
"""

# %%
"""
#### Logestic Regression
"""

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split  
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import classification_report

# %%
from sklearn.model_selection import train_test_split

#x_train, x_test, y_train, y_test = train_test_split(x, Y, test_size=0.2, random_state=12)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=12)
from sklearn.pipeline import Pipeline
pipe_lr=Pipeline(steps=[('cv',CountVectorizer()),('lr',LogisticRegression())])
pipe_lr.fit(x_train,y_train)
pred=pipe_lr.predict(x_test)
score=accuracy_score(y_test,pred)
score

# %%
x_test

# %%
classifier = LogisticRegression()
classifier.fit(x_train,y_train)

pred=classifier.predict(x_test)

# %%
from sklearn.metrics import accuracy_score
LOG=accuracy_score(y_test,pred)
LOG

# %%
pd.crosstab(y_test,pred)

# %%
"""
#### KNN
"""

# %%
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# %%
n_neighbors=np.array([2*i+1 for i in range(0,20)])
param_grid=dict(n_neighbors=n_neighbors)

KNN=KNeighborsClassifier()
grid=GridSearchCV(estimator=KNN,param_grid=param_grid,cv=10)
grid.fit(x,Y)
KNN_grid=grid.best_score_
KNN_param=grid.best_params_

print(KNN_grid,":",KNN_param)

# %%
k_range=[2*i+1 for i in range(0,20)]
k_scores=[]
for k in k_range:
    KNN=KNeighborsClassifier(n_neighbors=k)
    scores=cross_val_score(KNN,x,Y,cv=10)
    k_scores.append(scores.mean())
    
plt.bar(k_range,k_scores)
plt.plot(k_range,k_scores,color="red")
plt.xticks(k_range)
plt.show()

# %%
"""
#### Decision Tree
"""

# %%
from sklearn import datasets  
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing

# %%
x_train,x_test,y_train,y_test=train_test_split(x,Y,test_size=0.3,random_state=10)
model=DecisionTreeClassifier(criterion='gini',max_depth=15)
model.fit(x_train,y_train)

# %%
predict=model.predict(x_test)
pd.crosstab(y_test,predict)

# %%
DT=np.mean(y_test==predict)
DT

# %%
"""
#### Random Forest
"""

# %%
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import  RandomForestClassifier

from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing

# %%
k_range=[10,20,50,100,200,300,400,500]
k_scores=[]
max_features=5
for k in k_range:
    model=RandomForestClassifier(n_estimators=num_trees,max_samples=0.8,max_features=max_features,random_state=8)
    results=cross_val_score(model,x,Y,cv=10)
    k_scores.append(results.mean())
    
k_scores

# %%
kfold=KFold(n_splits=10)
num_trees=100
max_features=5
model=RandomForestClassifier(n_estimators=num_trees,max_samples=0.8,max_features=max_features,random_state=8)
results=cross_val_score(model,x,Y,cv=kfold)
RF=results.mean()
RF

# %%
from sklearn.model_selection import cross_val_predict
y_pred = cross_val_predict(model, x,Y, cv=kfold)
pd.crosstab(Y,y_pred)

# %%
np.mean(y_pred==Y)

# %%
"""
#### SVM
"""

# %%
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix

# %%
train_X,test_X,train_Y,test_Y=train_test_split(x,Y,test_size=0.3,random_state=10)

# %%
model_linear=SVC(kernel='linear')
model_linear.fit(train_X,train_Y)

train_pred_lin=model_linear.predict(train_X)
test_pred_lin=model_linear.predict(test_X)

train_lin_acc=np.mean(train_pred_lin==train_Y)
test_lin_acc=np.mean(test_pred_lin==test_Y)

SVM_LINEAR=test_lin_acc
SVM_LINEAR

# %%
model_rbf=SVC(C=15,gamma=0.0001,kernel='rbf')
model_rbf.fit(train_X,train_Y)

train_pred_rbf=model_linear.predict(train_X)
test_pred_rbf=model_linear.predict(test_X)

train_rbf_acc=np.mean(train_pred_lin==train_Y)
test_rbf_acc=np.mean(test_pred_lin==test_Y)


SVM_RBF=test_rbf_acc
SVM_RBF

# %%
"""
#### Bagging
"""

# %%
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier

# %%
kfold=KFold(n_splits=10)
num_trees=100
model=BaggingClassifier(max_samples=0.8,n_estimators=num_trees,random_state=8)
results=cross_val_score(model,x,Y,cv=kfold)
BAG=results.mean()
BAG

# %%
"""
#### AdaBoost
"""

# %%
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier

# %%
kfold=KFold(n_splits=10)
num_trees=100
model=AdaBoostClassifier(n_estimators=num_trees,learning_rate=0.8,random_state=8)
results=cross_val_score(model,x,Y,cv=kfold)
BOOST=results.mean()
BOOST

# %%
"""
#### Naive Bayes
"""

# %%
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB

# %%
train_X,test_X,train_Y,test_Y=train_test_split(x,Y,test_size=0.3,random_state=10)

# %%
Gmodel=GaussianNB()

Gmodel.fit(train_X,train_Y)
NB2=Gmodel.predict(test_X)
G_acc=np.mean(NB2==test_Y)
G_acc

# %%
Mmodel = MultinomialNB()
Mmodel.fit(train_X,train_Y)
MB = Mmodel.predict(test_X)
M_acc = np.mean(MB==test_Y) 
M_acc

# %%
my_dict={"Model":["Log_Reg","KNN","DecisionTree","RandomForest","SVM","ADABOOST","Naive Bayes_G",'Naive Bayes_M'],
         "Accuracy":[LOG,KNN_grid,DT,RF,SVM_LINEAR,BOOST,G_acc,M_acc]}
DF=pd.DataFrame(my_dict)
DF

# %%
"""
## DEPLOYMENT
"""

# %%
import joblib
import pickle
from pickle import dump
from pickle import load

# %%
dump(pipe_lr,open('Logistic_regression.pkl','wb'))

# %%


# %%


# %%


# %%
# import important modules
import numpy as np
import pandas as pd
# sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    plot_confusion_matrix,
)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# text preprocessing modules
from string import punctuation 
# text preprocessing modules
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
import re #regular expression
# Download dependency
for dependency in (
    "brown",
    "names",
    "wordnet",
    "averaged_perceptron_tagger",
    "universal_tagset",
):
    nltk.download(dependency)
    
import warnings
warnings.filterwarnings("ignore")

# %%
stop_words =  stopwords.words('english')
def text_cleaning(text, remove_stop_words=True, lemmatize_words=True):
    # Clean the text, with the option to remove stop_words and to lemmatize word
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text =  re.sub(r'http\S+',' link ', text)
    text = re.sub(r'\b\d+(?:\.\d+)?\s+', '', text) # remove numbers
        
    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])
    
    # Optionally, remove stop words
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)
    
    # Optionally, shorten words to their stems
    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer() 
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)
    
    # Return a list of words
    return(text)

# %%
#clean the review
df1["Reviews"] = df1["Reviews"].apply(text_cleaning)
df1["Reviews"].head()

# %%
Reviews=df1["Reviews"]
Target=df2["Response"]

# %%
# split data into train and validate
X_train, X_valid, y_train, y_valid = train_test_split(
    Reviews,
    Target,
    test_size=0.15,
    random_state=42,
    shuffle=True,
    stratify=Y,
)

# %%
# Create a classifier in pipeline
sentiment_classifier = Pipeline(steps=[
                               ('pre_processing',TfidfVectorizer(lowercase=False)),
                                 ('classifier',LogisticRegression())
                                 ])

# %%
sentiment_classifier.fit(X_train,y_train)

# %%
y_preds = sentiment_classifier.predict(X_valid)
accuracy_score(y_valid,y_preds)

# %%
#save model 
import joblib 
joblib.dump(sentiment_classifier, 'sentiment_model_pipeline.pkl')

# %%
"""
# Deployment
"""

# %%
!pip install fastapi
!pip install uvicorn

# %%
# text preprocessing modules
from string import punctuation
# text preprocessing modules
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re  # regular expression
import os
from os.path import dirname, join, realpath
import joblib
import uvicorn
from fastapi import FastAPI 

app = FastAPI(
    title="Sentiment Model API",
    description="A simple API that use NLP model to predict the sentiment of the movie's reviews",
    version="0.1",
)

# load the sentiment model
with open(
    "sentiment_model_pipeline.pkl", "rb"
) as f:
    model = joblib.load(f)


# cleaning the data
def text_cleaning(text, remove_stop_words=True, lemmatize_words=True):
    # Clean the text, with the option to remove stop_words and to lemmatize word
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"http\S+", " link ", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\s+", "", text)  # remove numbers
    
    # Remove punctuation from text
    text = "".join([c for c in text if c not in punctuation])
    
    # Optionally, remove stop words
    if remove_stop_words:
        # load stopwords
        stop_words = stopwords.words("english")
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)
        
    # Optionally, shorten words to their stems
    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)
        
    # Return a list of words
    return text

@app.get("/predict-review")
def predict_sentiment(review: str):
    """
    A simple function that receive a review content and predict the sentiment of the content.
    :param review:
    :return: prediction, probabilities
    """
    # clean the review
    cleaned_review = text_cleaning(review)
    
    # perform prediction
    prediction = model.predict([cleaned_review])
    output = int(prediction[0])
    probas = model.predict_proba([cleaned_review])
    output_probability = "{:.2f}".format(float(probas[:, output]))
    
    # output dictionary
    sentiments = {0: "Negative", 1: "Positive",2: "Neutral"}
    
    # show results
    result = {"prediction": sentiments[output], "Probability": output_probability}
    return result

# %%
