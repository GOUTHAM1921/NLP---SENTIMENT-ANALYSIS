# %%
import re
import pandas as pd
from sklearn.linear_model import LogisticRegression

import pickle
from pickle import dump
from pickle import load

import streamlit as st

import joblib

# %%
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# %%
loaded_model=load(open('Logistic_regression.pkl','rb'))
loaded_model

# %%
ps=PorterStemmer()
def preprocess(X):
    X=re.sub('[^a-zA-Z ]','',X)
    X=X.lower()
    X=X.split()
    X=[word for word in X if word not in set(stopwords.words('english'))]
    X=[ps.stem(word) for word in X]
    X=" ".join(X)
    return X

# %%

def SENTIMENT_PREDICTION(input):
    df=pd.DataFrame({"Review":input},index=[0])
    df["Review"]=df["Review"].apply(preprocess)
    X=df["Review"]
    
    
    prediction=loaded_model.predict(X)
    if(prediction[0]==0):
        return "NEGATIVE REVIEW"
    elif(prediction[0]==1):
        return "POSITIVE REVIEW"
    else:
        return "NEUTRAL REVIEW"

# %%
def main():
    st.title("Model Deployment:Logistic Regression")
    
    t1=st.text_input("Enter the Review")
    
    diagnosis=''
    if st.button("Emotion in the Review"):
        diagnosis=SENTIMENT_PREDICTION(t1)
    st.success(diagnosis)

if __name__ == '__main__':
    main()

# %%
#t1="This is the worst Mobile"
#SENTIMENT_PREDICTION(t1)
    