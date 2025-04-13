import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import gdown

# Define the path and Google Drive file ID
file_path = "WELFake_Dataset.csv"
file_id = "105Pg07PqFa0540QHi3aN_QkvAjk-3P3i"
url = f"https://drive.google.com/uc?id={file_id}"

# Download only if the file doesn't already exist
if not os.path.exists(file_path):
    st.info("Downloading dataset from Google Drive...")
    gdown.download(url, file_path, quiet=False)

# Read the CSV
st.info("Loading dataset...")
news_df = pd.read_csv(file_path)

# Display columns for debug
st.subheader("🧾 Column Names:")
st.write(news_df.columns)

st.subheader("🔍 Sample Data:")
st.write(news_df.head())

# Load Data
news_df = news_df.fillna(' ')
news_df['content'] = news_df['title'] + " " + news_df['text']
X = news_df.drop('label', axis=1)
y = news_df['label']

# Define Stemming Function...
ps = PorterStemmer()
def stemming(content):
  stemmend_content = re.sub('[^a-zA-Z]',' ',content)
  stemmend_content = stemmend_content.lower()
  stemmend_content = stemmend_content.split()
  stemmend_content = [ps.stem(word) for word in stemmend_content]
  stemmend_content = ' '.join(stemmend_content)
  return stemmend_content

#Apply Stemming Function To Content Column...
news_df['content'] = news_df['content'].apply(stemming)

# Vectorize Data...
X = news_df['content'].values
y = news_df['label'].values
vector = TfidfVectorizer()
vector.fit(X)
X = vector.transform(X)

# Split Data Into Train And Test sets....
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Fit Logistic regression Model...
model = LogisticRegression()
model.fit(X_train,y_train)

# Website...
st.title('Fake News detector')
input_text = st.text_input('Enter News Article:-')

def prediction(input_text):
    input_data = vector.transform([input_data])
    prediction = model.predict(input_data)
    return prediction[0]

if input_text:
    pred = prediction(input_text)  
    if pred == 1:
        st.write('The News Is Fake')
    else:
        st.write('The News Is Real')  
