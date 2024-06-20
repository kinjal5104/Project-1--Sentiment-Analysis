import streamlit as st
from textblob import TextBlob
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import cleantext
import matplotlib.pyplot as plt

# lets just get the dataset
df = pd.read_csv("Reviews.csv")

review_text = df["Text"]

# Initialize the VADER Sentiment Intensity Analyzer
analyzer = SentimentIntensityAnalyzer()

# Analyze sentiment and the subjectivity
sentiment_scores = []
blob_subj = []
for review in review_text:
    sentiment_scores.append(analyzer.polarity_scores(review)["compound"])
    blob = TextBlob(review)
    blob_subj.append(blob.subjectivity)

# classify sentiment based on the VADER scores
sentiment_classes = []
for sentiment_score in sentiment_scores:
    if sentiment_score > 0.8:
        sentiment_classes.append("Highly positive")    
    elif sentiment_score > 0.4:
        sentiment_classes.append("Positive")
    elif -0.4 <= sentiment_score <= 0.4:
        sentiment_classes.append("Neutral")
    elif sentiment_score < -0.4:
        sentiment_classes.append("Negative")
    else:
        sentiment_classes.append("Highly negative")
        
# streamlit app
st.title("Sentiment Analysis On Customer Feedback")

#user input
user_input = st.text_area("Enter the Feedback:")
blob = TextBlob(user_input)

user_sentiment_score = analyzer.polarity_scores(user_input)['compound']
if user_sentiment_score > 0.8:
    user_sentiment_class = "Highly positive"
elif user_sentiment_score >0.4:
    user_sentiment_class = "Positive"
elif -0.4 <= user_sentiment_score <= 0.4:
    user_sentiment_class = "Neutral"
elif user_sentiment_score < -0.4:
    user_sentiment_class = "Negative"
else:
    user_sentiment_class = "Highly negative"
    
st.write("** VADER Sentiment Class: ", user_sentiment_class, "**VADER Sentiment Scores: **", user_sentiment_score)
st.write("** TextBlob Polarity**", blob.sentiment.polarity, "** TextBlob Subjectiity:**", blob.sentiment.subjectivity)

# Display clean text
pre = st.text_input("Clean Text: ")
if pre:
    st.write(cleantext.clean(pre, clean_all= False, extra_spaces= True, stopwords= True, lowercase= True, numbers = True, punct= True))
else:
    st.write("No text is been provided from the user for cleaning.")
    
# Graphical Representation of Data
st.subheader("Graphical Representation of Data")
plt.figure(figsize=(10, 6))

sentiment_scores_by_class = {k: [] for k in set(sentiment_classes)}
for sentiment_score, sentiment_class in zip(sentiment_scores, sentiment_classes):
    sentiment_scores_by_class[sentiment_class].append(sentiment_score)
    
for sentiment_class, scores in sentiment_scores_by_class.items():
    plt.hist(scores, label= sentiment_class, alpha=0.5)
    
plt.xlabel("Sentiment score")
plt.ylabel("Count")
plt.title("Score Distribution by class")
plt.legend()
st.pyplot(plt)

# DataFrames with Sentiment Analysis results
df["Sentiment Class"] = sentiment_classes
df["Sentiment Score"] = sentiment_scores
df["Subjectivity"] = blob_subj

new_df = df[["Score", "Text", "Sentiment Score", "Sentiment Class", "Subjectivity"]]
st.subheader("Input Dataframe")
st.dataframe(new_df.head(30), use_container_width= True)