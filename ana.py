import streamlit as st
from textblob import TextBlob
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import cleantext
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load the dataset
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

# Classify sentiment based on the VADER scores
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
        
# Streamlit app
st.title("🌟 Sentiment Analysis On Customer Feedback 🌟")

# User input
st.subheader("Enter Your Feedback")
user_input = st.text_area("Type your feedback here:")

if user_input:
    blob = TextBlob(user_input)
    user_sentiment_score = analyzer.polarity_scores(user_input)['compound']
    if user_sentiment_score > 0.8:
        user_sentiment_class = "Highly positive"
    elif user_sentiment_score > 0.4:
        user_sentiment_class = "Positive"
    elif -0.4 <= user_sentiment_score <= 0.4:
        user_sentiment_class = "Neutral"
    elif user_sentiment_score < -0.4:
        user_sentiment_class = "Negative"
    else:
        user_sentiment_class = "Highly negative"
        
    st.markdown(f"**VADER Sentiment Class:** {user_sentiment_class}")
    st.markdown(f"**VADER Sentiment Score:** {user_sentiment_score}")
    st.markdown(f"**TextBlob Polarity:** {blob.sentiment.polarity}")
    st.markdown(f"**TextBlob Subjectivity:** {blob.sentiment.subjectivity}")

    st.subheader("Clean Text")
    clean_text = cleantext.clean(user_input, clean_all=False, extra_spaces=True, stopwords=True, lowercase=True, numbers=True, punct=True)
    st.write(clean_text)
else:
    st.write("Please enter feedback to analyze and clean.")

# Display graphical representation
st.subheader("Graphical Representation of Data")
fig, ax = plt.subplots(2, 1, figsize=(10, 12))

# Histogram
sentiment_scores_by_class = {k: [] for k in set(sentiment_classes)}
for sentiment_score, sentiment_class in zip(sentiment_scores, sentiment_classes):
    sentiment_scores_by_class[sentiment_class].append(sentiment_score)
    
for sentiment_class, scores in sentiment_scores_by_class.items():
    ax[0].hist(scores, label=sentiment_class, alpha=0.5)

ax[0].set_xlabel("Sentiment score")
ax[0].set_ylabel("Count")
ax[0].set_title("Score Distribution by Class")
ax[0].legend()

# Pie chart
sentiment_counts = pd.Series(sentiment_classes).value_counts()
ax[1].pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
ax[1].set_title("Sentiment Distribution")

st.pyplot(fig)

# Word cloud
st.subheader("Word Cloud of Frequent Terms")
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(review_text))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
st.pyplot(plt)

# DataFrames with Sentiment Analysis results
df["Sentiment Class"] = sentiment_classes
df["Sentiment Score"] = sentiment_scores
df["Subjectivity"] = blob_subj

new_df = df[["Score", "Text", "Sentiment Score", "Sentiment Class", "Subjectivity"]]
st.subheader("Input Dataframe")
st.dataframe(new_df.head(30), use_container_width= True)
