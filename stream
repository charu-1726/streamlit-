import streamlit as st
from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Stock Sentiment Analyzer", layout="centered")

st.title("📊 AI Financial News Sentiment Analyzer")
st.write("Analyze stock sentiment from financial news headlines")

# -----------------------------
# Default News
# -----------------------------
default_news = [
    "Company profits increased this quarter",
    "Stock prices are falling due to market crash",
    "New product launch receives positive response",
    "Company faces legal issues",
    "Strong growth expected in upcoming months"
]

# -----------------------------
# User Input Section
# -----------------------------
stock = st.text_input("Enter Stock / Company Name")

st.subheader("📰 Enter Financial News Headlines")

news_list = []

for i in range(3):
    news = st.text_input(f"Headline {i+1}")
    if news:
        news_list.append(news)

# Add default sample option
if st.checkbox("Use sample financial news"):
    news_list = default_news

# -----------------------------
# Sentiment Function
# -----------------------------
def analyze_sentiment(news_list):
    results = []
    scores = []

    for news in news_list:
        polarity = TextBlob(news).sentiment.polarity
        scores.append(polarity)

        if polarity > 0:
            sentiment = "Positive 😊"
        elif polarity < 0:
            sentiment = "Negative 😟"
        else:
            sentiment = "Neutral 😐"

        results.append([news, sentiment, round(polarity,2)])

    avg_score = sum(scores)/len(scores)
    return results, scores, avg_score

# -----------------------------
# Analyze Button
# -----------------------------
if st.button("🔍 Analyze Sentiment"):

    if stock == "":
        st.warning("⚠ Please enter company name")

    elif len(news_list) == 0:
        st.warning("⚠ Please enter at least one news headline")

    else:
        st.subheader(f"📌 Analysis for {stock}")

        results, scores, avg_score = analyze_sentiment(news_list)

        df = pd.DataFrame(results, columns=["News", "Sentiment", "Score"])
        st.dataframe(df)

        # -----------------------------
        # Chart
        # -----------------------------
        st.subheader("📈 Sentiment Score Distribution")

        plt.figure()
        plt.hist(scores)
        plt.xlabel("Sentiment Score")
        plt.ylabel("Frequency")

        st.pyplot(plt)

        # -----------------------------
        # Decision Logic
        # -----------------------------
        st.subheader("💡 Trading Decision")

        if avg_score > 0.1:
            st.success("📈 BUY Recommendation")
        elif avg_score < -0.1:
            st.error("📉 SELL Recommendation")
        else:
            st.warning("⏳ HOLD Recommendation")

        st.metric("Average Sentiment Score", round(avg_score,2))

        # Progress Bar Indicator
        st.progress(min(max((avg_score+1)/2,0),1))
