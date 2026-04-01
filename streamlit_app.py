import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# -----------------------------
# HuggingFace FinBERT API
# -----------------------------
API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"

headers = {
    "Authorization": "Bearer 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"
}

def get_sentiment(text):

    response = requests.post(
        API_URL,
        headers=headers,
        json={"inputs": text}
    )

    result = response.json()

    if isinstance(result, list):

        predictions = result[0]

        best = max(predictions, key=lambda x: x['score'])

        return best['label'], best['score']

    return "neutral", 0.5


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Financial News Sentiment And Stock Trends Analyzer",
    layout="wide"
)

# -----------------------------
# Background UI
# -----------------------------
st.markdown("""
<style>

.stApp {
background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
color:white;
}

h1,h2,h3,h4 {
color:white;
}

label {
color:white !important;
font-size:18px;
}

[data-testid="stMetricValue"] {
color:#00e6ff;
font-size:26px;
font-weight:bold;
}

[data-testid="stMetricLabel"] {
color:white;
font-size:16px;
}

.stButton>button {
background-color:#00c6ff;
color:white;
border-radius:10px;
height:3em;
width:100%;
font-size:18px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------
st.title("📊 Financial News Sentiment And Stock Trends Analyzer")

st.write(
"AI system that analyzes stock trend, sentiment and predicts future prices."
)

# -----------------------------
# User Input
# -----------------------------
stock = st.text_input(
"Enter Stock Name (Example: TCS, INFY, RELIANCE)"
)

# -----------------------------
# Button
# -----------------------------
if st.button("Analyze Stock"):

    if stock == "":

        st.warning("Please enter stock name")

    else:

        ticker = stock + ".NS"

        data = yf.download(
            ticker,
            period="6mo"
        )

        data = data.dropna()

        if data.empty:

            st.error("Stock not found")

        else:

            # -----------------------------
            # price metrics
            # -----------------------------
            last_day_price = float(
                data["Close"].iloc[-1]
            )

            previous_price = float(
                data["Close"].iloc[-2]
            )

            high_price = float(
                data["High"].max()
            )

            low_price = float(
                data["Low"].min()
            )

            change = last_day_price - previous_price

            if change > 0:
                trend = "Uptrend 📈"

            elif change < 0:
                trend = "Downtrend 📉"

            else:
                trend = "Stable ➖"

            # -----------------------------
            # prediction logic
            # -----------------------------
            recent_prices = data["Close"].tail(5)

            price_changes = recent_prices.diff().dropna()

            avg_change = float(
                price_changes.mean()
            )

            predicted_today = float(
                last_day_price + avg_change
            )

            predicted_next_day = float(
                predicted_today + avg_change
            )

            future_prices = [

                float(last_day_price),

                float(predicted_today),

                float(predicted_next_day)
            ]

            # -----------------------------
            # pseudo news
            # -----------------------------
            news_samples = [

                f"{stock} financial performance is strong",

                f"{stock} stock trend is {trend}",

                f"{stock} growth outlook based on recent data"
            ]

            sentiments = []

            scores = []

            for news in news_samples:

                label, score = get_sentiment(news)

                sentiments.append(label)

                scores.append(score)

            avg_sentiment = sum(scores)/len(scores)

            # -----------------------------
            # Metrics
            # -----------------------------
            st.subheader("📊 Stock Metrics")

            col1,col2,col3,col4,col5 = st.columns(5)

            col1.metric(
                "Last Close",
                round(last_day_price,2)
            )

            col2.metric(
                "Predicted Today",
                round(predicted_today,2)
            )

            col3.metric(
                "Predicted Next Day",
                round(predicted_next_day,2)
            )

            col4.metric(
                "6M High",
                round(high_price,2)
            )

            col5.metric(
                "6M Low",
                round(low_price,2)
            )

            st.write("Market Trend:", trend)

            # -----------------------------
            # historical graph
            # -----------------------------
            # -----------------------------
            # Compact Graph Layout
            # -----------------------------

            st.subheader("📊 Graph Analysis")

            col1,col2,col3 = st.columns(3)


            # Historical graph
            with col1:

                st.write("Historical")

                fig1 = plt.figure(figsize=(3,2))

                plt.plot(data["Close"])

                plt.xticks(rotation=45, fontsize=7)

                plt.yticks(fontsize=7)

                plt.title("Price", fontsize=9)

                plt.tight_layout()

                st.pyplot(fig1)


            # Prediction graph
            with col2:

                st.write("Prediction")

                fig2 = plt.figure(figsize=(3,2))

                plt.plot(["Last","Today","Next"],future_prices)

                plt.title("Future", fontsize=9)

                plt.xticks(fontsize=8)

                plt.yticks(fontsize=8)

                plt.tight_layout()

                st.pyplot(fig2)


            # Sentiment graph
            with col3:

                st.write("Sentiment")

                fig3 = plt.figure(figsize=(3,2))

                plt.bar(sentiments,scores)

                plt.xticks(fontsize=8)

                plt.yticks(fontsize=8)

                plt.tight_layout()

                st.pyplot(fig3)

            # -----------------------------
            # insight text
            # -----------------------------
            st.subheader("📄 Prediction Explanation")

            st.write(f"""

Stock Name: {stock}

Last Closing Price: {round(last_day_price,2)}

Predicted Price Today: {round(predicted_today,2)}

Predicted Price Next Day: {round(predicted_next_day,2)}

Highest Price in 6 months: {round(high_price,2)}

Lowest Price in 6 months: {round(low_price,2)}

Market Trend: {trend}

Average Sentiment Score: {round(avg_sentiment,2)}

""")

            # -----------------------------
            # recommendation
            # -----------------------------
            st.subheader("💡 AI Recommendation")

            if trend == "Uptrend 📈" and avg_sentiment > 0.5:

                st.success(
                    "BUY signal detected 📈"
                )

            elif trend == "Downtrend 📉":

                st.error(
                    "SELL signal detected 📉"
                )

            else:

                st.warning(
                    "HOLD recommendation ⏳"
                )
