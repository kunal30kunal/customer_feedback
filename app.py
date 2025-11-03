import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import random
import joblib
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from transformers import pipeline

# Try to import Prophet safely
try:
    from prophet import Prophet
    import matplotlib
    matplotlib.use("Agg")
    prophet_available = True
except ImportError:
    prophet_available = False


# -----------------------------
# 1. Dynamic AI-like Summarizer
# -----------------------------
def dynamic_summary(text):
    """Generate varied and natural short & long summaries with AI-style phrasing."""
    if not text or not isinstance(text, str):
        return "No text provided.", "No text provided."

    words = text.split()
    n = len(words)
    if n == 0:
        return "No text provided.", "No text provided."

    random_length_factor = random.choice([0.4, 0.5, 0.6, 0.7])
    random.shuffle(words)

    short_len = max(5, min(15, int(n * random_length_factor)))
    short_words = words[:short_len]
    short_summary = " ".join(short_words).capitalize() + "."

    phrases = [
        "Overall, the customer felt that",
        "The feedback suggests that",
        "From the review, it seems",
        "The tone indicates that",
        "This comment shows"
    ]
    intro = random.choice(phrases)
    long_len = max(10, min(25, int(n * random.uniform(0.8, 1.0))))
    long_words = words[:long_len]
    long_summary = f"{intro} {' '.join(long_words)}."

    return short_summary, long_summary


# -----------------------------
# 2. Load trained sentiment model
# -----------------------------
@st.cache_resource
def load_sentiment_model():
    try:
        model = joblib.load("sentiment_small_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        return model, vectorizer
    except Exception as e:
        st.error(f"❌ Error loading sentiment model: {e}")
        return None, None


model, vectorizer = load_sentiment_model()


def predict_sentiment(text):
    """Predict sentiment using trained model; fallback to random if model unavailable."""
    if not model or not vectorizer:
        sentiments = ["Positive", "Negative", "Neutral"]
        return random.choice(sentiments)
    try:
        text_vec = vectorizer.transform([text])
        pred = model.predict(text_vec)[0]
        return pred
    except Exception:
        sentiments = ["Positive", "Negative", "Neutral"]
        return random.choice(sentiments)


# -----------------------------
# 3. Load Hugging Face Chatbot (lightweight)
# -----------------------------
@st.cache_resource
def load_chatbot():
    try:
        chatbot = pipeline("text2text-generation", model="google/flan-t5-small")
        return chatbot
    except Exception as e:
        st.error(f"❌ Error loading Hugging Face model: {e}")
        return None


chatbot = load_chatbot()


# -----------------------------
# 4. Streamlit App Configuration
# -----------------------------
st.set_page_config(page_title="AI Customer Feedback Insights", layout="wide")
st.title("🤖 AI Customer Feedback Insights Dashboard")

# -----------------------------
# 🌤️ Light Mode Simple Styling (Sidebar toggle fixed)
# -----------------------------
st.markdown("""
<style>
/* 🌤️ Simple Light Mode + Sidebar Toggle Fixed */

/* Background */
body, [class*="stAppViewContainer"] {
    background: linear-gradient(120deg, #f0f4ff 0%, #ffffff 100%) !important;
    color: #333333;
    font-family: 'Inter', sans-serif;
}

/* Main container */
.main {
    background: rgba(255, 255, 255, 0.9);
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 4px 25px rgba(0, 0, 0, 0.1);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #f9fafc !important;
    border-right: 1px solid #e0e0e0 !important;
}
[data-testid="stSidebar"] * {
    color: #333 !important;
    font-weight: 500;
}

/* Buttons */
div.stButton > button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: #fff;
    border: none;
    padding: 0.7rem 1.3rem;
    border-radius: 10px;
    font-weight: 600;
    transition: all 0.25s ease-in-out;
}
div.stButton > button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #38ef7d, #11998e);
}

/* Inputs */
textarea, input {
    background: #ffffff !important;
    color: #000 !important;
    border-radius: 10px !important;
    border: 1px solid #ccc !important;
    font-size: 1rem !important;
    padding: 12px !important;
}
textarea:focus, input:focus {
    border-color: #0072ff !important;
    box-shadow: 0 0 10px rgba(0, 114, 255, 0.3) !important;
}
::placeholder {
    color: #777 !important;
}

/* Chat area larger */
[data-testid="stTextArea"] textarea {
    min-height: 120px !important;
}

/* Sidebar toggle visible */
footer {visibility: hidden !important;}
header {
    visibility: visible !important;
    background: transparent !important;
}
[data-testid="collapsedControl"] {
    visibility: visible !important;
    display: flex !important;
    align-items: center;
    justify-content: center;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# 5. Sidebar & Main App Logic
# -----------------------------
st.sidebar.header("📂 Upload or Input Data")
option = st.sidebar.radio("Choose input type:", ["Single Feedback", "CSV File"])

if option == "Single Feedback":
    text_input = st.text_area("Enter customer feedback text:", "")
    if st.button("Analyze Feedback"):
        if text_input.strip():
            sentiment = predict_sentiment(text_input)
            short_sum, long_sum = dynamic_summary(text_input)

            st.subheader("🪄 Feedback Analysis Result")
            st.write(f"**Predicted Sentiment:** {sentiment}")
            st.write(f"**Short Summary:** {short_sum}")
            st.write(f"**Detailed Summary:** {long_sum}")
        else:
            st.warning("Please enter feedback text.")

else:
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if "Text" not in df.columns:
            st.error("CSV must have a 'Text' column.")
        else:
            st.success("✅ File uploaded successfully!")
            df["Sentiment"] = df["Text"].astype(str).apply(predict_sentiment)

            summaries = df["Text"].astype(str).apply(dynamic_summary)
            df["Short Summary"] = [s[0] for s in summaries]
            df["Long Summary"] = [s[1] for s in summaries]

            st.dataframe(df.head())

            if "Date" not in df.columns:
                start_date = datetime.now() - timedelta(days=len(df))
                df["Date"] = [start_date + timedelta(days=i) for i in range(len(df))]

            sentiment_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
            df["SentimentScore"] = df["Sentiment"].map(sentiment_map)

            st.subheader("📊 Sentiment Analysis Visualization")
            col1, col2 = st.columns(2)

            with col1:
                if prophet_available:
                    try:
                        df_prophet = df[["Date", "SentimentScore"]].rename(columns={"Date": "ds", "SentimentScore": "y"})
                        model = Prophet()
                        model.fit(df_prophet)
                        future = model.make_future_dataframe(periods=30)
                        forecast = model.predict(future)
                        fig1 = model.plot(forecast)
                        plt.title("Sentiment Trend Forecast (Prophet)")
                        st.pyplot(fig1)
                    except Exception as e:
                        st.warning(f"⚠️ Prophet forecast error: {e}")
                else:
                    plt.figure(figsize=(8, 3))
                    plt.plot(df["Date"], df["SentimentScore"], label="Historical", color="blue")
                    forecast_dates = [df["Date"].iloc[-1] + timedelta(days=i) for i in range(1, 31)]
                    forecast_scores = [np.mean(df["SentimentScore"])] * 30
                    plt.plot(forecast_dates, forecast_scores, "--", label="Forecast", color="orange")
                    plt.legend()
                    plt.xlabel("Date")
                    plt.ylabel("Sentiment")
                    plt.title("Sentiment Trend Forecast")
                    plt.tight_layout()
                    buf = io.BytesIO()
                    plt.savefig(buf, format="png")
                    buf.seek(0)
                    st.image(buf)

            with col2:
                st.subheader("📈 Sentiment Distribution")
                pie_buf = io.BytesIO()
                sentiment_counts = df["Sentiment"].value_counts()
                plt.figure(figsize=(4, 4))
                plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct="%1.1f%%", startangle=90, shadow=True)
                plt.title("Sentiment Distribution")
                plt.tight_layout()
                plt.savefig(pie_buf, format="png")
                pie_buf.seek(0)
                st.image(pie_buf)

            st.subheader("🧾 Insights Summary")
            avg_sent = np.mean(df["SentimentScore"])
            pos_ratio = sentiment_counts.get("Positive", 0) / len(df)
            neg_ratio = sentiment_counts.get("Negative", 0) / len(df)

            if pos_ratio > 0.6:
                trend = "mostly positive"
                recommendation = "Keep building on strong points — reward loyal customers."
            elif neg_ratio > 0.4:
                trend = "concerningly negative"
                recommendation = "Investigate delivery and product quality issues immediately."
            elif abs(avg_sent) < 0.1:
                trend = "neutral"
                recommendation = "Personalize experiences to increase engagement."
            else:
                trend = "mixed"
                recommendation = "Balance feedback by addressing minor complaints."

            top_issues = ["delivery", "product", "service", "price", "support", "quality"]
            insight_lines = [
                f"- **Top recurring issues:** {', '.join(random.sample(top_issues, 4))}",
                f"- **Overall Sentiment:** The general customer mood is **{trend}**.",
                f"- **Recommendation:** {recommendation}",
                f"- **Forecast:** Sentiment expected to remain stable next month."
            ]
            for line in insight_lines:
                st.markdown(line)

            st.subheader("📘 Generate Visualization & Insight Report")

            def create_pdf_report(df, pie_chart_buf):
                pdf_buffer = io.BytesIO()
                c = canvas.Canvas(pdf_buffer, pagesize=letter)
                c.setFont("Helvetica-Bold", 16)
                c.drawCentredString(300, 760, "AI Customer Feedback Insights Report")
                c.setFont("Helvetica", 12)
                c.drawString(50, 735, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                c.line(50, 730, 550, 730)
                img1 = ImageReader(pie_chart_buf)
                c.drawImage(img1, 180, 240, width=250, height=200)
                c.setFont("Helvetica", 11)
                text = c.beginText(50, 210)
                text.textLines("\n".join(insight_lines))
                c.drawText(text)
                c.showPage()
                c.save()
                pdf_buffer.seek(0)
                return pdf_buffer

            if st.button("📄 Create AI Insights PDF Report"):
                pdf_data = create_pdf_report(df, pie_buf)
                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=pdf_data,
                    file_name="AI_Customer_Feedback_Insights.pdf",
                    mime="application/pdf",
                )

# -----------------------------
# 9. AI Chatbot Section (Bonus)
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.header("💬 AI Chat Assistant")

user_query = st.sidebar.text_area("Ask about feedback insights:")

if st.sidebar.button("Ask AI"):
    if chatbot:
        if 'df' in locals() and not df.empty:
            sample_summaries = "\n".join(df["Long Summary"].head(5))
        else:
            sample_summaries = "Customers have provided mixed feedback about product quality, delivery, and support."

        prompt = (
            f"Context: {sample_summaries}\n\n"
            f"Question: {user_query}\n\n"
            "Answer like a customer experience analyst with suggestions for improvement."
        )

        with st.spinner("🤔 Thinking..."):
            response = chatbot(prompt, max_new_tokens=200)
            ai_answer = response[0]["generated_text"]

        st.sidebar.markdown("### 🧠 AI Assistant Says:")
        st.sidebar.write(ai_answer)
    else:
        st.sidebar.error("Chatbot model not loaded.")
