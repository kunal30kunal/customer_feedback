import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import random
import joblib
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from transformers import pipeline
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

# -----------------------------
# Optional features
# -----------------------------
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

# -----------------------------
# Dynamic text summarizer
# -----------------------------
def dynamic_summary(text):
    if not text or not isinstance(text, str):
        return "No text provided.", "No text provided."
    words = text.split()
    if not words:
        return "No text provided.", "No text provided."
    random.shuffle(words)
    short_summary = " ".join(words[:15]).capitalize() + "."
    intro = random.choice(["Overall, the customer felt that", "Feedback suggests", "From the review, it seems"])
    long_summary = f"{intro} {' '.join(words[:30])}."
    return short_summary, long_summary

# -----------------------------
# Load sentiment model
# -----------------------------
@st.cache_resource
def load_sentiment_model():
    try:
        model = joblib.load("sentiment_small_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        return model, vectorizer
    except Exception:
        return None, None

model, vectorizer = load_sentiment_model()

def predict_sentiment(text):
    if model is None or vectorizer is None:
        if any(w in text.lower() for w in ["good", "love", "excellent", "happy"]):
            return "Positive"
        elif any(w in text.lower() for w in ["bad", "hate", "poor", "terrible"]):
            return "Negative"
        else:
            return "Neutral"
    try:
        return model.predict(vectorizer.transform([text]))[0]
    except Exception:
        return "Neutral"

# -----------------------------
# Hugging Face Chatbot & Emotion
# -----------------------------
@st.cache_resource
def load_chatbot_and_emotion():
    try:
        chatbot = pipeline("text2text-generation", model="google/flan-t5-small")
    except Exception:
        chatbot = None
    try:
        emotion = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
    except Exception:
        emotion = None
    return chatbot, emotion

chatbot, emotion_classifier = load_chatbot_and_emotion()

# -----------------------------
# Keyword extraction
# -----------------------------
def extract_top_keywords(texts, top_n=10):
    try:
        tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
        X = tfidf.fit_transform(texts)
        sums = np.array(X.sum(axis=0)).flatten()
        words = np.array(tfidf.get_feature_names_out())
        sorted_idx = np.argsort(sums)[::-1]
        return words[sorted_idx][:top_n]
    except Exception:
        all_words = ' '.join(texts).split()
        return [w for w, _ in Counter(all_words).most_common(top_n)]

# -----------------------------
# Email Utility
# -----------------------------
def send_email_report(receiver_email, subject, message, sender_email, sender_password):
    try:
        msg = MIMEText(message, 'plain')
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = receiver_email

        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Email failed: {e}")
        return False

# -----------------------------
# PDF Generator
# -----------------------------
def generate_pdf(counts, ai_summary):
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(100, height - 80, "AI Customer Feedback Report")
    pdf.setFont("Helvetica", 11)
    pdf.drawString(100, height - 110, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    y = height - 150
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(100, y, "Sentiment Distribution:")
    y -= 20
    pdf.setFont("Helvetica", 11)
    for label, value in counts.items():
        pdf.drawString(120, y, f"{label}: {value}")
        y -= 15

    y -= 10
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(100, y, "AI Summary & Recommendations:")
    y -= 20
    text_obj = pdf.beginText(120, y)
    text_obj.setFont("Helvetica", 10)
    for line in ai_summary.split(". "):
        text_obj.textLine(line.strip())
    pdf.drawText(text_obj)

    pdf.showPage()
    pdf.save()
    buffer.seek(0)
    return buffer

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="AI Customer Feedback Insights", layout="wide")
st.title("🧠 AI Customer Feedback Insights Dashboard")

st.sidebar.header("📂 Upload or Input Data")
option = st.sidebar.radio("Choose input type:", ["Single Feedback", "CSV File"])

if option == "Single Feedback":
    text = st.text_area("Enter feedback:")
    if st.button("Analyze Feedback"):
        sentiment = predict_sentiment(text)
        short_sum, long_sum = dynamic_summary(text)
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Short Summary:** {short_sum}")
        st.write(f"**Long Summary:** {long_sum}")
else:
    uploaded_file = st.file_uploader("Upload CSV with 'Text' column", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'Text' in df.columns:
            df['Sentiment'] = df['Text'].apply(predict_sentiment)
            df['Short Summary'], df['Long Summary'] = zip(*df['Text'].apply(dynamic_summary))
            st.dataframe(df.head())

            # Pie Chart
            counts = df['Sentiment'].value_counts()
            plt.figure(figsize=(4,4))
            plt.pie(counts, labels=counts.index, autopct='%1.1f%%')
            st.pyplot(plt)

            # -----------------------------
            # AI Summary & Recommendations (Automatic)
            # -----------------------------
            st.subheader("🧠 AI Summary & Recommendations")

            summary_prompt = (
                f"Analyze this customer feedback sentiment summary and provide key insights with actionable recommendations:\n\n"
                f"Sentiment Counts:\n{counts.to_string()}\n\n"
                f"Sample Feedback:\n{df['Text'].head(5).to_string(index=False)}\n\n"
                f"Your task: Summarize key themes, suggest improvements, and recommend how to increase positive feedback."
            )

            if chatbot:
                try:
                    ai_summary = chatbot(summary_prompt, max_length=300)[0]['generated_text']
                except Exception as e:
                    ai_summary = f"(AI summary unavailable due to error: {e})"
            else:
                ai_summary = (
                    "Customers show mixed opinions. To improve satisfaction, focus on response time, product quality, "
                    "and communication clarity. Encourage happy customers to leave public reviews."
                )

            st.write(ai_summary)

            # PDF Generation
            if st.button("📄 Download PDF Report"):
                pdf_buffer = generate_pdf(counts, ai_summary)
                st.download_button(
                    label="⬇️ Click to Download PDF",
                    data=pdf_buffer,
                    file_name="AI_Feedback_Report.pdf",
                    mime="application/pdf"
                )

            # Email Section
            st.subheader("📧 Send Report via Email")
            receiver = st.text_input("Receiver Email")
            sender = st.text_input("Sender Email (Gmail)")
            password = st.text_input("Sender Password", type="password")
            subject = "AI Feedback Insights Report"
            message = (
                f"Report generated on {datetime.now()}\n\n"
                f"Sentiment distribution:\n{counts.to_string()}\n\n"
                f"AI Summary:\n{ai_summary}"
            )

            if st.button("Send Email Report"):
                if send_email_report(receiver, subject, message, sender, password):
                    st.success("✅ Email sent successfully!")
                else:
                    st.error("❌ Failed to send email.")

st.sidebar.markdown("---")
st.sidebar.header("💬 Ask AI")
query = st.sidebar.text_area("Ask about feedback insights:")
if st.sidebar.button("Ask AI"):
    if chatbot:
        ans = chatbot(f"Question: {query}")[0]['generated_text']
        st.sidebar.write(ans)
    else:
        st.sidebar.warning("Chatbot not available.")




