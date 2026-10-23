
import streamlit as st
import pickle
import time
if "history" not in st.session_state:
    st.session_state.history = []
    
# Load trained model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Page Config
st.set_page_config(
    page_title="AI Spam Detector",
    page_icon="📧",
    layout="centered"
)
st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        -45deg,
        #141e30,
        #243b55,
        #0f2027,
        #203a43
    );
    background-size: 400% 400%;
    animation: gradient 12s ease infinite;
}

@keyframes gradient {
    0% {background-position:0% 50%;}
    50% {background-position:100% 50%;}
    100% {background-position:0% 50%;}
}
            .fade-in {
    animation: fadeIn 1.5s ease-in;
}

@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(25px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
.stButton > button {
    border-radius: 12px;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 15px rgba(255,255,255,0.3);
}
</style>
""", unsafe_allow_html=True)

# Custom CSS
st.markdown("""
<style>

.stApp {
    background: linear-gradient(to right, #141e30, #243b55);
    color: white;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #ffffff;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #dcdcdc;
    margin-bottom: 30px;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}

.footer {
    text-align: center;
    margin-top: 50px;
    color: gray;
}

</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🚀 Project Details")

st.sidebar.markdown("""
### Technologies Used

- Python
- Scikit-Learn
- NLP
- Streamlit
- Machine Learning

### Model

Multinomial Naive Bayes

### Developer

Raunak Soni
""")
st.sidebar.metric("🎯 Model Accuracy", "98%")

# Main Title
st.markdown("""
<div class="fade-in">
    <h1 style="text-align:center;">
        📧 AI Spam Detector
    </h1>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">Smart Machine Learning model to detect spam messages instantly.</div>',
    unsafe_allow_html=True
)

# Input box
message = st.text_area(
    "✍️ Enter your message here:",
    height=180,
    placeholder="Example: Congratulations! You won a free iPhone..."
)
if message:
    words = len(message.split())
    chars = len(message)

    col1, col2 = st.columns(2)

    col1.metric("Words", words)
    col2.metric("Characters", chars)
    st.expander("📝 Try Sample Messages").write("""
Spam:
Congratulations! You won ₹50,000. Click now!

Genuine:
Hey, are we meeting tomorrow for class?
""")
# Predict button
if st.button("Analyze Message"):

    if message.strip() == "":
        st.warning("⚠️ Please enter a message first.")
    else:

        with st.spinner("Analyzing message..."):
            time.sleep(2)

        # Transform text
        data = vectorizer.transform([message])

        # Prediction
        prediction = model.predict(data)[0]

        # Probability
        probability = model.predict_proba(data)

        spam_prob = probability[0][1] * 100
        ham_prob = probability[0][0] * 100

        # Results
        if prediction == 1:

            st.markdown(f"""
            <div class="result-box" style="background-color:#ff4b4b;">
            🚨 Spam Message Detected
            </div>
            """, unsafe_allow_html=True)

            st.error(f"Spam Confidence: {spam_prob:.2f}%")
            st.progress(int(spam_prob))
            st.session_state.history.append(
    f"🚨 Spam ({spam_prob:.1f}%)"
)

        else:

            st.markdown(f"""
            <div class="result-box" style="background-color:#00c853;">
            ✅ This looks like a Genuine Message
            </div>
            """, unsafe_allow_html=True)

            st.success(f"Genuine Confidence: {ham_prob:.2f}%")
            st.progress(int(ham_prob))
            st.session_state.history.append(
    f"✅ Genuine ({ham_prob:.1f}%)"
)

# Footer
st.markdown(
    '<div class="footer">Made with ❤️ using Machine Learning & Streamlit</div>',
    unsafe_allow_html=True
)