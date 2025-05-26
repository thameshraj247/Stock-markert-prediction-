import streamlit as st
from login import login_user
from predict import show_prediction_ui
# from pages.profit_loss_summary import show_profit_loss_summary

import os

st.set_page_config(
    page_title="📈 Stock Price Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Stock Market Price Predictor")
st.write("Use the sidebar to navigate.")

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# Sidebar Navigation
if 'choice' not in st.session_state:
    st.session_state.choice = "🔐 Login" if not st.session_state.logged_in else "🏠 Home"

if not st.session_state.logged_in:
    st.session_state.choice = st.sidebar.selectbox("Menu", ["🔐 Login"], index=0)
else:
    menu_options = [
        "🏠 Home",
        "📈 Predict Prices",
        "📊 Model Accuracy",
        "💰 Profit or Loss",
        "🚪 Logout"
    ]
    current_choice = st.session_state.choice if st.session_state.choice in menu_options else "🏠 Home"
    st.session_state.choice = st.sidebar.selectbox("Menu", menu_options, index=menu_options.index(current_choice))

# Route Logic
if st.session_state.choice == "🔐 Login":
    login_user()

elif st.session_state.choice == "🏠 Home":
    st.success(f"Welcome, {st.session_state.username}!")
    if st.button("Start Analysis"):
        st.session_state.choice = "📈 Predict Prices"

elif st.session_state.choice == "📈 Predict Prices":
    if st.session_state.logged_in:
        st.subheader("📉 LSTM-Based Stock Price Prediction")
        show_prediction_ui()
    else:
        st.warning("⚠️ Please login first to access predictions.")

elif st.session_state.choice == "📊 Model Accuracy":
    if st.session_state.logged_in:
        st.subheader("📊 Model Accuracy Comparison")
        if os.path.exists("model_accuracy.txt"):
            with open("model_accuracy.txt", "r") as file:
                st.text(file.read())
        else:
            st.warning("⚠️ Run prediction first to generate accuracy report.")
    else:
        st.warning("⚠️ Please login to view model accuracy.")

# elif st.session_state.choice == "💰 Profit or Loss":
#     if st.session_state.logged_in:
#         st.subheader("💰 Profit or Loss Summary")
#         show_profit_loss_summary()
#     else:
#         st.warning("⚠️ Please login to access profit/loss data.")


elif st.session_state.choice == "🚪 Logout":
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.success("Logged out successfully.")
    st.rerun()
