import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Simulate Data
data = {
    'member_id': range(1, 201),
    'booking_frequency': np.random.randint(0, 20, 200),
    'lesson_participation': np.random.randint(0, 10, 200),
    'membership_duration_months': np.random.randint(1, 60, 200),
    'days_since_last_booking': np.random.randint(0, 365, 200),
    'feedback_score': np.random.randint(1, 6, 200),
    'churn': np.random.choice([0, 1], size=200, p=[0.7, 0.3])
}
df = pd.DataFrame(data)
df['recent_inactivity'] = (df['days_since_last_booking'] > 90).astype(int)

# Prepare Data
X = df[['booking_frequency', 'lesson_participation', 'membership_duration_months', 'recent_inactivity', 'feedback_score']]
y = df['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Streamlit App
st.title("⛳ Member Churn Prediction App")

# Sidebar for New Member Input
st.sidebar.header("Input New Member Data")
booking_frequency = st.sidebar.slider("Booking Frequency (per month)", 0, 20, 5)
lesson_participation = st.sidebar.slider("Lesson Participation (last 6 months)", 0, 10, 2)
membership_duration = st.sidebar.slider("Membership Duration (months)", 1, 60, 12)
recent_inactivity = st.sidebar.selectbox("Recent Inactivity (>90 days)", [0, 1])
feedback_score = st.sidebar.slider("Feedback Score (1-5)", 1, 5, 3)

# Predict Button
if st.sidebar.button("Predict Churn"):
    new_member = pd.DataFrame({
        'booking_frequency': [booking_frequency],
        'lesson_participation': [lesson_participation],
        'membership_duration_months': [membership_duration],
        'recent_inactivity': [recent_inactivity],
        'feedback_score': [feedback_score]
    })

    churn_prediction = rf_model.predict(new_member)
    result = "Likely to Churn 🚩" if churn_prediction[0] == 1 else "Active 😊"
    st.success(f"Prediction: {result}")

# Display Model Performance
st.header("📊 Model Performance")
y_pred = rf_model.predict(X_test)
st.text(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Feature Importance
st.subheader("🔍 Feature Importance")
feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
fig, ax = plt.subplots()
feature_importance.sort_values().plot(kind='barh', ax=ax)
st.pyplot(fig)

# Upload Data for Batch Predictions (Optional)
st.header("📥 Upload CSV for Batch Predictions")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file:
    uploaded_data = pd.read_csv(uploaded_file)
    if set(X.columns).issubset(uploaded_data.columns):
        batch_predictions = rf_model.predict(uploaded_data[X.columns])
        uploaded_data['Churn Prediction'] = np.where(batch_predictions == 1, 'Likely to Churn', 'Active')
        st.write(uploaded_data)
    else:
        st.error("CSV format mismatch. Please ensure all required columns are included.")
