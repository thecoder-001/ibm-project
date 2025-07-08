import streamlit as st
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

sentiment_instruction = """
Classify the sentiment of the car rental review as Positive, Negative, or Neutral.

Review: The car was clean and the pickup process was very smooth.
Sentiment: Positive

Review: I had to wait for an hour and the car had a strange smell.
Sentiment: Negative

Review: The rental car was adequate for our trip.
Sentiment: Neutral

Review: {review_text}
Sentiment:
"""

issue_instruction = """
Identify the main issue from the review. Choose from: Car Condition, Staff Interaction, Booking Process, Billing Issue, or Other.

Review: The online booking was easy but they charged me for extra insurance I didn't ask for.
Topic: Billing Issue

Review: The representative at the counter was rude and unhelpful.
Topic: Staff Interaction

Review: The car's tires were worn out and the AC was not working.
Topic: Car Condition

Review: {review_text}
Topic:
"""

@st.cache_resource
def initialize_model():
    try:
        credentials = {
            "url": st.secrets["URL"],
            "apikey": st.secrets["IBM_CLOUD_API_KEY"]
        }
        project_id = st.secrets["PROJECT_ID"]
        
        params = {
            GenParams.MAX_NEW_TOKENS: 50,
            GenParams.TEMPERATURE: 0.1
        }

        model = Model(
            model_id='google/flan-ul2',
            params=params,
            credentials=credentials,
            project_id=project_id
        )
        return model
    except Exception as e:
        st.error(f"Error initializing model: {e}")
        st.error("Please check your API keys and Project ID in the .streamlit/secrets.toml file.")
        return None

st.set_page_config(page_title="Car Rental Feedback Analyzer", page_icon="🚗")

st.title("🚗 Car Rental Customer Feedback Analyzer")
st.markdown(
    "Enter a customer review below to automatically analyze its sentiment (Positive, Negative, or Neutral) "
    "and identify the key issue if the feedback is negative. Powered by **IBM Watsonx.ai**."
)

review_input = st.text_area(
    "Enter Customer Review Here:",
    height=150,
    placeholder="e.g., 'The car was fantastic and fuel-efficient, but the wait time at the counter was too long.'"
)

if st.button("Analyze Feedback", type="primary"):
    if not review_input.strip():
        st.warning("Please enter a review to analyze.")
    else:
        with st.spinner("Analyzing... Please wait."):
            model = initialize_model()
            
            if model:
                prompt_sentiment = sentiment_instruction.format(review_text=review_input)
                predicted_sentiment = model.generate_text(prompt=prompt_sentiment).strip()
                
                predicted_issue = None
                if predicted_sentiment.lower() == 'negative':
                    prompt_issue = issue_instruction.format(review_text=review_input)
                    predicted_issue = model.generate_text(prompt=prompt_issue).strip()
                
                st.subheader("Analysis Results")
                
                if predicted_sentiment.lower() == 'positive':
                    st.success(f"**Sentiment: {predicted_sentiment}**")
                    st.balloons()
                elif predicted_sentiment.lower() == 'negative':
                    st.error(f"**Sentiment: {predicted_sentiment}**")
                    if predicted_issue:
                        st.info(f"**Identified Issue: {predicted_issue}**")
                else:
                    st.warning(f"**Sentiment: {predicted_sentiment}**")

st.markdown("---")
