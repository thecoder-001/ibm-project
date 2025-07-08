import streamlit as st
import pandas as pd
import time
from io import BytesIO
import matplotlib.pyplot as plt
from fpdf import FPDF

from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams


sentiment_instruction = """
Classify the sentiment of the car rental review as Positive, Negative, or Neutral.
Review: {review_text}
Sentiment:
"""

issue_instruction = """
Identify the main issue from the review. Choose from: Car Condition, Staff Interaction, Pickup/Dropoff Experience, Billing/Pricing, Booking Process, Location/Facilities, Add-ons/Extras, Other.

Review: The online booking was easy but they charged me for extra insurance I didn't ask for.
Topic: Billing/Pricing

Review: The agent at the counter was rude and unhelpful.
Topic: Staff Interaction

Review: The car's tires were worn out and the AC was not working.
Topic: Car Condition

Review: I ordered a child seat, but it wasn't in the car when I arrived.
Topic: Add-ons/Extras

Review: The pickup took over an hour, the lines were huge.
Topic: Pickup/Dropoff Experience

Review: {review_text}
Topic:
"""

summary_generation_prompt = """
You are an expert operations analyst for a car rental company.
Based on the following summary statistics from customer reviews, write a brief, professional, and actionable summary for management.
Focus on the top 2-3 most critical areas for improvement and mention one key strength to maintain.
Use bullet points for clarity.

Statistics:
{stats_text}

Analysis:
"""


class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Car Rental Feedback Analysis Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_pdf_report(df_results, summary_text):
    pdf = PDF()
    pdf.add_page()
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '1. Overall Performance Summary', 0, 1)
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 5, f"Total reviews analyzed: {len(df_results)}\nReport generated on: {pd.Timestamp.now().strftime('%Y-%m-%d')}")
    pdf.ln(5)

    sentiment_counts = df_results['Predicted_Sentiment'].value_counts()
    fig1, ax1 = plt.subplots(figsize=(4, 3))
    ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['#d4edda', '#f8d7da', '#fff3cd'])
    ax1.axis('equal')
    plt.title('Sentiment Distribution')
    
    img_buffer1 = BytesIO()
    plt.savefig(img_buffer1, format='png', bbox_inches='tight')
    img_buffer1.seek(0)
    pdf.image(img_buffer1, x=10, w=pdf.w - 20)
    plt.close(fig1)
    pdf.ln(5)

    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '2. Breakdown of Negative Feedback', 0, 1)
    
    negative_reviews = df_results[df_results['Predicted_Sentiment'] == 'Negative']
    if not negative_reviews.empty:
        issue_counts = negative_reviews['Predicted_Issue'].value_counts()
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        issue_counts.sort_values().plot(kind='barh', ax=ax2, color='#f5c6cb')
        ax2.set_xlabel('Number of Mentions')
        plt.title('Top Issues in Negative Reviews')

        img_buffer2 = BytesIO()
        plt.savefig(img_buffer2, format='png', bbox_inches='tight')
        img_buffer2.seek(0)
        pdf.image(img_buffer2, x=10, w=pdf.w - 20)
        plt.close(fig2)
    else:
        pdf.set_font('Arial', '', 11)
        pdf.cell(0, 10, 'No negative reviews were found in this batch.', 0, 1)
    pdf.ln(5)

    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '3. AI-Generated Actionable Insights', 0, 1)
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 6, summary_text.encode('latin-1', 'replace').decode('latin-1'))
    
    return bytes(pdf.output(dest='S'))


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
        model = Model(model_id='google/flan-ul2', params=params, credentials=credentials, project_id=project_id)
        return model
    except Exception as e:
        st.error(f"Error initializing model: {e}")
        return None

def analyze_single_review(model, review_text):
    prompt_sentiment = sentiment_instruction.format(review_text=review_text)
    predicted_sentiment = model.generate_text(prompt=prompt_sentiment).strip()
    predicted_issue = "N/A"
    if predicted_sentiment.lower() == 'negative':
        prompt_issue = issue_instruction.format(review_text=review_text)
        predicted_issue = model.generate_text(prompt=prompt_issue).strip()
    return predicted_sentiment, predicted_issue

st.set_page_config(page_title="Car Rental Feedback Analyzer", page_icon="🚗", layout="wide")
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose Analysis Mode", ["Interactive Analysis", "Batch Processing (Upload CSV)"])
st.title("🚗 Car Rental Feedback Analyzer")
st.markdown("An automated tool to analyze customer feedback for sentiment and key issues, powered by **IBM Watsonx.ai**.")

model = initialize_model()

if app_mode == "Interactive Analysis":
    st.header("Single Review Analysis")
    review_input = st.text_area("Enter Customer Review Here:", height=150)
    if st.button("Analyze Feedback", type="primary"):
        if review_input.strip() and model:
            with st.spinner("Analyzing..."):
                sentiment, issue = analyze_single_review(model, review_input)
                # Display results
                if sentiment.lower() == 'positive': st.success(f"**Sentiment: {sentiment}**"); st.balloons()
                elif sentiment.lower() == 'negative': st.error(f"**Sentiment: {sentiment}**"); st.info(f"**Identified Issue: {issue}**")
                else: st.warning(f"**Sentiment: {sentiment}**")

elif app_mode == "Batch Processing (Upload CSV)":
    st.header("Batch Feedback Analysis")
    uploaded_file = st.file_uploader("Upload a CSV file with a header row", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of your data:")
            st.dataframe(df.head())
            review_column = st.selectbox("Which column contains the reviews?", df.columns)
            
            if st.button("Start Batch Analysis", type="primary"):
                if model:
                    results_list = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    for index, row in df.iterrows():
                        sentiment, issue = analyze_single_review(model, str(row[review_column]))
                        results_list.append({'Review': str(row[review_column]), 'Predicted_Sentiment': sentiment, 'Predicted_Issue': issue})
                        time.sleep(0.5)
                        progress = (index + 1) / len(df)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing review {index + 1}/{len(df)}")
                    
                    status_text.success("Batch analysis complete!")
                    results_df = pd.DataFrame(results_list)
                    
                    st.subheader("Analysis Results")
                    st.dataframe(results_df)

                    with st.spinner("Generating summary report..."):
                        sentiment_counts = results_df['Predicted_Sentiment'].value_counts(normalize=True).mul(100).round(1).to_dict()
                        negative_issue_counts = results_df[results_df['Predicted_Sentiment'] == 'Negative']['Predicted_Issue'].value_counts(normalize=True).mul(100).round(1).to_dict()
                        stats = f"""- Total Reviews: {len(results_df)}
- Sentiment Distribution: {sentiment_counts}
- Breakdown of Negative Issues: {negative_issue_counts if negative_issue_counts else 'None'}"""
                        
                        summary_prompt = summary_generation_prompt.format(stats_text=stats)
                        ai_summary_text = model.generate_text(prompt=summary_prompt)

                        pdf_data = create_pdf_report(results_df, ai_summary_text)

                    st.subheader("Download Reports")
                    col1, col2 = st.columns(2)
                    with col1:
                        csv_output = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(label="📥 Download Results as CSV", data=csv_output, file_name='analyzed_feedback.csv', mime='text/csv')
                    with col2:
                        st.download_button(label="📄 Download Summary Report as PDF", data=pdf_data, file_name='feedback_analysis_report.pdf', mime='application/pdf')

        except Exception as e:
            st.error(f"An error occurred: {e}")