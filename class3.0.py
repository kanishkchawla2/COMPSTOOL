import streamlit as st
import pandas as pd
import time
import google.generativeai as genai
import re
import json
import io
from datetime import datetime

# --- Page Configuration: Basic setup for the app ---
st.set_page_config(
    page_title="Industry Peer Analysis Tool",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS: To make the app look better ---
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #0284c7 0%, #0369a1 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .stButton > button {
        background: linear-gradient(90deg, #0ea5e9 0%, #0284c7 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.4);
    }
    .metric-container {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #0ea5e9;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization: To store user data across interactions ---
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'api_keys' not in st.session_state:
    st.session_state.api_keys = []
if 'selected_company' not in st.session_state:
    st.session_state.selected_company = ""


# --- Utility Functions: Helper functions for smaller tasks ---

@st.cache_data
def load_data(filepath="stock_info.xlsx"):
    """Loads data from the Excel file and caches it for performance."""
    try:
        df = pd.read_excel(filepath)
        # Standardize column names
        df.rename(columns={'Symbol': 'Company Name', 'BD': 'Business Description'}, inplace=True)
        required_cols = ['Company Name', 'Business Description', 'Industry']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Data file must contain the columns: {', '.join(required_cols)}")
            return None
        return df
    except FileNotFoundError:
        st.error(f"Error: The file '{filepath}' was not found. Please place it in the same folder as the script.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return None


def clean_relevance_score(score):
    """Safely converts the relevance score to a float between 0 and 100."""
    if pd.isna(score): return 0.00
    if isinstance(score, (int, float)): return float(score)
    if isinstance(score, str):
        cleaned = re.sub(r'[^\d.]', '', str(score))
        try:
            return float(cleaned) if cleaned else 0.00
        except ValueError:
            return 0.00
    return 0.00


def load_gemini_model(api_key):
    """Loads and validates the Gemini model with the given API key."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        # Simple test to validate the key and model
        if "OK" not in model.generate_content("Say OK").text:
            raise RuntimeError("Gemini model did not respond as expected.")
        return model
    except Exception as e:
        raise Exception(f"Failed to initialize Gemini model. Check your API Key. Error: {e}")


def process_batch(batch_df, target_bd, model):
    """Processes a single batch of companies by sending them to the AI."""
    companies_data = []
    for _, row in batch_df.iterrows():
        comp_name = row["Company Name"]
        comp_bd = row["Business Description"]
        if pd.isna(comp_bd) or str(comp_bd).strip() == "":
            comp_bd = "No business description available"
        companies_data.append({"name": str(comp_name), "description": str(comp_bd)})

    # This detailed prompt guides the AI to produce a structured JSON output
    prompt = f"""
You are a financial analyst specializing in competitive intelligence. Your task is to analyze a list of companies and compare them to a primary target company based on their business descriptions.

**TARGET COMPANY'S BUSINESS DESCRIPTION:**
{target_bd}

**COMPANIES TO ANALYZE (PEERS IN THE SAME INDUSTRY):**
{chr(10).join([f"{i + 1}. {comp['name']}: {comp['description']}" for i, comp in enumerate(companies_data)])}

For each company in the list, provide the following analysis:

1.  **Business Summary**: A concise 1-2 sentence summary of what the company does.
2.  **Business Model**: How the company primarily generates revenue (e.g., B2B, B2C, SaaS, advertising).
3.  **Key Products/Services**: The main products or services offered.
4.  **Relevance Score**: A numerical score from 1.00 to 100.00 indicating how similar the company's business is to the target company. A higher score means a more direct competitor. For the target company itself, this score should be 100.00.
5.  **Relevance Reason**: A brief 1-2 sentence explanation for the given relevance score.

**Required Response Format (Strict JSON):**
```json
{{
  "companies": [
    {{
      "company_name": "Company Name",
      "business_summary": "Clear summary of what they do.",
      "business_model": "How they make money.",
      "key_products_services": "Main products/services.",
      "relevance_score": 85.50,
      "relevance_reason": "Reason for the score, comparing to the target."
    }}
  ]
}}
```

IMPORTANT: The relevance_score MUST be a numeric value (like 85.50). Ensure the JSON is perfectly formatted.
"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            full_response = response.text.strip()

            # Robustly extract the JSON block from the AI's response
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', full_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_start = full_response.find('{')
                json_end = full_response.rfind('}') + 1
                if json_start == -1 or json_end == 0: raise ValueError("No JSON found in response")
                json_str = full_response[json_start:json_end]

            parsed_data = json.loads(json_str)
            companies_analysis = parsed_data.get('companies', [])

            batch_results = []
            for i, comp_data in enumerate(companies_data):
                original_row = batch_df.iloc[i]
                analysis = companies_analysis[i] if i < len(companies_analysis) else {}
                result_entry = {
                    "Company Name": comp_data["name"],
                    "Industry": original_row["Industry"],
                    "Original Business Description": comp_data["description"],
                    "Business Summary": analysis.get("business_summary", "N/A"),
                    "Business Model": analysis.get("business_model", "N/A"),
                    "Key Products/Services": analysis.get("key_products_services", "N/A"),
                    "Relevance Score": analysis.get("relevance_score", 0.00),
                    "Relevance Reason": analysis.get("relevance_reason", "AI did not return data for this company.")
                }
                batch_results.append(result_entry)
            return batch_results, None  # Success
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff for retries
                continue
            else:  # Final attempt failed
                error_results = [{
                    "Company Name": comp["name"], "Industry": batch_df.iloc[i]["Industry"],
                    "Original Business Description": comp["description"], "Business Summary": "Processing failed",
                    "Business Model": "Error", "Key Products/Services": "Error", "Relevance Score": 0.00,
                    "Relevance Reason": f"API/Parsing Error: {str(e)}"
                } for i, comp in enumerate(companies_data)]
                return error_results, f"A batch failed after {max_retries} attempts. Error: {e}"


def run_analysis(df_to_process, target_bd, batch_size, key_usage_limit):
    """Manages the analysis process, including progress bars and key rotation."""
    api_keys = st.session_state.api_keys
    if not api_keys:
        st.error("Cannot start analysis: No API keys have been provided.")
        return

    progress_bar = st.progress(0, "Initializing...")

    try:
        current_key_index = 0
        calls_with_current_key = 0
        model = load_gemini_model(api_keys[current_key_index])

        total_batches = (len(df_to_process) + batch_size - 1) // batch_size
        all_results = []

        for i in range(total_batches):
            # Rotate API key if usage limit is reached
            if calls_with_current_key >= key_usage_limit:
                current_key_index = (current_key_index + 1) % len(api_keys)
                st.toast(f"Switching to API Key #{current_key_index + 1}")
                model = load_gemini_model(api_keys[current_key_index])
                calls_with_current_key = 0  # Reset counter
            calls_with_current_key += 1

            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(df_to_process))
            batch_df = df_to_process.iloc[start_idx:end_idx]

            progress_bar.progress((i + 1) / total_batches, f"Processing Batch {i + 1}/{total_batches}...")
            batch_results, error = process_batch(batch_df, target_bd, model)

            if error: st.warning(error)  # Show non-fatal error to user

            all_results.extend(batch_results)
            if i < total_batches - 1: time.sleep(1)  # Brief pause between batches

        final_df = pd.DataFrame(all_results)
        final_df['Relevance Score'] = final_df['Relevance Score'].apply(clean_relevance_score).clip(0, 100)
        final_df = final_df.sort_values(by='Relevance Score', ascending=False)

        st.session_state.results_df = final_df
        st.session_state.processing_complete = True
        progress_bar.empty()
        st.success(f"✅ Analysis Complete! Processed {len(final_df)} companies.")
        st.balloons()

    except Exception as e:
        st.error(f"❌ A critical error occurred during analysis: {e}")
        progress_bar.empty()


# --- Main Application UI ---

master_df = load_data()  # Load data at the start

st.markdown("""
<div class="main-header">
    <h1>🔎 Industry Peer Analysis Tool</h1>
    <p>Select a company to analyze its direct competitors within the same industry using AI.</p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    st.subheader("🔐 Gemini API Keys")
    new_api_key = st.text_input("Add API Key", type="password", placeholder="Enter your Gemini API key here")

    col1, col2 = st.columns(2)
    if col1.button("➕ Add Key", use_container_width=True):
        if new_api_key and new_api_key not in st.session_state.api_keys:
            st.session_state.api_keys.append(new_api_key)
            st.success("API key added!")
            time.sleep(1);
            st.rerun()
    if col2.button("🗑️ Clear All", use_container_width=True):
        st.session_state.api_keys = []
        st.success("All keys cleared!")
        time.sleep(1);
        st.rerun()

    if st.session_state.api_keys:
        st.write(f"**Current Keys:** {len(st.session_state.api_keys)}")
        for i, key in enumerate(st.session_state.api_keys):
            st.code(f"Key {i + 1}: {key[:8]}...{key[-4:]}")
    else:
        st.warning("Please add at least one Gemini API key.")

    st.subheader("⚙️ Processing Settings")
    batch_size = st.slider("Batch Size", 1, 10, 5, help="Number of companies to process in each API call.")
    key_usage_limit = st.slider("Key Usage Limit", 5, 50, 20, help="API calls per key before rotating.")

# --- Main Content Tabs ---
tab1, tab2, tab3 = st.tabs(["▶️ Select & Analyze", "📊 Results", "📈 Analytics"])

with tab1:
    st.header("1. Select a Company for Analysis")
    if master_df is not None:
        company_list = master_df['Company Name'].unique().tolist()
        selected_company = st.selectbox(
            "Search for a company:",
            options=[""] + sorted(company_list),
            format_func=lambda x: "Select a company..." if x == "" else x,
            help="Choose the company you want to analyze."
        )

        # Store selected company in session state
        if selected_company:
            st.session_state.selected_company = selected_company

        if selected_company:
            target_company_data = master_df[master_df['Company Name'] == selected_company].iloc[0]
            target_bd = target_company_data['Business Description']
            target_industry = target_company_data['Industry']

            st.subheader(f"Target: {selected_company} ({target_industry})")
            with st.expander("Show Business Description"):
                st.write(target_bd)

            # This dataframe now includes the target company for analysis.
            companies_to_analyze_df = master_df[
                (master_df['Industry'] == target_industry)
            ].copy()

            st.header("2. Start Analysis")
            if not companies_to_analyze_df.empty:
                st.write(
                    f"Found **{len(companies_to_analyze_df)}** companies in the **'{target_industry}'** industry to analyze (including the target company).")
                if st.button(f"🚀 Analyze Industry for {selected_company}", type="primary"):
                    if st.session_state.api_keys:
                        run_analysis(companies_to_analyze_df, target_bd, batch_size, key_usage_limit)
                    else:
                        st.warning("⚠️ Please add at least one API key in the sidebar to start.")
            else:
                st.warning(f"No companies found in the '{target_industry}' industry to compare against.")
    else:
        st.error("Data could not be loaded. Please check the `stock_info.xlsx` file.")

with tab2:
    st.header("📊 Processing Results")
    if st.session_state.processing_complete and st.session_state.results_df is not None:
        df_results = st.session_state.results_df

        st.subheader("Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Companies Analyzed", len(df_results))
        col2.metric("High Relevance (>70)", len(df_results[df_results['Relevance Score'] >= 70]))
        col3.metric("Medium Relevance (50-69)",
                    len(df_results[(df_results['Relevance Score'] >= 50) & (df_results['Relevance Score'] < 70)]))
        col4.metric("Average Relevance Score", f"{df_results['Relevance Score'].mean():.2f}")

        st.subheader("📋 Detailed Results")
        st.dataframe(df_results, use_container_width=True, height=500)

        # Create an in-memory Excel file for download
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_results.to_excel(writer, sheet_name='Analysis_Results', index=False)
        output.seek(0)

        current_date = datetime.now().strftime('%Y%m%d_%H%M%S')
        st.download_button(
            label="📥 Download Results (Excel)",
            data=output,
            file_name=f"peer_analysis_{st.session_state.selected_company}_{current_date}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.info("📋 Results will appear here after an analysis is completed.")

with tab3:
    st.header("📈 Analytics Dashboard")
    if st.session_state.processing_complete and st.session_state.results_df is not None:
        df_results = st.session_state.results_df

        st.subheader("📊 Relevance Score Distribution")
        st.bar_chart(df_results['Relevance Score'].value_counts().sort_index())

        st.subheader("🏆 Top 10 Companies by Relevance (Including Target)")
        # Show all companies including the target company (which should have score 100)
        top_companies = df_results.head(10)

        # Highlight the target company in the display
        display_df = top_companies[['Company Name', 'Relevance Score', 'Business Model', 'Business Summary']].copy()

        # Add a column to indicate which is the target company
        if st.session_state.selected_company:
            display_df['Is Target'] = display_df['Company Name'] == st.session_state.selected_company
            display_df['Company Type'] = display_df['Is Target'].apply(lambda x: '🎯 TARGET' if x else '🏢 Peer')
            display_df = display_df[
                ['Company Name', 'Company Type', 'Relevance Score', 'Business Model', 'Business Summary']]

        st.dataframe(display_df, use_container_width=True)

        st.subheader("💼 Business Model Distribution")
        model_counts = df_results['Business Model'].value_counts().head(10)
        st.bar_chart(model_counts)

        # Additional section showing only peer companies (excluding target)
        if st.session_state.selected_company:
            st.subheader("🔍 Top Peer Companies (Excluding Target)")
            peer_companies = df_results[df_results['Company Name'] != st.session_state.selected_company].head(10)
            if not peer_companies.empty:
                st.dataframe(peer_companies[['Company Name', 'Relevance Score', 'Business Model', 'Business Summary']],
                             use_container_width=True)
            else:
                st.info("No peer companies found (only target company was analyzed).")
    else:
        st.info("📈 Analytics will be available after you run an analysis.")

# --- Footer ---
st.markdown("---")
st.markdown("Industry Peer Analysis Tool v2.1 | Built with Streamlit & Gemini")
