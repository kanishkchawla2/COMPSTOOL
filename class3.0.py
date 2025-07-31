import streamlit as st
import pandas as pd
import numpy as np
import time
import google.generativeai as genai
import re
import json
import io
import yfinance as yf
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
    /* Hide GitHub button */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
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
    .yfinance-section {
        background: #f1f5f9;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        margin: 1rem 0;
    }
    /* Style for summary rows */
    .summary-row {
        font-weight: bold;
        background-color: #e2e8f0 !important;
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
if 'yfinance_data' not in st.session_state:
    st.session_state.yfinance_data = None
if 'available_yfinance_cols' not in st.session_state:
    st.session_state.available_yfinance_cols = []
if 'selected_yfinance_cols' not in st.session_state:
    st.session_state.selected_yfinance_cols = []


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


def add_ns_suffix(symbol):
    """Add .NS suffix to symbol if it doesn't already have it for Indian stocks."""
    if not (symbol.endswith('.NS') or symbol.endswith('.BO')):
        return symbol + '.NS'
    return symbol


def fetch_yfinance_data(symbols):
    """Fetch all available yfinance data for a list of company symbols."""
    data = []
    progress_bar = st.progress(0, "Fetching financial data...")
    successful_fetches = 0
    failed_fetches = 0

    for i, symbol in enumerate(symbols):
        try:
            yf_symbol = add_ns_suffix(symbol)
            ticker = yf.Ticker(yf_symbol)
            info = ticker.info

            if info and len(info) > 1:
                info["Company Name"] = symbol  # Use original symbol as key
                info["yf_symbol"] = yf_symbol  # Store the yfinance symbol
                data.append(info)
                successful_fetches += 1
                st.toast(f"✅ Fetched data for {symbol}")
            else:
                failed_fetches += 1
                st.toast(f"⚠️ No data for {symbol}")

        except Exception as e:
            failed_fetches += 1
            st.toast(f"❌ Failed for {symbol}: {str(e)}")

        progress_bar.progress((i + 1) / len(symbols), f"Processing {i + 1}/{len(symbols)} symbols...")
        
        # Add 5-second delay after every 5 stocks to prevent rate limiting
        if (i + 1) % 5 == 0 and (i + 1) < len(symbols):
            st.info(f"⏳ Taking a 5-second break after processing {i + 1} stocks to prevent rate limiting...")
            time.sleep(5)

    progress_bar.empty()

    if data:
        df = pd.DataFrame(data)
        st.success(f"✅ Successfully fetched data for {successful_fetches} companies. {failed_fetches} failed.")
        return df
    else:
        st.error("❌ No financial data was successfully fetched")
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
        model = genai.GenerativeModel("gemini-2.5-flash-lite-preview-06-17")
        # Simple test to validate the key and model
        if "OK" not in model.generate_content("Say OK").text:
            raise RuntimeError("Gemini model did not respond as expected.")
        return model
    except Exception as e:
        raise Exception(f"Failed to initialize Gemini model. Check your API Key. Error: {e}")


def process_batch(batch_df, target_bd, model, target_company_name):
    """Processes a single batch of companies by sending them to the AI."""
    companies_data = []
    for _, row in batch_df.iterrows():
        comp_name = row["Company Name"]
        comp_bd = row["Business Description"]
        if pd.isna(comp_bd) or str(comp_bd).strip() == "":
            comp_bd = "No business description available"
        companies_data.append({"name": str(comp_name), "description": str(comp_bd)})

    # --- MODIFIED PROMPT ---
    # Added a more forceful instruction to ensure all companies are returned.
    prompt = f"""
You are a financial analyst specializing in competitive intelligence. Your task is to analyze a list of companies and compare them to a primary target company based on their business descriptions.

**TARGET COMPANY'S NAME:** {target_company_name}
**TARGET COMPANY'S BUSINESS DESCRIPTION:**
{target_bd}

**COMPANIES TO ANALYZE (PEERS IN THE SAME INDUSTRY):**
{chr(10).join([f"{i + 1}. {comp['name']}: {comp['description']}" for i, comp in enumerate(companies_data)])}

For each company in the list, provide the following analysis:

1.  **Business Summary**: A concise 1-2 sentence summary of what the company does.
2.  **Business Model**: How the company primarily generates revenue (e.g., B2B, B2C, SaaS, advertising).
3.  **Key Products/Services**: The main products or services offered.
4.  **Relevance Score**: A numerical score from 1.00 to 100.00 indicating how similar the company's business is to the target company. A higher score means a more direct competitor. If the company being analyzed is the target company itself ({target_company_name}), its score MUST be 100.00.
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

IMPORTANT:
- The `relevance_score` MUST be a numeric value (like 85.50).
- The JSON response MUST be perfectly formatted.
- It is MANDATORY to return a JSON object for every single company provided in the input list. Do not omit any company. If you lack specific information for a company, fill the fields with your best estimate or state "Information not available".
"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            full_response = response.text.strip()

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
                # Find the matching analysis by name, case-insensitively
                analysis = next((item for item in companies_analysis if
                                 item.get("company_name", "").lower() == comp_data["name"].lower()), {})

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
            return batch_results, None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            else:
                error_results = [{
                    "Company Name": comp["name"], "Industry": batch_df.iloc[i]["Industry"],
                    "Original Business Description": comp["description"], "Business Summary": "Processing failed",
                    "Business Model": "Error", "Key Products/Services": "Error", "Relevance Score": 0.00,
                    "Relevance Reason": f"API/Parsing Error: {str(e)}"
                } for i, comp in enumerate(companies_data)]
                return error_results, f"A batch failed after {max_retries} attempts. Error: {e}"


def run_analysis(df_to_process, target_bd, batch_size, key_usage_limit, target_company_name):
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
            if calls_with_current_key >= key_usage_limit:
                current_key_index = (current_key_index + 1) % len(api_keys)
                st.toast(f"Switching to API Key #{current_key_index + 1}")
                model = load_gemini_model(api_keys[current_key_index])
                calls_with_current_key = 0
            calls_with_current_key += 1

            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(df_to_process))
            batch_df = df_to_process.iloc[start_idx:end_idx]

            progress_bar.progress((i + 1) / total_batches, f"Processing Batch {i + 1}/{total_batches}...")
            batch_results, error = process_batch(batch_df, target_bd, model, target_company_name)

            if error: st.warning(error)

            all_results.extend(batch_results)
            if i < total_batches - 1: time.sleep(1)

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


def merge_yfinance_data(results_df, yfinance_df, selected_cols):
    """Merge the analysis results with selected yfinance data."""
    if yfinance_df is None or results_df is None or not selected_cols:
        return results_df

    yf_subset = yfinance_df[['Company Name'] + selected_cols].copy()
    merged_df = pd.merge(results_df, yf_subset, on='Company Name', how='left')
    return merged_df


# --- Main Application UI ---

master_df = load_data()

st.markdown("""
<div class="main-header">
    <h1>🔎 Industry Peer Analysis Tool</h1>
    <p>Select a company to analyze its direct competitors within the same industry using AI and financial data.</p>
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
tab1, tab2, tab3, tab4 = st.tabs(["▶️ Select & Analyze", "📊 Results", "📈 Analytics", "💰 Financial Data"])

with tab1:
    st.header("1. Select a Company for Analysis")
    if master_df is not None:
        symbols_list = master_df['Company Name'].unique().tolist()
        selected_symbol = st.selectbox(
            "Search for a company by its symbol:",
            options=[""] + sorted(symbols_list),
            format_func=lambda x: "Select a symbol..." if x == "" else x,
            help="Choose the company you want to analyze."
        )

        if selected_symbol:
            target_company_data = master_df[master_df['Company Name'] == selected_symbol].iloc[0]
            target_bd = target_company_data['Business Description']
            target_industry = target_company_data['Industry']

            st.subheader(f"Target: {selected_symbol} ({target_industry})")
            with st.expander("Show Business Description"):
                st.write(target_bd)

            # This dataframe now includes the target company for analysis.
            companies_to_analyze_df = master_df[
                master_df['Industry'] == target_industry
                ].copy()

            st.header("2. Start Analysis")
            if not companies_to_analyze_df.empty:
                st.write(
                    f"Found **{len(companies_to_analyze_df)}** companies in the **'{target_industry}'** industry to analyze (including the target).")
                if st.button(f"🚀 Analyze Peers of {selected_symbol}", type="primary"):
                    if st.session_state.api_keys:
                        run_analysis(companies_to_analyze_df, target_bd, batch_size, key_usage_limit, selected_symbol)
                    else:
                        st.warning("⚠️ Please add at least one API key in the sidebar to start.")
            else:
                st.warning(f"No companies found in the '{target_industry}' industry to compare against.")
    else:
        st.error("Data could not be loaded. Please check the `stock_data.xlsx` file.")

with tab2:
    st.header("📊 Processing Results")
    if st.session_state.processing_complete and st.session_state.results_df is not None:
        # Make a copy to work with, preserving the original state
        df_results = st.session_state.results_df.copy()

        st.markdown('<div class="yfinance-section">', unsafe_allow_html=True)
        st.subheader("💰 Enhance with Financial Data")

        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("🔄 Fetch Financial Data", help="Fetch yfinance data for all companies in results"):
                with st.spinner("Fetching data... this may take a moment."):
                    company_symbols = df_results['Company Name'].unique().tolist()
                    yf_data = fetch_yfinance_data(company_symbols)
                    if yf_data is not None:
                        st.session_state.yfinance_data = yf_data
                        available_cols = sorted(
                            [col for col in yf_data.columns if col not in ['Company Name', 'yf_symbol']])
                        st.session_state.available_yfinance_cols = available_cols
                        st.rerun()

        with col2:
            if st.session_state.yfinance_data is not None:
                st.metric("Financial Data Points", f"{len(st.session_state.yfinance_data)} companies")

        if st.session_state.available_yfinance_cols:
            st.subheader("📋 Select Financial Metrics to Include")

            b_col1, b_col2, b_col3 = st.columns(3)
            if b_col1.button("✅ Select All Metrics", use_container_width=True):
                st.session_state.selected_yfinance_cols = st.session_state.available_yfinance_cols
                st.rerun()
            if b_col2.button("📈 Select Key Valuation Metrics", use_container_width=True):
                st.session_state.selected_yfinance_cols = [
                    c for c in
                    ['currentPrice', 'marketCap', 'enterpriseValue', 'forwardPE', 'trailingPE', 'priceToBook',
                     'priceToSalesTrailing12Months']
                    if c in st.session_state.available_yfinance_cols
                ]
                st.rerun()
            if b_col3.button("🗑️ Clear Selection", use_container_width=True):
                st.session_state.selected_yfinance_cols = []
                st.rerun()

            selected_cols = st.multiselect(
                "Choose financial metrics:",
                options=st.session_state.available_yfinance_cols,
                default=st.session_state.selected_yfinance_cols,
                help="Select the financial metrics you want to include in your analysis"
            )
            if selected_cols != st.session_state.selected_yfinance_cols:
                st.session_state.selected_yfinance_cols = selected_cols
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

        # Merge data if selections are made
        if st.session_state.selected_yfinance_cols and st.session_state.yfinance_data is not None:
            df_results = merge_yfinance_data(
                df_results,
                st.session_state.yfinance_data,
                st.session_state.selected_yfinance_cols
            )

        # --- NEW: Add the relevance score slider ---
        relevance_threshold = st.slider(
            "Filter by Minimum Relevance Score",
            min_value=0,
            max_value=100,
            value=0,  # Default to show all
            help="Show companies with a relevance score greater than or equal to the selected value."
        )

        # --- Apply the filter for display ---
        df_for_display = df_results[df_results['Relevance Score'] >= relevance_threshold]

        # --- Update metrics based on the filtered view ---
        st.subheader("📊 Filtered Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Companies Displayed", len(df_for_display))
        col2.metric("High Relevance (>70)", len(df_for_display[df_for_display['Relevance Score'] >= 70]))
        col3.metric("Medium Relevance (50-69)",
                    len(df_for_display[
                            (df_for_display['Relevance Score'] >= 50) & (df_for_display['Relevance Score'] < 70)]))
        avg_score = df_for_display['Relevance Score'].mean() if not df_for_display.empty else 0
        col4.metric("Average Relevance Score", f"{avg_score:.2f}")

        st.subheader("📋 Detailed Results Editor")
        st.info("ℹ️ You can delete rows from the table. Double-click any cell to see its full content.")

        # --- Give the filtered data to the editor ---
        edited_df = st.data_editor(
            df_for_display,
            key="results_editor",
            use_container_width=True,
            height=500,
            num_rows="dynamic"  # Allows row deletion
        )

        # --- Check if the edited view is different from the displayed view ---
        if not edited_df.equals(df_for_display):
            # If the user deletes a row, update the main session state dataframe
            # by keeping only the rows that are still present in the edited_df.
            # This works because edited_df preserves the original index.
            st.session_state.results_df = st.session_state.results_df.loc[edited_df.index]
            st.rerun()

        # --- Calculate and display summary rows based on the final displayed data ---
        if not edited_df.empty:
            st.subheader("📊 Summary Statistics for Displayed Data")
            numeric_cols = edited_df.select_dtypes(include=np.number).columns.tolist()

            if numeric_cols:
                avg_series = edited_df[numeric_cols].mean()
                median_series = edited_df[numeric_cols].median()
                summary_df = pd.DataFrame([avg_series, median_series], index=['Average', 'Median']).round(2)
                st.dataframe(summary_df, use_container_width=True)
            else:
                st.warning("No numeric columns available to calculate summary statistics for the current view.")

        # --- Download button uses the final displayed (and potentially edited) data ---
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            edited_df.to_excel(writer, sheet_name='Filtered_Analysis_Results', index=False)
            if st.session_state.yfinance_data is not None:
                st.session_state.yfinance_data.to_excel(writer, sheet_name='Raw_Financial_Data', index=False)
        output.seek(0)

        filename_suffix = "enhanced_filtered" if st.session_state.selected_yfinance_cols else "basic_filtered"
        st.download_button(
            label="📥 Download Displayed Results (Excel)",
            data=output,
            file_name=f"peer_analysis_{filename_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.info("📋 Results will appear here after an analysis is completed.")

with tab3:
    st.header("📈 Analytics Dashboard")
    # --- MODIFIED: Use the potentially edited dataframe from session state ---
    if st.session_state.processing_complete and st.session_state.results_df is not None:
        df_results = st.session_state.results_df

        st.subheader("📊 Relevance Score Distribution (All Results)")
        st.bar_chart(df_results.set_index('Company Name')['Relevance Score'])

        st.subheader("🏆 Top 10 Most Relevant Peers (All Results)")
        top_peers = df_results.head(11)[['Company Name', 'Relevance Score', 'Business Model', 'Business Summary']]
        st.dataframe(top_peers, use_container_width=True)

        st.subheader("💼 Business Model Distribution (All Results)")
        model_counts = df_results['Business Model'].value_counts().head(10)
        st.bar_chart(model_counts)
    else:
        st.info("📈 Analytics will be available after you run an analysis.")

with tab4:
    st.header("💰 Financial Data Management")
    if st.session_state.yfinance_data is not None:
        st.subheader("📊 Available Financial Data")
        st.write(f"Financial data available for {len(st.session_state.yfinance_data)} companies.")

        st.subheader("🔍 Preview of Financial Data")
        st.dataframe(st.session_state.yfinance_data.head(10), use_container_width=True)

        st.subheader("📋 All Available Financial Metrics")
        st.info(f"A total of {len(st.session_state.available_yfinance_cols)} metrics were fetched from yfinance.")

        num_cols = 4
        cols = st.columns(num_cols)
        for i, col_name in enumerate(st.session_state.available_yfinance_cols):
            with cols[i % num_cols]:
                st.write(f"• `{col_name}`")

        st.subheader("📥 Download Raw Financial Data")
        financial_output = io.BytesIO()
        st.session_state.yfinance_data.to_excel(financial_output, index=False, engine='openpyxl')
        financial_output.seek(0)

        st.download_button(
            label="📥 Download All Financial Data (Excel)",
            data=financial_output,
            file_name=f"raw_financial_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.info("💡 Financial data will appear here after you fetch it from the Results tab.")

# --- Footer ---
st.markdown("---")
st.markdown("Industry Peer Analysis Tool v5.0 | Built with Streamlit, Gemini & yfinance")
