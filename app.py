# app.py

import os
import io
import glob
import joblib
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

from preprocessing import clean_text


# -------------- Helper functions for models -------------- #

def get_embedding_model():
    """
    Load the MiniLM model once and keep it in session_state.
    This avoids reloading on every button click.
    """
    if "embedding_model" not in st.session_state:
        st.session_state["embedding_model"] = SentenceTransformer("all-MiniLM-L6-v2")
    return st.session_state["embedding_model"]


def get_classifier():
    """
    Load the trained classifier from models/priority_classifier.pkl
    and keep it in session_state.
    """
    if "classifier" not in st.session_state:
        model_path = os.path.join("models", "priority_classifier.pkl")
        if not os.path.exists(model_path):
            # This will show a clear error in the UI and stop execution
            st.error(
                "Trained model not found.\n\n"
                "Please run `python train_model.py` in the project folder first."
            )
            st.stop()
        st.session_state["classifier"] = joblib.load(model_path)
    return st.session_state["classifier"]


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure subject+body exist and create a cleaned text column.
    """
    if "subject" not in df.columns or "body" not in df.columns:
        raise ValueError("CSV must contain 'subject' and 'body' columns.")

    df = df.copy()
    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")
    df["text"] = (df["subject"] + " " + df["body"]).apply(clean_text)
    return df


def predict_urgency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe with 'text' column, add 'priority_predicted'.
    """
    model = get_embedding_model()
    clf = get_classifier()

    texts = df["text"].tolist()
    embeddings = model.encode(texts, show_progress_bar=False)
    preds = clf.predict(embeddings)

    df = df.copy()
    df["priority_predicted"] = preds
    return df


# -------------- Streamlit UI -------------- #

def main():
    # Custom CSS for better styling
    st.markdown("""
    <style>
    /* Main background */
    .main {
        background-color: #f0f5ff;
    }
    .stApp {
        background: linear-gradient(135deg, #f0f5ff 0%, #e6f0ff 100%);
    }
    .stButton>button {
        background-color: #4a6cf7;
        color: white;
        font-weight: 500;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #3a5bd9;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(74, 108, 247, 0.2);
    }
    /* Styling for all dataframes */
    .stDataFrame {
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        background-color: white;
        border: 1px solid #e1e4e8;
    }
    
    /* Styling for dataframe headers */
    .stDataFrame thead th {
        background-color: #4a6cf7 !important;
        color: white !important;
        font-weight: 600 !important;
    }
    
    /* Styling for dataframe rows */
    .stDataFrame tbody tr:nth-child(even) {
        background-color: #f8f9ff;
    }
    
    .stDataFrame tbody tr:hover {
        background-color: #f0f4ff !important;
    }
    .stSelectbox, .stFileUploader {
        margin-bottom: 1rem;
    }
    .stAlert {
        border-radius: 8px;
    }
    .stSuccess {
        background-color: #e8f5e9;
        color: #2e7d32;
    }
    .stError {
        background-color: #ffebee;
        color: #c62828;
    }
    .stInfo {
        background-color: #e3f2fd;
        color: #1565c0;
    }
    /* Headers styling */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #2c3e50;
        border-bottom: 2px solid #e1e4e8;
        padding-bottom: 8px;
        margin-top: 1.5em;
    }
    /* Paragraph and text styling */
    .stMarkdown p {
        color: #2c3e50;
        line-height: 1.6;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f0f7ff 0%, #e6f0ff 100%) !important;
        border-right: 1px solid #d0e0ff !important;
        box-shadow: 2px 0 10px rgba(0, 0, 0, 0.05) !important;
    }
    
    /* Sidebar header */
    [data-testid="stSidebar"] .st-emotion-cache-1cypcdb {
        background: transparent !important;
        padding: 2rem 1.5rem 1rem !important;
    }
    
    /* Sidebar content */
    [data-testid="stSidebar"] .st-emotion-cache-1cypcdb > div:first-child {
        background: transparent !important;
    }
    
    /* Sidebar text */
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown label,
    [data-testid="stSidebar"] .stMarkdown strong {
        color: #2c3e50 !important;
    }
    
    /* Sidebar inputs */
    [data-testid="stSidebar"] .stSelectbox,
    [data-testid="stSidebar"] .stFileUploader,
    [data-testid="stSidebar"] .stTextInput {
        background-color: rgba(255, 255, 255, 0.7) !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
        border: 1px solid #d0e0ff !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
    }
    
    /* Sidebar buttons */
    [data-testid="stSidebar"] .stButton>button {
        background: linear-gradient(90deg, #4a6cf7 0%, #3a5bd9 100%) !important;
        border: none !important;
        width: 100% !important;
        margin: 5px 0 !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stSidebar"] .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(74, 108, 247, 0.2) !important;
    }
    
    /* Sidebar section headers */
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #2c3e50 !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #d0e0ff !important;
        padding-bottom: 8px !important;
        margin: 1.5rem 0 1rem 0 !important;
    }
    
    /* Card styling */
    .stAlert, .stInfo, .stSuccess, .stError {
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #4a6cf7;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("📧 Email Priority Classification Dashboard")
    st.markdown("""
        <div style='background-color: #f0f4ff; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;'>
            <p style='margin: 0; color: #1a237e;'>
                This app classifies emails into <strong>Urgent</strong>, <strong>Normal</strong>, or <strong>Low Priority</strong>
                using semantic text embeddings (MiniLM) and a machine learning classifier.
            </p>
        </div>
    """, unsafe_allow_html=True)


    # Get top 5 CSV or XLSX files from data folder
    data_dir = "data"
    sample_files = {}
    if os.path.exists(data_dir) and os.path.isdir(data_dir):
        # Get all CSV and XLSX files, sort them alphabetically, and take top 5
        data_files = []
        for ext in ('*.csv', '*.xlsx'):
            data_files.extend(glob.glob(os.path.join(data_dir, ext)))
        
        # Sort files alphabetically and take top 5
        data_files = sorted(data_files)[:5]
        
        # Create mapping of display names to file paths
        sample_files = {
            f"{os.path.basename(f)} (Sample)": f 
            for f in data_files
        }

    dataset_option = st.sidebar.selectbox(
        "Choose a sample dataset (optional):",
        ["None"] + list(sample_files.keys()),
    )

    st.sidebar.write("**OR** upload your own file:")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV or Excel file with 'subject' and 'body' columns",
        type=["csv", "xlsx"],
    )
    
    # Add a button to show model metrics
    st.sidebar.markdown("---")
    if st.sidebar.button("📊 View Model Metrics"):
        st.sidebar.markdown("### Model Performance")
        st.sidebar.markdown("**Accuracy:** 97.3%")
        
        # More compact table using HTML/CSS
        st.sidebar.markdown("""
        <style>
        .metrics-table {
            font-size: 12px;
            width: 100%;
            border-collapse: collapse;
            margin: 8px 0;
        }
        .metrics-table th, .metrics-table td {
            padding: 4px 6px;
            text-align: left;
            border-bottom: 1px solid #e1e4e8;
        }
        .metrics-table th {
            background-color: #f0f4ff;
            font-weight: 600;
        }
        .metrics-table tr:last-child td {
            border-bottom: none;
        }
        .metrics-avg {
            font-size: 11px;
            margin-top: 4px;
            color: #4a5568;
        }
        </style>
        
        <table class="metrics-table">
            <tr><th>Class</th><th>Prec</th><th>Rec</th><th>F1</th><th>support</th></tr>
            <tr><td>Low</td><td>1.00</td><td>0.92</td><td>0.96</td><td>12</td></tr>
            <tr><td>Normal</td><td>1.00</td><td>1.00</td><td>1.00</td><td>13</td></tr>
            <tr><td>Urgent</td><td>0.92</td><td>1.00</td><td>0.96</td><td>12</td></tr>
        </table>
        
        <div class="metrics-avg">
            <div>Macro Avg: Prec 0.97, Rec 0.97, F1 0.97</div>
            <div>Weighted: Prec 0.98, Rec 0.97, F1 0.97</div>
        </div>
        """, unsafe_allow_html=True)

    df = None

    # If user selected a sample dataset or uploaded a file
    if dataset_option != "None" or uploaded_file is not None:
        try:
            if dataset_option != "None":
                # Handle sample dataset
                file_path = sample_files[dataset_option]
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                else:  # .xlsx
                    df = pd.read_excel(file_path)
                st.sidebar.success(f"Loaded {len(df)} emails from sample dataset")
            elif uploaded_file is not None:  # Handle uploaded file
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:  # .xlsx
                    df = pd.read_excel(uploaded_file)
                st.sidebar.success(f"Successfully uploaded {len(df)} emails")
            
            # Preprocess the dataframe
            df = preprocess_dataframe(df)
            
            # Show preview of the data with index starting from 1
            st.subheader("Data Preview")
            # Only show the original columns, not the processed 'text' column
            preview_df = df[['subject', 'body']].head().copy()
            preview_df.index = range(1, len(preview_df) + 1)
            
            # Set pandas display options for better text wrapping
            pd.set_option('display.max_colwidth', 500)
            
            # Display the dataframe with custom CSS for scrollable cells
            st.markdown("""
            <style>
            .dataframe tbody tr th:first-child {display:none}
            .dataframe thead th:first-child {display:none}
            .dataframe tbody td {
                max-width: 500px;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }
            .dataframe tbody td:hover {
                overflow: auto;
                text-overflow: clip;
                white-space: pre-wrap;
                max-height: 200px;
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.dataframe(preview_df, use_container_width=True)

            if st.button("🚀 Run Priority Classification"):
                with st.spinner("Classifying emails..."):
                    try:
                        df_pred = predict_urgency(df)

                        st.subheader("Classification Results")
                        # Show classification results with index starting from 1
                        results_df = df_pred[["subject", "body", "priority_predicted"]].head(50).copy()
                        results_df.index = range(1, len(results_df) + 1)
                        # Use st.dataframe with use_container_width=True to match the preview width
                        st.dataframe(results_df, use_container_width=True)

                        # Summary counts
                        st.subheader("Priority Distribution")
                        counts = df_pred["priority_predicted"].value_counts().reset_index()
                        counts.columns = ["urgency", "count"]
                        # Add some vertical spacing and use container width for the chart
                        st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)
                        st.bar_chart(data=counts, x="urgency", y="count", use_container_width=True)

                        # Download button
                        csv_buffer = io.StringIO()
                        df_pred.to_csv(csv_buffer, index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv_buffer.getvalue(),
                            file_name="email_priority_predictions.csv",
                            mime="text/csv",
                        )
                    except Exception as e:
                        st.error(f"Error during processing: {e}")
                        
        except Exception as e:
            error_msg = f"Error loading sample dataset: {e}" if dataset_option != "None" else f"Error processing uploaded file: {e}"
            st.sidebar.error(error_msg)
                        
        except Exception as e:
            st.sidebar.error(f"Error processing uploaded file: {e}")
    
    # Show initial instructions if no file is selected or uploaded
    if df is None:
        st.info("Please select a sample dataset or upload a file to begin.")


if __name__ == "__main__":
    st.set_page_config(
        page_title="Email Classification",
        page_icon="📧",
        layout="wide"
    )
    main()
