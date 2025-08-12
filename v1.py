# Tamil Metaphor Classifier Pro - Enhanced Version
# All imports first
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import re
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import io
import time
from datetime import datetime

# Page configuration
# st.set_page_config(
#     page_title="Tamil Metaphor Classifier Pro",
#     page_icon="🎭",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# Clean Light Theme CSS without animations
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(to bottom, #f8fafc, #e2e8f0);
        color: #2d3748;
    }
    
    .main-header {
        background: linear-gradient(135deg, #e6f3ff 0%, #f0f8ff 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        color: #2c5282;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(44, 82, 130, 0.1);
        border: 2px solid #bee3f8;
    }
    
    .main-header h1 {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        font-size: 3.5rem;
        margin-bottom: 1rem;
        color: #2c5282;
    }
    
    .main-header p {
        font-family: 'Inter', sans-serif;
        font-weight: 400;
        font-size: 1.3rem;
        color: #4a5568;
    }
    
    /* Light Cards */
    .metric-card {
        background: linear-gradient(135deg, #ffffff, #f7fafc);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 2px 15px rgba(0,0,0,0.05);
        border: 2px solid #e2e8f0;
        margin-bottom: 1.5rem;
        position: relative;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #4299e1, #3182ce);
        border-radius: 15px 15px 0 0;
    }
    
    .stat-number {
        font-family: 'Poppins', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        color: #2c5282;
        margin: 0;
    }
    
    .stat-label {
        font-family: 'Inter', sans-serif;
        color: #4a5568;
        font-size: 0.95rem;
        font-weight: 600;
        margin: 0.5rem 0 0 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Input Card */
    .input-card {
        background: linear-gradient(135deg, #ffffff, #f7fafc);
        border-radius: 20px;
        padding: 2.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        border: 2px solid #e2e8f0;
        margin-bottom: 2rem;
    }
    
    .input-card h2 {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        color: #2c5282;
        margin-bottom: 2rem;
        font-size: 2rem;
    }
    
    /* Enhanced Text Areas and Inputs */
    .stTextArea textarea {
        border-radius: 12px !important;
        border: 2px solid #cbd5e0 !important;
        padding: 15px !important;
        font-size: 16px !important;
        font-family: 'Inter', sans-serif !important;
        background: #ffffff !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #4299e1 !important;
        box-shadow: 0 0 0 3px rgba(66, 153, 225, 0.1) !important;
    }
    
    /* File Uploader */
    .file-uploader {
        border: 3px dashed #a0aec0;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: linear-gradient(135deg, #f7fafc, #edf2f7);
    }
    
    .file-uploader:hover {
        border-color: #4299e1;
        background: linear-gradient(135deg, #e6f3ff, #bee3f8);
    }
    
    /* Enhanced Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.8rem 2rem !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        letter-spacing: 0.5px !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 15px rgba(66, 153, 225, 0.2) !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #3182ce 0%, #2c5282 100%) !important;
        box-shadow: 0 6px 20px rgba(66, 153, 225, 0.3) !important;
    }
    
    /* Info Card */
    .info-card {
        background: linear-gradient(135deg, #f0f8ff, #e6f3ff);
        border-radius: 15px;
        padding: 2rem;
        border: 2px solid #bee3f8;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.03);
    }
    
    .info-card h4 {
        font-family: 'Poppins', sans-serif;
        color: #2c5282;
        margin-bottom: 1.5rem;
        font-weight: 600;
        font-size: 1.3rem;
    }
    
    .info-item {
        display: flex;
        align-items: flex-start;
        margin-bottom: 1rem;
        padding: 0.5rem;
        border-radius: 8px;
    }
    
    .info-item:hover {
        background: rgba(66, 153, 225, 0.05);
    }
    
    .info-icon {
        color: #4299e1;
        font-size: 1.4rem;
        margin-right: 1rem;
        min-width: 2rem;
        text-align: center;
    }
    
    /* Light Badges */
    .metaphor-badge {
        background: linear-gradient(135deg, #fed7d7, #fc8181);
        color: #742a2a;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        border: 1px solid #feb2b2;
    }
    
    .literal-badge {
        background: linear-gradient(135deg, #c6f6d5, #68d391);
        color: #22543d;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        border: 1px solid #9ae6b4;
    }
    
    /* Result Cards */
    .result-card {
        background: linear-gradient(135deg, #ffffff, #f7fafc);
        border-radius: 15px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
        border: 1px solid #e2e8f0;
    }
    
    /* Enhanced Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: linear-gradient(135deg, #f7fafc, #edf2f7);
        border-radius: 12px;
        padding: 8px;
        border: 1px solid #e2e8f0;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background: #ffffff;
        border-radius: 8px;
        border: 1px solid #cbd5e0;
        padding: 12px 20px;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        color: #4a5568;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #f0f8ff;
        border-color: #4299e1;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4299e1, #3182ce) !important;
        color: white !important;
        border-color: #3182ce !important;
    }
    
    /* Enhanced Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #f7fafc, #edf2f7);
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #4299e1, #3182ce);
        border-radius: 8px;
    }
    
    /* Success Messages */
    .success-message {
        background: linear-gradient(135deg, #f0fff4, #c6f6d5);
        border: 2px solid #9ae6b4;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #22543d;
    }
    
    /* Light Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #cbd5e0, #a0aec0);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #a0aec0, #718096);
    }
    
    /* Light Warning and Error Messages */
    .stAlert {
        background: #ffffff !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 12px !important;
        color: #2d3748 !important;
    }
    
    /* Light DataFrames */
    .stDataFrame {
        background: #ffffff;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }
    
    /* Light Metrics */
    .stMetric {
        background: linear-gradient(135deg, #ffffff, #f7fafc);
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #e2e8f0;
    }
    
    /* Light Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f7fafc, #edf2f7) !important;
        border-radius: 8px !important;
        border: 1px solid #e2e8f0 !important;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2.5rem;
        }
        
        .input-card {
            padding: 1.5rem;
        }
        
        .metric-card {
            padding: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Function definitions
@st.cache_resource
def load_model():
    """Mock model loading for UI testing"""
    try:
        # Mock tokenizer and model for faster loading
        class MockTokenizer:
            def __call__(self, texts, return_tensors, truncation, padding, max_length):
                return {"input_ids": [[0] * max_length] * len(texts)}

        class MockModel:
            def eval(self):
                pass

            def __call__(self, **kwargs):
                class MockOutput:
                    logits = torch.tensor([[0.1, 0.9]] * len(kwargs["input_ids"]))
                return MockOutput()

        tokenizer = MockTokenizer()
        model = MockModel()
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading mock model: {str(e)}")
        return None, None

def split_sentences(text):
    """Enhanced sentence splitting with better punctuation handling"""
    # Handle Tamil punctuation and common sentence endings
    sentences = re.split(r'[.!?।॥]\s*|[\n]+', text)
    # Clean and filter sentences
    cleaned_sentences = []
    for s in sentences:
        s = s.strip()
        if s and len(s) > 3:  # Filter out very short fragments
            cleaned_sentences.append(s)
    return cleaned_sentences

def predict_metaphor_batch(texts, tokenizer, model):
    """Batch prediction with error handling"""
    if not texts or not tokenizer or not model:
        return [], []
    
    try:
        inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        predictions = np.argmax(probs, axis=-1)
        confidences = np.max(probs, axis=-1)
        return predictions.tolist(), confidences.tolist()  # Convert to lists
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return [], []

def calculate_advanced_statistics(df):
    """Calculate comprehensive statistics"""
    if df.empty:
        return {}
    
    stats_dict = {}
    
    # Basic counts
    total_sentences = len(df)
    metaphor_count = sum(df["Prediction"] == "Metaphor")
    literal_count = sum(df["Prediction"] == "Literal")
    
    # Proportions
    metaphor_ratio = metaphor_count / total_sentences if total_sentences > 0 else 0
    literal_ratio = literal_count / total_sentences if total_sentences > 0 else 0
    
    # Confidence statistics
    overall_confidence = df["Confidence"].mean()
    confidence_std = df["Confidence"].std()
    high_confidence_threshold = 0.8
    high_confidence_count = sum(df["Confidence"] >= high_confidence_threshold)
    
    # Length statistics
    avg_length = df["Length"].mean()
    length_std = df["Length"].std()
    median_length = df["Length"].median()
    
    # Class-specific statistics
    metaphor_df = df[df["Prediction"] == "Metaphor"]
    literal_df = df[df["Prediction"] == "Literal"]
    
    metaphor_avg_conf = metaphor_df["Confidence"].mean() if not metaphor_df.empty else 0
    literal_avg_conf = literal_df["Confidence"].mean() if not literal_df.empty else 0
    
    metaphor_avg_length = metaphor_df["Length"].mean() if not metaphor_df.empty else 0
    literal_avg_length = literal_df["Length"].mean() if not literal_df.empty else 0
    
    # Statistical tests
    if not metaphor_df.empty and not literal_df.empty:
        # T-test for confidence differences
        conf_tstat, conf_pvalue = stats.ttest_ind(metaphor_df["Confidence"], literal_df["Confidence"])
        # T-test for length differences
        length_tstat, length_pvalue = stats.ttest_ind(metaphor_df["Length"], literal_df["Length"])
    else:
        conf_tstat = conf_pvalue = length_tstat = length_pvalue = 0
    
    # Complexity metrics
    unique_words = len(set(' '.join(df["Sentence"]).lower().split()))
    avg_words_per_sentence = df["Length"].mean()
    vocabulary_richness = unique_words / sum(df["Length"]) if sum(df["Length"]) > 0 else 0
    
    return {
        'total_sentences': total_sentences,
        'metaphor_count': metaphor_count,
        'literal_count': literal_count,
        'metaphor_ratio': metaphor_ratio,
        'literal_ratio': literal_ratio,
        'overall_confidence': overall_confidence,
        'confidence_std': confidence_std,
        'high_confidence_count': high_confidence_count,
        'avg_length': avg_length,
        'length_std': length_std,
        'median_length': median_length,
        'metaphor_avg_conf': metaphor_avg_conf,
        'literal_avg_conf': literal_avg_conf,
        'metaphor_avg_length': metaphor_avg_length,
        'literal_avg_length': literal_avg_length,
        'conf_tstat': conf_tstat,
        'conf_pvalue': conf_pvalue,
        'length_tstat': length_tstat,
        'length_pvalue': length_pvalue,
        'unique_words': unique_words,
        'vocabulary_richness': vocabulary_richness,
        'avg_words_per_sentence': avg_words_per_sentence
    }

def create_advanced_visualizations(df, stats_dict):
    """Create comprehensive visualizations with error handling"""
    figures = {}
    
    if df.empty:
        return figures
    
    try:
        # 1. Enhanced Donut Chart
        if stats_dict['metaphor_count'] > 0 or stats_dict['literal_count'] > 0:
            fig_donut = go.Figure(data=[go.Pie(
                labels=["Metaphor", "Literal"],
                values=[stats_dict['metaphor_count'], stats_dict['literal_count']],
                hole=0.6,
                marker=dict(colors=['#ff6b6b', '#26de81'], line=dict(color='white', width=3)),
                textinfo='label+percent+value',
                textfont=dict(size=14, color='white'),
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            )])
            
            fig_donut.add_annotation(
                text=f"<b>{stats_dict['total_sentences']}</b><br>Total<br>Sentences",
                x=0.5, y=0.5,
                font_size=16,
                showarrow=False
            )
            
            fig_donut.update_layout(
                title="<b>Classification Distribution</b>",
                title_x=0.5,
                font=dict(family="Arial", size=14),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
                margin=dict(t=80, b=60, l=40, r=40),
                height=400
            )
            figures['donut'] = fig_donut
        
        # 2. Confidence Distribution Violin Plot
        if len(df) > 1 and df["Confidence"].std() > 0:
            fig_violin = go.Figure()
            
            for prediction, color in [("Metaphor", "#ff6b6b"), ("Literal", "#26de81")]:
                subset = df[df["Prediction"] == prediction]
                if not subset.empty and len(subset) > 1:
                    fig_violin.add_trace(go.Violin(
                        y=subset["Confidence"],
                        name=prediction,
                        box_visible=True,
                        meanline_visible=True,
                        fillcolor=color,
                        line_color=color,
                        opacity=0.7
                    ))
            
            if len(fig_violin.data) > 0:
                fig_violin.update_layout(
                    title="<b>Confidence Distribution by Class</b>",
                    title_x=0.5,
                    yaxis_title="Confidence Score",
                    xaxis_title="Prediction Class",
                    font=dict(family="Arial", size=12),
                    height=400,
                    margin=dict(t=80, b=60, l=60, r=40)
                )
                figures['violin'] = fig_violin

        # 3. Length vs Confidence Scatter Plot
        fig_scatter = px.scatter(
            df, x="Length", y="Confidence", color="Prediction",
            color_discrete_map={"Metaphor": "#ff6b6b", "Literal": "#26de81"},
            size="Confidence", hover_data=["Sentence"],
            title="<b>Sentence Length vs Confidence</b>",
            labels={"Length": "Sentence Length (words)", "Confidence": "Confidence Score"}
        )
        
        fig_scatter.update_layout(
            title_x=0.5,
            font=dict(family="Arial", size=12),
            height=400,
            margin=dict(t=80, b=60, l=60, r=40)
        )
        figures['scatter'] = fig_scatter
        
        # 4. Confidence Histogram with Statistical Overlay
        fig_hist = go.Figure()
        
        for prediction, color in [("Metaphor", "#ff6b6b"), ("Literal", "#26de81")]:
            subset = df[df["Prediction"] == prediction]
            if not subset.empty:
                fig_hist.add_trace(go.Histogram(
                    x=subset["Confidence"],
                    name=prediction,
                    opacity=0.7,
                    nbinsx=25,
                    marker_color=color,
                    histnorm='probability density'
                ))
        
        fig_hist.update_layout(
            title="<b>Confidence Distribution Density</b>",
            title_x=0.5,
            xaxis_title="Confidence Score",
            yaxis_title="Density",
            barmode='overlay',
            font=dict(family="Arial", size=12),
            height=400,
            margin=dict(t=80, b=60, l=60, r=40)
        )
        figures['histogram'] = fig_hist
        
        # 5. Word Frequency Analysis
        all_words = []
        for sentence in df["Sentence"]:
            words = re.findall(r'\w+', sentence.lower())
            all_words.extend(words)
        
        if all_words:
            word_freq = Counter(all_words)
            top_words = word_freq.most_common(20)
            
            if top_words:
                words_df = pd.DataFrame(top_words, columns=["Word", "Frequency"])
                
                fig_words = px.bar(
                    words_df, x="Frequency", y="Word",
                    orientation='h',
                    title="<b>Top 20 Most Frequent Words</b>",
                    color="Frequency",
                    color_continuous_scale="viridis"
                )
                
                fig_words.update_layout(
                    title_x=0.5,
                    yaxis={'categoryorder': 'total ascending'},
                    font=dict(family="Arial", size=12),
                    height=500,
                    margin=dict(t=80, b=60, l=100, r=40)
                )
                figures['words'] = fig_words
        
        # 6. Statistical Summary Heatmap
        if stats_dict['metaphor_count'] > 0 and stats_dict['literal_count'] > 0:
            summary_data = [
                ['Average Confidence', stats_dict['metaphor_avg_conf'], stats_dict['literal_avg_conf']],
                ['Average Length', stats_dict['metaphor_avg_length'], stats_dict['literal_avg_length']],
                ['Count', stats_dict['metaphor_count'], stats_dict['literal_count']]
            ]
            
            summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Metaphor', 'Literal'])
            summary_matrix = summary_df.set_index('Metric')[['Metaphor', 'Literal']].values
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=summary_matrix,
                x=['Metaphor', 'Literal'],
                y=['Avg Confidence', 'Avg Length', 'Count'],
                colorscale='RdYlBu_r',
                text=summary_matrix,
                texttemplate="%{text:.2f}",
                textfont={"size": 12},
                hovertemplate='<b>%{y}</b><br>%{x}: %{z:.2f}<extra></extra>'
            ))
            
            fig_heatmap.update_layout(
                title="<b>Class Comparison Heatmap</b>",
                title_x=0.5,
                font=dict(family="Arial", size=12),
                height=300,
                margin=dict(t=80, b=40, l=120, r=40)
            )
            figures['heatmap'] = fig_heatmap

        # 7. NEW: Confidence Ranges Distribution
        df_copy = df.copy()
        df_copy['Confidence_Range'] = pd.cut(df_copy['Confidence'], 
                                           bins=[0, 0.5, 0.7, 0.85, 1.0], 
                                           labels=['Low (0-0.5)', 'Medium (0.5-0.7)', 'High (0.7-0.85)', 'Very High (0.85-1.0)'])
        
        confidence_counts = df_copy.groupby(['Prediction', 'Confidence_Range']).size().reset_index(name='Count')
        
        if not confidence_counts.empty:
            fig_confidence_ranges = px.bar(
                confidence_counts, x='Confidence_Range', y='Count', color='Prediction',
                color_discrete_map={"Metaphor": "#ff6b6b", "Literal": "#26de81"},
                title="<b>Confidence Ranges Distribution</b>",
                labels={'Confidence_Range': 'Confidence Range', 'Count': 'Number of Sentences'}
            )
            
            fig_confidence_ranges.update_layout(
                title_x=0.5,
                font=dict(family="Arial", size=12),
                height=400,
                margin=dict(t=80, b=60, l=60, r=40)
            )
            figures['confidence_ranges'] = fig_confidence_ranges

        # 8. NEW: Sentence Length Distribution
        fig_length_dist = go.Figure()
        
        for prediction, color in [("Metaphor", "#ff6b6b"), ("Literal", "#26de81")]:
            subset = df[df["Prediction"] == prediction]
            if not subset.empty:
                fig_length_dist.add_trace(go.Histogram(
                    x=subset["Length"],
                    name=prediction,
                    opacity=0.7,
                    marker_color=color,
                    nbinsx=15
                ))
        
        fig_length_dist.update_layout(
            title="<b>Sentence Length Distribution</b>",
            title_x=0.5,
            xaxis_title="Sentence Length (words)",
            yaxis_title="Count",
            barmode='overlay',
            font=dict(family="Arial", size=12),
            height=400,
            margin=dict(t=80, b=60, l=60, r=40)
        )
        figures['length_distribution'] = fig_length_dist

        # 9. NEW: Confidence vs Length Box Plot
        if len(df) > 5:
            df_copy = df.copy()
            df_copy['Length_Category'] = pd.cut(df_copy['Length'], 
                                              bins=3, 
                                              labels=['Short', 'Medium', 'Long'])
            
            fig_box = px.box(
                df_copy, x='Length_Category', y='Confidence', color='Prediction',
                color_discrete_map={"Metaphor": "#ff6b6b", "Literal": "#26de81"},
                title="<b>Confidence by Sentence Length Category</b>",
                labels={'Length_Category': 'Sentence Length Category', 'Confidence': 'Confidence Score'}
            )
            
            fig_box.update_layout(
                title_x=0.5,
                font=dict(family="Arial", size=12),
                height=400,
                margin=dict(t=80, b=60, l=60, r=40)
            )
            figures['confidence_by_length'] = fig_box

        # 10. NEW: Word Density Analysis
        if df['Word_Density'].std() > 0:
            fig_density = px.scatter(
                df, x="Word_Density", y="Confidence", color="Prediction",
                color_discrete_map={"Metaphor": "#ff6b6b", "Literal": "#26de81"},
                title="<b>Word Density vs Confidence</b>",
                labels={'Word_Density': 'Word Density (words/character)', 'Confidence': 'Confidence Score'},
                hover_data=["Length", "Sentence"]
            )
            
            fig_density.update_layout(
                title_x=0.5,
                font=dict(family="Arial", size=12),
                height=400,
                margin=dict(t=80, b=60, l=60, r=40)
            )
            figures['word_density'] = fig_density

        # 11. NEW: Character Count vs Confidence
        fig_char_conf = px.scatter(
            df, x="Character_Count", y="Confidence", color="Prediction",
            color_discrete_map={"Metaphor": "#ff6b6b", "Literal": "#26de81"},
            title="<b>Character Count vs Confidence</b>",
            labels={'Character_Count': 'Character Count', 'Confidence': 'Confidence Score'},
            hover_data=["Length", "Sentence"]
        )
        
        fig_char_conf.update_layout(
            title_x=0.5,
            font=dict(family="Arial", size=12),
            height=400,
            margin=dict(t=80, b=60, l=60, r=40)
        )
        figures['char_confidence'] = fig_char_conf

        # 12. NEW: Prediction Confidence Trend (if sentences have order)
        df_indexed = df.reset_index()
        fig_trend = px.line(
            df_indexed, x='index', y='Confidence', color='Prediction',
            color_discrete_map={"Metaphor": "#ff6b6b", "Literal": "#26de81"},
            title="<b>Confidence Trend Across Sentences</b>",
            labels={'index': 'Sentence Order', 'Confidence': 'Confidence Score'}
        )
        
        fig_trend.update_layout(
            title_x=0.5,
            font=dict(family="Arial", size=12),
            height=400,
            margin=dict(t=80, b=60, l=60, r=40)
        )
        figures['confidence_trend'] = fig_trend

        # 13. NEW: Advanced Statistical Summary
        if stats_dict['metaphor_count'] > 0 and stats_dict['literal_count'] > 0:
            # Create a comprehensive comparison chart
            metrics = ['Count', 'Avg Confidence', 'Avg Length', 'Avg Char Count', 'Avg Word Density']
            metaphor_values = [
                stats_dict['metaphor_count'],
                stats_dict['metaphor_avg_conf'],
                stats_dict['metaphor_avg_length'],
                df[df["Prediction"] == "Metaphor"]["Character_Count"].mean() if not df[df["Prediction"] == "Metaphor"].empty else 0,
                df[df["Prediction"] == "Metaphor"]["Word_Density"].mean() if not df[df["Prediction"] == "Metaphor"].empty else 0
            ]
            literal_values = [
                stats_dict['literal_count'],
                stats_dict['literal_avg_conf'],
                stats_dict['literal_avg_length'],
                df[df["Prediction"] == "Literal"]["Character_Count"].mean() if not df[df["Prediction"] == "Literal"].empty else 0,
                df[df["Prediction"] == "Literal"]["Word_Density"].mean() if not df[df["Prediction"] == "Literal"].empty else 0
            ]
            
            fig_comparison = go.Figure()
            
            fig_comparison.add_trace(go.Bar(
                name='Metaphor',
                x=metrics,
                y=metaphor_values,
                marker_color='#ff6b6b',
                text=[f'{v:.2f}' for v in metaphor_values],
                textposition='auto'
            ))
            
            fig_comparison.add_trace(go.Bar(
                name='Literal',
                x=metrics,
                y=literal_values,
                marker_color='#26de81',
                text=[f'{v:.2f}' for v in literal_values],
                textposition='auto'
            ))
            
            fig_comparison.update_layout(
                title='<b>Comprehensive Comparison: Metaphor vs Literal</b>',
                title_x=0.5,
                xaxis_title='Metrics',
                yaxis_title='Values',
                barmode='group',
                font=dict(family="Arial", size=12),
                height=450,
                margin=dict(t=80, b=60, l=60, r=40)
            )
            figures['comprehensive_comparison'] = fig_comparison

        # 14. NEW: Correlation Matrix
        numerical_cols = ['Confidence', 'Length', 'Character_Count', 'Word_Density']
        correlation_matrix = df[numerical_cols].corr()
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            text=correlation_matrix.round(3).values,
            texttemplate="%{text}",
            textfont={"size": 12},
            hovertemplate='<b>%{y}</b> vs <b>%{x}</b><br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig_corr.update_layout(
            title="<b>Feature Correlation Matrix</b>",
            title_x=0.5,
            font=dict(family="Arial", size=12),
            height=400,
            margin=dict(t=80, b=60, l=80, r=40)
        )
        figures['correlation_matrix'] = fig_corr
            
    except Exception as e:
        st.error(f"Error creating visualizations: {str(e)}")
    
    return figures

# Main Application
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎭 Tamil Metaphor Classifier Pro</h1>
        <p style="font-size: 1.2rem; margin-bottom: 0;">
            Advanced AI-powered metaphor detection for Tamil text with comprehensive analytics
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    with st.spinner("🔄 Loading AI model..."):
        tokenizer, model = load_model()
    
    if tokenizer is None or model is None:
        st.error("❌ Failed to load the model. Please check the model path.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Initialize session state for persistent data
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'original_df' not in st.session_state:
        st.session_state.original_df = None
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Analysis Controls")

        # --- Show filters only if we have results ---
        if st.session_state.get("analysis_results") is not None:

            st.markdown("### 🔍 Real-time Filters")
            st.info("💡 Filters apply instantly to your results!")

            # ---- Initialize session state only once ----
            if "filter_classes" not in st.session_state:
                st.session_state.filter_classes = ["Metaphor", "Literal"]

            if "confidence_threshold" not in st.session_state:
                st.session_state.confidence_threshold = 0.0

            if "max_confidence" not in st.session_state:
                st.session_state.max_confidence = 1.0

            if "min_length" not in st.session_state:
                st.session_state.min_length = 1

            if "max_length" not in st.session_state:
                st.session_state.max_length = 100

            if "search_term" not in st.session_state:
                st.session_state.search_term = ""

            # --- Get original DF stats ---
            if st.session_state.get("original_df") is not None:
                df = st.session_state.original_df
                min_conf_value = float(df["Confidence"].min())
                max_conf_value = float(df["Confidence"].max())

                if max_conf_value <= min_conf_value:
                    max_conf_value = min_conf_value + 0.01

                min_length_value = int(df["Length"].min())
                max_length_value = int(df["Length"].max())

                if max_length_value <= min_length_value:
                    max_length_value = min_length_value + 1
            else:
                min_conf_value, max_conf_value = 0.0, 1.0
                min_length_value, max_length_value = 1, 100

            # --- Widgets ---
            st.session_state.filter_classes = st.multiselect(
                "Show classifications:",
                options=["Metaphor", "Literal"],
                default=st.session_state.filter_classes,  # session-driven default
                key="filter_classes_widget",  # different key for widget instance
                help="Select which types of classifications to display"
            )

            st.session_state.confidence_threshold = st.slider(
                "Minimum confidence:",
                min_value=0.0,
                max_value=1.0,
                value=max(st.session_state.confidence_threshold, min_conf_value),
                step=0.01,
                key="confidence_threshold_widget",
                help="Filter results by minimum confidence score"
            )

            st.session_state.max_confidence = st.slider(
                "Maximum confidence:",
                min_value=0.0,
                max_value=1.0,
                value=min(st.session_state.max_confidence, max_conf_value),
                step=0.01,
                key="max_confidence_widget",
                help="Filter results by maximum confidence score"
            )

            st.session_state.min_length = st.number_input(
                "Minimum sentence length (words):",
                min_value=1,
                max_value=max(100, max_length_value),
                value=max(st.session_state.min_length, min_length_value),
                key="min_length_widget",
                help="Filter by minimum number of words in sentence"
            )

            st.session_state.max_length = st.number_input(
                "Maximum sentence length (words):",
                min_value=max(1, min_length_value),
                max_value=max(100, max_length_value + 10),
                value=max(st.session_state.max_length, max_length_value),
                key="max_length_widget",
                help="Filter by maximum number of words in sentence"
            )

            st.session_state.search_term = st.text_input(
                "🔍 Search in sentences:",
                placeholder="Enter keyword to search...",
                value=st.session_state.search_term,
                key="search_term_widget",
                help="Search for specific words or phrases in sentences"
            )

            # --- Reset Filters ---
            if st.button("🔄 Reset All Filters", help="Clear all filters"):
                st.session_state.update({
                    "filter_classes": ["Metaphor", "Literal"],
                    "confidence_threshold": 0.0,
                    "max_confidence": 1.0,
                    "min_length": 1,
                    "max_length": 100,
                    "search_term": ""
                })
                st.rerun()

            # --- Filter Summary ---
            total_results = len(st.session_state.original_df) if st.session_state.get("original_df") is not None else 0

            try:
                temp_df = df.copy()
                if st.session_state.filter_classes:
                    temp_df = temp_df[temp_df["Prediction"].isin(st.session_state.filter_classes)]

                temp_df = temp_df[
                    (temp_df["Confidence"] >= st.session_state.confidence_threshold) &
                    (temp_df["Confidence"] <= st.session_state.max_confidence) &
                    (temp_df["Length"] >= st.session_state.min_length) &
                    (temp_df["Length"] <= st.session_state.max_length)
                ]

                if st.session_state.search_term.strip():
                    temp_df = temp_df[
                        temp_df["Sentence"].str.contains(st.session_state.search_term.strip(), case=False, regex=False, na=False)
                    ]

                filtered_results = len(temp_df)
            except:
                filtered_results = 0

            st.markdown(f"**Total sentences:** {total_results}")
            st.markdown(f"**Filtered results:** {filtered_results}")

        else:
            st.markdown("### 📋 Instructions")
            st.info("Enter text and click 'Analyze' to see filtering options here!")

        st.markdown("---")
        st.markdown("### 📊 Export Options")
        export_format = st.selectbox(
            "Export format:",
            ["CSV", "JSON"],
            help="Choose format for downloading results"
        )

    
    # Main input area - Enhanced
    st.markdown("""
    <div class="input-card">
        <h2 style="margin-bottom: 1.5rem; color: #464646;">📝 Text Input</h2>
    """, unsafe_allow_html=True)
    
    # Create tabs for different input methods
    input_method = st.radio(
        "Choose input method:",
        ["Text Input", "File Upload", "Sample Examples"],
        horizontal=True,
        key="input_method",
        help="Select how you want to provide text for analysis"
    )
    
    if input_method == "Text Input":
        user_input = st.text_area(
            "Enter Tamil text for analysis:",
            height=220,
            placeholder="பாடல் வரிகள் அல்லது உரையை இங்கே உள்ளிடவும்...",
            help="Enter Tamil song lyrics, poetry, or any text for metaphor analysis"
        )
        
    elif input_method == "File Upload":
        st.markdown("""
        <div class="file-uploader">
            <p style="color: #667eea; font-size: 1.1rem; font-weight: 600; margin-bottom: 0.8rem;">📄 Upload a Text File</p>
            <p style="color: #64748b; font-size: 0.9rem; margin-bottom: 1rem;">Supported formats: .txt, .csv, .doc</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("", type=["txt", "csv", "doc"])
        
        if uploaded_file is not None:
            try:
                # Read file contents based on type
                if uploaded_file.type == "text/plain":
                    user_input = uploaded_file.getvalue().decode("utf-8")
                elif uploaded_file.type == "text/csv":
                    df = pd.read_csv(uploaded_file)
                    if len(df.columns) == 1:
                        user_input = " ".join(df.iloc[:, 0].astype(str).tolist())
                    else:
                        user_input = " ".join(df.iloc[:, 0].astype(str).tolist())
                        st.info("Using only the first column of the CSV file.")
                else:
                    user_input = "File format not fully supported. Please paste content directly."
                    
                st.success(f"File '{uploaded_file.name}' loaded successfully!")
                st.text_area("File Content Preview:", value=user_input[:500] + "..." if len(user_input) > 500 else user_input, height=150)
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                user_input = ""
        else:
            user_input = ""
    
    else:  # Sample Examples
        st.markdown("""
        <p style="color: #64748b; margin-bottom: 1rem;">Select an example to analyze:</p>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        example_texts = {
            "Tamil Poetry": "வானத்தில் நிலவு பால் போல ஒளிர்கிறது.\nகாற்று என்னை வருடிச் செல்கிறது.\nமனித வாழ்க்கை ஒரு நதி போன்றது.\nசிந்தனைகள் பறவைகள் போல பறக்கின்றன.\nஅவள் கண்கள் நட்சத்திரங்கள்.",
            "Tamil Song": "உன் கண்ணில் நான் விழுந்தேன்\nஉன் காதல் கடலில் மூழ்கினேன்\nநீ என் வாழ்வின் வானம்\nஉன் கரங்கள் என் வாழ்வின் கோட்டை",
            "Tamil Proverbs": "அகத்தின் அழகு முகத்தில் தெரியும்.\nஅடி மழையில் தவளை நடனமாடும்.\nஆழம் அறியாமல் காலை விடாதே.\nஅன்புக்கு அடிமையானவன் பாக்கியசாலி."
        }
        
        selected_example = None
        
        with col1:
            if st.button("Tamil Poetry", key="example1", use_container_width=True):
                selected_example = "Tamil Poetry"
        
        with col2:
            if st.button("Tamil Song", key="example2", use_container_width=True):
                selected_example = "Tamil Song"
                
        with col3:
            if st.button("Tamil Proverbs", key="example3", use_container_width=True):
                selected_example = "Tamil Proverbs"
                
        if selected_example:
            user_input = example_texts[selected_example]
            st.text_area("Selected Example:", value=user_input, height=150)
        else:
            user_input = ""
    
    # Info section with better design
    st.markdown("""
    <div class="info-card">
        <h4>💡 Pro Tips</h4>
        <div class="info-item">
            <div class="info-icon">📏</div>
            <div>Enter complete sentences for better results</div>
        </div>
        <div class="info-item">
            <div class="info-icon">🔄</div>
            <div>Mix of metaphorical and literal sentences works best</div>
        </div>
        <div class="info-item">
            <div class="info-icon">🔤</div>
            <div>The model works best with Tamil text</div>
        </div>
        <div class="info-item">
            <div class="info-icon">📊</div>
            <div>Longer texts provide richer analytics</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # End of input card div
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Analysis button with enhanced styling
    if st.button("🚀 Analyze Text", type="primary", use_container_width=True, key="analyze_button"):
        if not user_input.strip():
            st.warning("⚠️ Please enter some text to analyze.")
            return
            
        # Show a custom progress indicator
        progress_container = st.container()
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner(""):
            try:
                # Process text with visual feedback
                status_text.markdown("🔍 **Analyzing your text...**")
                progress_bar.progress(20)
                
                # Split sentences
                sentences = split_sentences(user_input)
                if not sentences:
                    st.error("❌ Could not extract sentences from the input. Please check your text.")
                    return
                
                status_text.markdown("🤖 **Applying AI model...**")
                progress_bar.progress(40)
                
                # Get predictions
                predictions, confidences = predict_metaphor_batch(sentences, tokenizer, model)
                if len(predictions) == 0 or len(confidences) == 0:
                    st.error("❌ Failed to analyze the text. Please try again.")
                    return
                
                status_text.markdown("📊 **Calculating statistics...**")
                progress_bar.progress(60)
                
                # Ensure we have valid data
                if len(predictions) != len(sentences) or len(confidences) != len(sentences):
                    st.error("❌ Data mismatch in analysis. Please try again.")
                    return
                
                status_text.markdown("📈 **Preparing visualizations...**")
                progress_bar.progress(80)
                
                # Create DataFrame with validation
                df_data = {
                    "Sentence": sentences,
                    "Prediction": ["Metaphor" if p == 1 else "Literal" for p in predictions],
                    "Confidence": confidences,
                    "Length": [len(s.split()) for s in sentences],
                    "Character_Count": [len(s) for s in sentences],
                    "Word_Density": [len(s.split())/len(s) if len(s) > 0 else 0 for s in sentences]
                }
                
                # Validate data before creating DataFrame
                for key, values in df_data.items():
                    if len(values) != len(sentences):
                        st.error(f"❌ Data validation failed for {key}. Please try again.")
                        return
                
                df = pd.DataFrame(df_data)
                
                # Additional validation
                if df.empty or df["Confidence"].isna().any():
                    st.error("❌ Invalid analysis results. Please try again.")
                    return
                
                status_text.markdown("✨ **Finalizing results...**")
                progress_bar.progress(100)
                
                # Store results in session state
                st.session_state.original_df = df.copy()
                st.session_state.analysis_results = True
                
                # Reset filters to safe defaults
                st.session_state.filter_classes = ["Metaphor", "Literal"]
                st.session_state.confidence_threshold = float(df["Confidence"].min())
                st.session_state.max_confidence = float(df["Confidence"].max())
                st.session_state.min_length = int(df["Length"].min())
                st.session_state.max_length = int(df["Length"].max())
                st.session_state.search_term = ""
                
                # Success message with animation
                status_text.markdown("""
                <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #f0fff4, #e6fffa); border-radius: 12px; margin: 1rem 0;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" fill="currentColor" class="bi bi-check2-circle" viewBox="0 0 16 16" style="color: #38b2ac; margin-bottom: 0.5rem;">
                        <path d="M2.5 8a5.5 5.5 0 0 1 8.25-4.764.5.5 0 0 0 .5-.866A6.5 6.5 0 1 0 14.5 8a.5.5 0 0 0-1 0 5.5 5.5 0 1 1-11 0z"/>
                        <path d="M15.354 3.354a.5.5 0 0 0-.708-.708L8 9.293 5.354 6.646a.5.5 0 1 0-.708.708l3 3a.5.5 0 0 0 .708 0l7-7z"/>
                    </svg>
                    <p style="font-size: 1.2rem; font-weight: bold; color: #2c7a7b; margin: 0;">Analysis Complete!</p>
                    <p style="color: #4a5568; margin-top: 0.5rem;">Use the sidebar filters to explore your results</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Automatically refresh to show results
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ An error occurred during analysis: {str(e)}")
                st.info("Please try again with different text or refresh the page.")
            
            finally:
                # Clean up progress indicators
                time.sleep(0.5)
                progress_bar.empty()

    # Display results if available
    if st.session_state.analysis_results is not None and st.session_state.original_df is not None:
        # Apply real-time filters
        filtered_df = st.session_state.original_df.copy()
        
        # Get current filter values from session state
        current_filter_classes = st.session_state.get('filter_classes', ["Metaphor", "Literal"])
        current_confidence_threshold = st.session_state.get('confidence_threshold', 0.0)
        current_max_confidence = st.session_state.get('max_confidence', 1.0)
        current_min_length = st.session_state.get('min_length', 1)
        current_max_length = st.session_state.get('max_length', 100)
        current_search_term = st.session_state.get('search_term', "")
        
        # Apply filters step by step
        if current_filter_classes:
            filtered_df = filtered_df[filtered_df["Prediction"].isin(current_filter_classes)]
        
        filtered_df = filtered_df[
            (filtered_df["Confidence"] >= current_confidence_threshold) &
            (filtered_df["Confidence"] <= current_max_confidence) &
            (filtered_df["Length"] >= current_min_length) &
            (filtered_df["Length"] <= current_max_length)
        ]
        
        if current_search_term.strip():
            filtered_df = filtered_df[
                filtered_df["Sentence"].str.contains(current_search_term.strip(), case=False, regex=False, na=False)
            ]
        # Show filter status
        total_sentences = len(st.session_state.original_df)
        filtered_count = len(filtered_df)
        
        if filtered_count < total_sentences:
            st.info(f"📊 Showing {filtered_count} of {total_sentences} sentences (filtered). Adjust filters in sidebar to see more results.")
        else:
            st.success(f"📊 Showing all {total_sentences} sentences.")
        
        if filtered_df.empty:
            st.warning("⚠️ No sentences match the current filters. Try adjusting your filter settings in the sidebar.")
            st.markdown("### 🔧 Suggested Actions:")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Lower confidence threshold", key="suggest_lower_conf"):
                    st.session_state.confidence_threshold = max(0.0, current_confidence_threshold - 0.1)
                    st.rerun()
            with col2:
                if st.button("Include all sentence types", key="suggest_all_types"):
                    st.session_state.filter_classes = ["Metaphor", "Literal"]
                    st.rerun()
            with col3:
                if st.button("Clear search term", key="suggest_clear_search"):
                    st.session_state.search_term = ""
                    st.rerun()
            return
        
        # Calculate statistics
        stats = calculate_advanced_statistics(filtered_df)
        
        # Display results
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        
        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <p class="stat-number">{stats['total_sentences']}</p>
                <p class="stat-label">Filtered Sentences</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <p class="stat-number">{stats['metaphor_count']}</p>
                <p class="stat-label">Metaphors ({stats['metaphor_ratio']:.1%})</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <p class="stat-number">{stats['literal_count']}</p>
                <p class="stat-label">Literal ({stats['literal_ratio']:.1%})</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <p class="stat_number">{stats['overall_confidence']:.3f}</p>
                <p class="stat-label">Avg Confidence</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown(f"""
            <div class="metric-card">
                <p class="stat-number">{stats['high_confidence_count']}</p>
                <p class="stat-label">High Confidence (>0.8)</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed Statistics
        with st.expander("📈 Detailed Statistical Analysis", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Descriptive Statistics")
                st.markdown(f"""
                **Text Complexity:**
                - Unique words: {stats['unique_words']}
                - Vocabulary richness: {stats['vocabulary_richness']:.3f}
                - Average sentence length: {stats['avg_length']:.1f} words
                - Median sentence length: {stats['median_length']:.1f} words
                - Length std deviation: {stats['length_std']:.2f}
                
                **Confidence Metrics:**
                - Overall confidence: {stats['overall_confidence']:.3f} ± {stats['confidence_std']:.3f}
                - High confidence sentences: {stats['high_confidence_count']} ({stats['high_confidence_count']/stats['total_sentences']:.1%})
                """)
            
            with col2:
                st.markdown("### 🔬 Comparative Analysis")
                st.markdown(f"""
                **Metaphor vs Literal:**
                - Avg confidence (Metaphor): {stats['metaphor_avg_conf']:.3f}
                - Avg confidence (Literal): {stats['literal_avg_conf']:.3f}
                - Avg length (Metaphor): {stats['metaphor_avg_length']:.1f} words
                - Avg length (Literal): {stats['literal_avg_length']:.1f} words
                
                **Statistical Tests:**
                - Confidence t-test p-value: {stats['conf_pvalue']:.4f}
                - Length t-test p-value: {stats['length_pvalue']:.4f}
                """)
                
                if stats['conf_pvalue'] < 0.05:
                    st.success("✅ Significant difference in confidence between classes")
                else:
                    st.info("ℹ️ No significant difference in confidence between classes")
                
                if stats['length_pvalue'] < 0.05:
                    st.success("✅ Significant difference in length between classes")
                else:
                    st.info("ℹ️ No significant difference in length between classes")
        
        # Display sentence results (moved the entire display logic here)
        # Sentence-level results with multiple display options
        st.markdown("### 📝 Sentence Analysis")
        
        # Display options
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            display_mode = st.selectbox(
                "Display Mode:",
                ["Paginated View", "Compact Table", "Side-by-Side", "Expandable Cards"],
                help="Choose how to display the analysis results"
            )
        
        with col2:
            items_per_page = st.selectbox(
                "Items per page:",
                [5, 10, 20, 50, 100],
                index=1,
                help="Number of sentences to show per page"
            )
        
        with col3:
            sort_by = st.selectbox(
                "Sort by:",
                ["Original Order", "Confidence (High to Low)", "Confidence (Low to High)", 
                 "Length (Long to Short)", "Length (Short to Long)", "Prediction Type"],
                help="Sort sentences by different criteria"
            )
        
        
        
        # Apply sorting
        display_df = filtered_df.copy().reset_index(drop=True)
        if sort_by == "Confidence (High to Low)":
            display_df = display_df.sort_values("Confidence", ascending=False).reset_index(drop=True)
        elif sort_by == "Confidence (Low to High)":
            display_df = display_df.sort_values("Confidence", ascending=True).reset_index(drop=True)
        elif sort_by == "Length (Long to Short)":
            display_df = display_df.sort_values("Length", ascending=False).reset_index(drop=True)
        elif sort_by == "Length (Short to Long)":
            display_df = display_df.sort_values("Length", ascending=True).reset_index(drop=True)
        elif sort_by == "Prediction Type":
            display_df = display_df.sort_values("Prediction", ascending=False).reset_index(drop=True)
        
        # Display based on selected mode (keeping all the existing display logic)
        if display_mode == "Compact Table":
            # Compact table view
            st.markdown("#### 📋 Compact Table View")
            
            # Create display dataframe with formatting
            table_df = display_df.copy()
            table_df["Confidence"] = table_df["Confidence"].apply(lambda x: f"{x:.3f}")
            table_df["Prediction"] = table_df["Prediction"].apply(
                lambda x: "🎭 Metaphor" if x == "Metaphor" else "📝 Literal"
            )
            table_df["Length"] = table_df["Length"].apply(lambda x: f"{x} words")
            
            # Rename columns for display
            table_df = table_df.rename(columns={
                "Sentence": "Text",
                "Prediction": "Type",
                "Confidence": "Conf.",
                "Length": "Words"
            })
            
            st.dataframe(
                table_df[["Type", "Text", "Conf.", "Words"]].head(items_per_page),
                use_container_width=True,
                height=min(600, len(table_df) * 35 + 100)
            )
            
        elif display_mode == "Side-by-Side":
            # Side-by-side comparison
            st.markdown("#### ⚖️ Side-by-Side Comparison")
            
            metaphor_df = display_df[display_df["Prediction"] == "Metaphor"].head(items_per_page//2)
            literal_df = display_df[display_df["Prediction"] == "Literal"].head(items_per_page//2)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### 🎭 Metaphors")
                if metaphor_df.empty:
                    st.info("No metaphors found with current filters")
                else:
                    for idx, row in metaphor_df.iterrows():
                        confidence_width = row["Confidence"] * 100
                        st.markdown(f"""
                        <div style="background: #fff5f5; border-left: 4px solid #ff6b6b; padding: 1rem; margin-bottom: 0.5rem; border-radius: 0 8px 8px 0;">
                            <div style="font-size: 0.9rem; margin-bottom: 0.5rem;">
                                <strong>Confidence:</strong> {row["Confidence"]:.3f}
                                | <strong>Length:</strong> {row['Length']} words
                            </div>
                            <div style="font-size: 1rem; line-height: 1.4;">{row["Sentence"]}</div>
                            <div style="background: #ff6b6b; height: 3px; width: {confidence_width}%; border-radius: 2px; margin-top: 0.5rem;"></div>
                        </div>
                        """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("##### 📝 Literal Statements")
                if literal_df.empty:
                    st.info("No literal statements found with current filters")
                else:
                    for idx, row in literal_df.iterrows():
                        confidence_width = row["Confidence"] * 100
                        st.markdown(f"""
                        <div style="background: #f5fff5; border-left: 4px solid #26de81; padding: 1rem; margin-bottom: 0.5rem; border-radius: 0 8px 8px 0;">
                            <div style="font-size: 0.9rem; margin-bottom: 0.5rem;">
                                <strong>Confidence:</strong> {row["Confidence"]:.3f}
                                | <strong>Length:</strong> {row['Length']} words
                            </div>
                            <div style="font-size: 1rem; line-height: 1.4;">{row["Sentence"]}</div>
                            <div style="background: #26de81; height: 3px; width: {confidence_width}%; border-radius: 2px; margin-top: 0.5rem;"></div>
                        </div>
                        """, unsafe_allow_html=True)
        
        elif display_mode == "Expandable Cards":
            # Expandable cards view
            st.markdown("#### 📑 Expandable Cards View")
            
            # Group by prediction type
            metaphor_sentences = display_df[display_df["Prediction"] == "Metaphor"]
            literal_sentences = display_df[display_df["Prediction"] == "Literal"]
            
            # Metaphor expandable section
            if not metaphor_sentences.empty:
                with st.expander(f"🎭 Metaphors ({len(metaphor_sentences)} sentences)", expanded=True):
                    for idx, row in metaphor_sentences.head(items_per_page//2).iterrows():
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.markdown(f"**{row['Sentence']}**")
                        with col2:
                            st.metric("Confidence", f"{row['Confidence']:.3f}")
                            st.caption(f"{row['Length']} words")
            
            # Literal expandable section
            if not literal_sentences.empty:
                with st.expander(f"📝 Literal Statements ({len(literal_sentences)} sentences)", expanded=True):
                    for idx, row in literal_sentences.head(items_per_page//2).iterrows():
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.markdown(f"**{row['Sentence']}**")
                        with col2:
                            st.metric("Confidence", f"{row['Confidence']:.3f}")
                            st.caption(f"{row['Length']} words")
        
        else:  # Paginated View (default)
            st.markdown("#### 📄 Paginated Results")
            
            # Pagination controls
            total_items = len(display_df)
            total_pages = (total_items - 1) // items_per_page + 1 if total_items > 0 else 1
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                current_page = st.select_slider(
                    "Page:",
                    options=list(range(1, total_pages + 1)),
                    value=1,
                    format_func=lambda x: f"Page {x} of {total_pages}"
                )
            
            # Calculate start and end indices
            start_idx = (current_page - 1) * items_per_page
            end_idx = min(start_idx + items_per_page, total_items)
            
            # Display current page items
            current_items = display_df.iloc[start_idx:end_idx]
            
            if current_items.empty:
                st.info("No sentences to display on this page.")
            else:
                for idx, row in current_items.iterrows():
                    prediction = row["Prediction"]
                    confidence = row["Confidence"]
                    sentence = row["Sentence"]
                    length = row["Length"]
                    
                    # Determine styling based on prediction
                    if prediction == "Metaphor":
                        badge_class = "metaphor-badge"
                        confidence_color = "#ff6b6b"
                        bg_color = "#fff5f5"
                    else:
                        badge_class = "literal-badge"
                        confidence_color = "#26de81"
                        bg_color = "#f5fff5"
                    
                    st.markdown(f"""
                    <div style="background: {bg_color}; border: 1px solid {confidence_color}; border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                            <span class="{badge_class}">{prediction}</span>
                            <div style="text-align: right; font-size: 0.9rem; color: #666;">
                                <div><strong>Confidence:</strong> {confidence:.3f}</div>
                                <div><strong>Length:</strong> {length} words</div>
                            </div>
                        </div>
                        <div style="font-size: 1.1rem; line-height: 1.5; margin-bottom: 0.5rem;">{sentence}</div>
                        <div style="background: #e9ecef; height: 6px; border-radius: 3px; overflow: hidden;">
                            <div style="background: {confidence_color}; height: 100%; width: {confidence*100}%; border-radius: 3px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Show pagination info
            st.markdown(f"""
            <div style="text-align: center; color: #666; margin-top: 1rem;">
                Showing {start_idx + 1}-{end_idx} of {total_items} sentences
            </div>
            """, unsafe_allow_html=True)

        # Visualizations
        st.markdown("---")
        st.markdown("## 📈 Advanced Visualizations")
        
        figures = create_advanced_visualizations(filtered_df, stats)
        
        # Layout visualizations in tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Distribution", 
            "📈 Analysis", 
            "🔤 Word Frequency", 
            "🔥 Comparison",
            "📏 Length & Density",
            "🔗 Correlations"
        ])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                if 'donut' in figures:
                    st.plotly_chart(figures['donut'], use_container_width=True)
            with col2:
                if 'confidence_ranges' in figures:
                    st.plotly_chart(figures['confidence_ranges'], use_container_width=True)
            
            if 'length_distribution' in figures:
                st.plotly_chart(figures['length_distribution'], use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                if 'scatter' in figures:
                    st.plotly_chart(figures['scatter'], use_container_width=True)
            with col2:
                if 'violin' in figures:
                    st.plotly_chart(figures['violin'], use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if 'histogram' in figures:
                    st.plotly_chart(figures['histogram'], use_container_width=True)
            with col2:
                if 'confidence_by_length' in figures:
                    st.plotly_chart(figures['confidence_by_length'], use_container_width=True)
        
        with tab3:
            if 'words' in figures:
                st.plotly_chart(figures['words'], use_container_width=True)
            
            if 'confidence_trend' in figures:
                st.plotly_chart(figures['confidence_trend'], use_container_width=True)
        
        with tab4:
            if 'heatmap' in figures:
                st.plotly_chart(figures['heatmap'], use_container_width=True)
            
            if 'comprehensive_comparison' in figures:
                st.plotly_chart(figures['comprehensive_comparison'], use_container_width=True)
        
        with tab5:
            col1, col2 = st.columns(2)
            with col1:
                if 'word_density' in figures:
                    st.plotly_chart(figures['word_density'], use_container_width=True)
            with col2:
                if 'char_confidence' in figures:
                    st.plotly_chart(figures['char_confidence'], use_container_width=True)
        
        with tab6:
            if 'correlation_matrix' in figures:
                st.plotly_chart(figures['correlation_matrix'], use_container_width=True)

        # Export functionality
        st.markdown("---")
        st.markdown("## 💾 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if export_format == "CSV":
                csv_data = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=f'metaphor_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                    mime='text/csv'
                )
        
        with col2:
            if export_format == "JSON":
                json_data = filtered_df.to_json(orient='records', indent=2)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_data,
                    file_name=f'metaphor_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
                    mime='application/json'
                )
        
        with col3:
            # Create a summary report
            summary_report = f"""
Tamil Metaphor Analysis Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

SUMMARY STATISTICS:
- Total sentences analyzed: {stats['total_sentences']}
- Metaphors detected: {stats['metaphor_count']} ({stats['metaphor_ratio']:.1%})
- Literal statements: {stats['literal_count']} ({stats['literal_ratio']:.1%})
- Average confidence: {stats['overall_confidence']:.3f}
- High confidence predictions: {stats['high_confidence_count']}

TEXT COMPLEXITY:
- Unique words: {stats['unique_words']}
- Vocabulary richness: {stats['vocabulary_richness']:.3f}
- Average sentence length: {stats['avg_length']:.1f} words

COMPARATIVE ANALYSIS:
- Metaphor avg confidence: {stats['metaphor_avg_conf']:.3f}
- Literal avg confidence: {stats['literal_avg_conf']:.3f}
- Metaphor avg length: {stats['metaphor_avg_length']:.1f} words
- Literal avg length: {stats['literal_avg_length']:.1f} words

STATISTICAL SIGNIFICANCE:
- Confidence difference p-value: {stats['conf_pvalue']:.4f}
- Length difference p-value: {stats['length_pvalue']:.4f}
            """
            
            st.download_button(
                label="📊 Download Report",
                data=summary_report,
                file_name=f'metaphor_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt',
                mime='text/plain'
            )

if __name__ == "__main__":
    main()