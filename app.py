"""
Streamlit Web UI for Iris Classification Model
Complete deployment package with UI, predictions, and data insights
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

from scripts.predict import PredictionEngine

# Configure page
st.set_page_config(
    page_title="Iris Classification - ML Deployment",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .metric-card { background-color: #f0f2f6; padding: 1.5rem; border-radius: 0.5rem; margin: 1rem 0; }
    h1 { color: #1f77b4; margin-bottom: 1.5rem; }
    h2 { color: #2ca02c; margin-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'engine' not in st.session_state:
    try:
        st.session_state.engine = PredictionEngine()
        st.session_state.engine_loaded = True
    except Exception as e:
        st.session_state.engine_loaded = False
        st.session_state.error = str(e)


def load_sample_data():
    """Load Iris dataset for insights"""
    data_path = Path(__file__).parent / "data" / "iris_train.csv"
    return pd.read_csv(data_path)


def load_latest_metrics():
    """Load the latest evaluation metrics from the training history database."""
    db_path = Path(__file__).parent / "data" / "training_history.db"
    if not db_path.exists():
        return None

    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT accuracy, loss, f1_score, precision, recall, epochs_trained, timestamp
            FROM training_metrics
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def display_data_insights():
    """Display dataset statistics and visualizations"""
    st.header("Data Insights and Analysis")
    
    df = load_sample_data()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        st.metric("Features", 4)
    with col3:
        st.metric("Classes", len(df['species'].unique()))
    with col4:
        st.metric("Dataset Shape", f"{df.shape[0]} × {df.shape[1]}")
    
    # Class distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Class Distribution")
        class_dist = df['species'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 5))
        class_dist.plot(kind='bar', color=['#FF6B6B', '#4ECDC4', '#45B7D1'], ax=ax)
        ax.set_title('Number of Samples per Iris Species', fontsize=12, fontweight='bold')
        ax.set_xlabel('Species')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Feature Statistics")
        st.dataframe(df.describe().round(2))
    
    # Feature distributions
    st.subheader("Feature Distributions by Species")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Distribution of Iris Features', fontsize=14, fontweight='bold')
    
    features = df.columns[:-1]
    for idx, feature in enumerate(features):
        ax = axes[idx // 2, idx % 2]
        for species in df['species'].unique():
            data = df[df['species'] == species][feature]
            ax.hist(data, alpha=0.6, label=species)
        ax.set_xlabel(feature)
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Feature correlation
    st.subheader("Feature Correlation Matrix")
    fig, ax = plt.subplots(figsize=(8, 6))
    correlation = df.drop('species', axis=1).corr()
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title('Feature Correlation Heatmap', fontweight='bold')
    st.pyplot(fig)


def display_prediction_interface():
    """Display prediction input interface"""
    st.header("Make a Prediction")
    
    if not st.session_state.engine_loaded:
        st.error(f"Could not load model: {st.session_state.error}")
        return
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Manual Input", "CSV File"],
        horizontal=True
    )
    
    if input_method == "Manual Input":
        st.subheader("Enter Iris Flower Features")
        
        col1, col2 = st.columns(2)
        with col1:
            sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.5, step=0.1)
            petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0, step=0.1)
        
        with col2:
            sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0, step=0.1)
            petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.2, step=0.1)
        
        if st.button("Run Prediction", use_container_width=True, type="primary"):
            features = [sepal_length, sepal_width, petal_length, petal_width]
            
            try:
                report = st.session_state.engine.get_prediction_report(features)
                
                # Display results
                st.success("Prediction completed successfully.")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Prediction Result")
                    st.metric(
                        "Predicted Species",
                        report['predicted_class'].upper(),
                        f"{report['confidence']:.1f}% confidence"
                    )
                
                with col2:
                    st.subheader("Probability Distribution")
                    prob_data = report['probability_distribution']
                    fig, ax = plt.subplots(figsize=(8, 5))
                    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
                    bars = ax.bar(prob_data.keys(), prob_data.values(), color=colors)
                    ax.set_ylabel('Probability (%)')
                    ax.set_title('Model Confidence Distribution')
                    ax.set_ylim([0, 100])
                    
                    # Add percentage labels on bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.1f}%', ha='center', va='bottom')
                    
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                
                # Feature details
                st.subheader("Input Features Used")
                features_df = pd.DataFrame({
                    'Feature': ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
                    'Value (cm)': features
                })
                st.table(features_df)
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
    
    else:  # CSV File upload
        st.subheader("Upload CSV for Batch Prediction")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file:
            try:
                df_input = pd.read_csv(uploaded_file)
                st.write(f"Uploaded {len(df_input)} samples")
                st.dataframe(df_input.head())
                
                if st.button("Run Batch Prediction", use_container_width=True, type="primary"):
                    with st.spinner("Processing predictions..."):
                        # Save temp file and predict
                        temp_path = "temp_input.csv"
                        df_input.to_csv(temp_path, index=False)
                        
                        results = st.session_state.engine.predict_from_csv_file(temp_path)
                        results_df = pd.DataFrame(results)
                        
                        st.success("Batch predictions completed successfully.")
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            "Download Results",
                            csv,
                            "predictions.csv",
                            "text/csv"
                        )
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")


def display_model_info():
    """Display model information and specifications"""
    st.header("Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Specifications")
        st.write("""
        - **Architecture**: Feed-forward neural network
        - **Input Features**: 4 (Sepal Length, Sepal Width, Petal Length, Petal Width)
        - **Output Classes**: 3 (Setosa, Versicolor, Virginica)
        - **Hidden Layers**: 64 → 32 → 16 units with ReLU activation
        - **Output Layer**: Softmax for class probabilities
        - **Deployment Artifacts**: Native Keras model + saved StandardScaler
        """)
    
    with col2:
        st.subheader("Optimization Techniques")
        st.write("""
        - Data standardization with `StandardScaler`
        - L2 regularization across hidden layers
        - Dropout at 30 percent
        - Adam optimizer for training stability
        - Early stopping based on validation loss
        """)
    
    st.subheader("Evaluation Metrics")
    latest_metrics = load_latest_metrics()
    if latest_metrics:
        metrics_df = pd.DataFrame(
            {
                "Metric": ["Accuracy", "Loss", "F1 Score", "Precision", "Recall"],
                "Value": [
                    f"{latest_metrics['accuracy']:.4f}",
                    f"{latest_metrics['loss']:.4f}",
                    f"{latest_metrics['f1_score']:.4f}",
                    f"{latest_metrics['precision']:.4f}",
                    f"{latest_metrics['recall']:.4f}",
                ],
            }
        )
        st.table(metrics_df)
        st.caption(
            f"Latest retraining run: {latest_metrics['timestamp']} | "
            f"Epochs trained: {latest_metrics['epochs_trained']}"
        )
    else:
        st.info("No training metrics are available yet. Run scripts/retrain.py to populate the dashboard.")


def display_deployment_info():
    """Display deployment package information"""
    st.header("Deployment Package")
    
    st.write("""
    This application showcases a complete ML deployment package including:
    
    **Components:**
    - Training notebook with model evaluation
    - Retraining pipeline with database integration
    - Prediction engine with batch processing
    - Web UI built with Streamlit
    - Docker containerization
    - Data insights and visualizations
    
    **File Structure:**
    ```
    Summative assignment - MLOP/
    ├── notebooks/model_training.ipynb       # Training & evaluation
    ├── scripts/
    │   ├── retrain.py                       # Retraining pipeline
    │   ├── predict.py                       # Prediction engine
    │   └── preprocess.py                    # Data preprocessing
    ├── models/iris_model.keras              # Trained model
    ├── app.py                               # Streamlit web UI
    ├── Dockerfile                           # Container config
    └── requirements.txt                     # Dependencies
    ```
    """)
    
    st.info("""
    **To Deploy:**
    1. Build Docker image: `docker build -t iris-model .`
    2. Run container: `docker run -p 8501:8501 iris-model`
    3. Access at: http://localhost:8501
    """)


# Main app navigation
def main():
    # Sidebar navigation
    st.sidebar.title("Iris Classification")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["Data Insights", "Predictions", "Model Info", "Deployment"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About
    Machine Learning Operations (MLOps) project demonstrating:
    - Model training & evaluation
    - Data preprocessing & optimization
    - Production deployment
    - Real-time predictions
    
    **Status**: Ready for deployment
    """)
    
    # Route to pages
    if page == "Data Insights":
        display_data_insights()
    elif page == "Predictions":
        display_prediction_interface()
    elif page == "Model Info":
        display_model_info()
    elif page == "Deployment":
        display_deployment_info()


if __name__ == "__main__":
    main()