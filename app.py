# requirements.txt content:
# tensorflow
# streamlit
# numpy
# matplotlib
# seaborn
# scikit-learn
# plotly
# pillow

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
import plotly.graph_objects as go
from PIL import Image
import os

# ==========================================
# 1. PAGE SETUP
# ==========================================
st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design Look
st.markdown("""
<style>
    /* Styling for Titles and Headers */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .subtitle {
        font-size: 1.25rem;
        color: #4A5568;
        font-weight: 500;
        margin-bottom: 2rem;
    }
    /* Info Card Styling */
    .info-card {
        background-color: #f7fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .info-card h4 {
        color: #2b6cb0;
        margin-top: 0;
    }
    /* Prediction Container */
    .prediction-container {
        background: linear-gradient(to right, #ebf8ff, #ebf8ff);
        border: 1px solid #bee3f8;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 15px -3px rgba(66, 153, 225, 0.1);
    }
    .predicted-class {
        font-size: 2.2rem;
        font-weight: 800;
        color: #2b6cb0;
        margin: 0.5rem 0;
    }
    .confidence-score {
        font-size: 1.2rem;
        font-weight: 600;
        color: #4a5568;
    }
    /* Styled Predictions Table */
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1.5rem 0;
        font-size: 1rem;
        text-align: left;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    .styled-table thead tr {
        background-color: #2b6cb0;
        color: #ffffff;
        font-weight: bold;
    }
    .styled-table th, .styled-table td {
        padding: 12px 15px;
    }
    .styled-table tbody tr {
        border-bottom: 1px solid #dddddd;
        background-color: #ffffff;
    }
    .styled-table tbody tr:nth-of-type(even) {
        background-color: #f7fafc;
    }
    .styled-table tbody tr:last-of-type {
        border-bottom: 2px solid #2b6cb0;
    }
    /* Footer Style */
    .footer-container {
        text-align: center;
        padding: 2rem 0;
        margin-top: 3rem;
        border-top: 1px solid #e2e8f0;
        color: #718096;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Define all 10 CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Sidebar Section
st.sidebar.image("https://img.icons8.com/clouds/200/brain.png", width=100)
st.sidebar.title("Internship Project")
st.sidebar.markdown("**Image Classification using Transfer Learning (VGG16)**")
st.sidebar.markdown("---")

st.sidebar.subheader("Dataset Details")
st.sidebar.info(
    "The **CIFAR-10** dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. "
    "We use transfer learning with **VGG16** (pre-trained on ImageNet) to classify these images."
)

st.sidebar.subheader("CIFAR-10 Classes")
for idx, name in enumerate(class_names):
    st.sidebar.markdown(f"**{idx + 1}.** {name.capitalize()}")

st.sidebar.markdown("---")
st.sidebar.markdown("**Submission by:** Machine Learning Intern")
st.sidebar.markdown("**Company:** Navodita Infotech")


# ==========================================
# 2. HEADER SECTION
# ==========================================
st.markdown("<h1 class='main-title'>CIFAR-10 Image Classifier</h1>", unsafe_allow_html=True)
st.markdown(
    "<h3 class='subtitle'>Real-time Image Classification using Transfer Learning (VGG16)</h3>", 
    unsafe_allow_html=True
)

st.markdown(
    "Welcome to the interactive CIFAR-10 Image Classifier! This web application loads a custom neural network "
    "built on top of the **VGG16** model pre-trained on ImageNet. Upload any image corresponding to the 10 classes "
    "(airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) and witness the model predict its class in real-time."
)


# ==========================================
# 3. IMAGE UPLOAD SECTION
# ==========================================
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("<div class='info-card'><h4>📥 Upload Image</h4></div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose an image file...", 
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )

    if uploaded_file is not None:
        # Load and display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Display image dimensions
        width, height = image.size
        st.markdown(f"**Image Dimensions:** {width} x {height} pixels")
        
        # Action button to trigger predictions
        predict_button = st.button("🔮 Classify Image", type="primary", use_container_width=True)
    else:
        st.warning("Please upload an image file to make predictions.")
        predict_button = False


# ==========================================
# 4. PREDICTION LOGIC & SECTION
# ==========================================
# Helper function to cache model loading
@st.cache_resource
def load_classification_model():
    model_paths = ['cifar10_vgg16_model.h5', 'best_cifar10_model.h5']
    loaded_model = None
    for path in model_paths:
        if os.path.exists(path):
            try:
                # Load the model
                loaded_model = tf.keras.models.load_model(path)
                return loaded_model, path
            except Exception as e:
                st.error(f"Error loading model from {path}: {str(e)}")
    return None, None

with col2:
    if uploaded_file is not None and predict_button:
        st.markdown("<div class='info-card'><h4>⚡ Classification Results</h4></div>", unsafe_allow_html=True)
        
        # Load the model
        with st.spinner("Loading trained VGG16 model..."):
            model, path_used = load_classification_model()
            
        if model is None:
            st.error(
                "Could not load the model. Please make sure that 'cifar10_vgg16_model.h5' "
                "or 'best_cifar10_model.h5' exists in the application directory by running the training script."
            )
        else:
            st.success(f"Model successfully loaded from: `{path_used}`")
            
            # Preprocess the uploaded image
            # VGG16 needs 48x48x3 images
            img = image.convert("RGB")
            img_resized = img.resize((48, 48))
            
            # Convert to numpy array
            img_array = np.array(img_resized, dtype=np.float32)
            
            # Expand dimensions to create a batch (1, 48, 48, 3)
            img_batch = np.expand_dims(img_array, axis=0)
            
            # Preprocess using preprocess_input (BGR conversion and mean subtraction)
            img_preprocessed = preprocess_input(img_batch)
            
            # Run prediction
            with st.spinner("Analyzing image..."):
                predictions_probs = model.predict(img_preprocessed)[0]
            
            # Best class details
            pred_idx = np.argmax(predictions_probs)
            pred_class = class_names[pred_idx]
            confidence = predictions_probs[pred_idx]
            
            # Display Prediction Card
            st.markdown(f"""
            <div class='prediction-container'>
                <span>Top Prediction:</span>
                <div class='predicted-class'>{pred_class.upper()}</div>
                <div class='confidence-score'>Confidence: {confidence * 100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            # ==========================================
            # 5. TOP 3 PREDICTIONS TABLE
            # ==========================================
            st.markdown("##### 🏆 Top 3 Predictions")
            top_3_indices = np.argsort(predictions_probs)[::-1][:3]
            
            table_rows = ""
            for idx in top_3_indices:
                class_label = class_names[idx].capitalize()
                prob = predictions_probs[idx] * 100
                table_rows += f"<tr><td>{class_label}</td><td>{prob:.2f}%</td></tr>"
                
            table_html = f"""
            <table class='styled-table'>
                <thead>
                    <tr>
                        <th>Class Label</th>
                        <th>Confidence</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
            """
            st.markdown(table_html, unsafe_allow_html=True)
            
            # Plotly Bar Chart of probabilities
            st.markdown("##### 📊 Confidence Score Distribution")
            fig = go.Figure(go.Bar(
                x=[name.capitalize() for name in class_names],
                y=predictions_probs * 100,
                marker_color=['#2b6cb0' if i == pred_idx else '#cbd5e0' for i in range(10)],
                text=[f"{p*100:.1f}%" for p in predictions_probs],
                textposition='auto',
            ))
            
            fig.update_layout(
                margin=dict(l=20, r=20, t=10, b=10),
                height=300,
                xaxis_title="Classes",
                yaxis_title="Probability (%)",
                yaxis=dict(range=[0, 100]),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
    elif uploaded_file is not None:
        st.info("Click the **'Classify Image'** button to run prediction.")


# ==========================================
# 6. FOOTER
# ==========================================
st.markdown("""
<div class='footer-container'>
    <p>© 2026 CIFAR-10 Transfer Learning Internship Project</p>
    <p>Developed for evaluation at <strong>Navodita Infotech</strong></p>
</div>
""", unsafe_allow_html=True)
