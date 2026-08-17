import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import time

# Page Configuration
st.set_page_config(
    page_title="EcoSegregate | AI Waste Classification",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End CSS
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #0B1120;
        color: #F8FAFC;
    }
    
    /* Headers & Typography */
    h1, h2, h3, h4 {
        color: #F8FAFC;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    
    /* Metrics and Cards */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        color: #10B981;
        font-weight: 700;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #94A3B8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .card-container {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    
    /* Button Customization */
    div.stButton > button:first-child {
        background-color: #059669;
        color: #FFFFFF;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s ease-in-out;
    }
    
    div.stButton > button:first-child:hover {
        background-color: #10B981;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2);
    }
    
    /* Table Styling */
    div[data-testid="stDataFrame"] {
        border-radius: 6px;
        border: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session History
if "history" not in st.session_state:
    st.session_state.history = []

# Model Loader
@st.cache_resource(show_spinner=False)
def load_model(weights_path="best.pt"):
    try:
        model = YOLO(weights_path)
        return model
    except Exception as e:
        st.error(f"Failed to load model from {weights_path}: {e}")
        return None

model = load_model("best.pt")

# Sidebar Configuration
with st.sidebar:
    st.title("EcoSegregate AI")
    st.caption("Automated Indian Municipal & Domestic Waste Sorting")
    st.markdown("---")
    
    st.subheader("Inference Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.10, 1.00, 0.40, 0.05)
    iou_threshold = st.slider("IoU Threshold (NMS)", 0.10, 1.00, 0.45, 0.05)
    
    st.markdown("---")
    st.subheader("System Status")
    if model:
        st.success("YOLO Engine: Active")
        st.caption(f"Target Classes: {len(model.names)}")
    else:
        st.error("YOLO Engine: Offline (Check best.pt)")

# Main Layout Tabs
tab_detect, tab_history, tab_about = st.tabs(["Classification Console", "Session History", "Model Overview"])

with tab_detect:
    st.title("Indian Waste Material Segregation")
    st.write("Upload an image or use an integrated camera feed to run YOLOv8 object detection and categorization.")
    
    input_mode = st.radio("Select Input Stream", ["Image Upload", "Camera Capture"], horizontal=True)
    
    raw_image = None
    if input_mode == "Image Upload":
        uploaded_file = st.file_uploader("Upload Image File", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_file:
            raw_image = Image.open(uploaded_file).convert("RGB")
    else:
        camera_file = st.camera_input("Capture Waste Frame")
        if camera_file:
            raw_image = Image.open(camera_file).convert("RGB")
            
    if raw_image is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Source Input")
            st.image(raw_image, use_container_width=True)
            
        with col2:
            st.subheader("Inference Output")
            if model is not None:
                with st.spinner("Processing image via YOLOv8 engine..."):
                    start_time = time.time()
                    results = model.predict(
                        source=np.array(raw_image),
                        conf=confidence_threshold,
                        iou=iou_threshold,
                        verbose=False
                    )
                    latency = round((time.time() - start_time) * 1000, 2)
                    
                    res = results[0]
                    annotated_frame = res.plot()
                    
                    st.image(annotated_frame, use_container_width=True)
                    
                    # Extract Detection Metrics
                    detections = []
                    for box in res.boxes:
                        cls_id = int(box.cls[0].item())
                        conf_val = float(box.conf[0].item())
                        label = model.names.get(cls_id, f"Class {cls_id}")
                        detections.append({
                            "Class": label,
                            "Confidence": f"{conf_val * 100:.1f}%"
                        })
                        
                    # Save to History
                    record = {
                        "timestamp": time.strftime("%H:%M:%S"),
                        "thumbnail": raw_image.copy(),
                        "detections_count": len(detections),
                        "details": detections,
                        "latency_ms": latency
                    }
                    st.session_state.history.append(record)
                    
        # Real-time Metrics Dashboard
        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Objects Detected", len(detections))
        m2.metric("Processing Latency", f"{latency} ms")
        m3.metric("Model Engine", "YOLOv8x Roboflow")
        
        if detections:
            st.subheader("Detected Material Breakdown")
            st.table(detections)
        else:
            st.info("No items identified above the selected confidence threshold.")

with tab_history:
    st.title("Scan History")
    
    if not st.session_state.history:
        st.info("No scans recorded in this session yet.")
    else:
        for idx, item in enumerate(reversed(st.session_state.history)):
            with st.expander(f"Scan #{len(st.session_state.history) - idx} - {item['timestamp']} ({item['detections_count']} items found)"):
                h_col1, h_col2 = st.columns([1, 2])
                with h_col1:
                    st.image(item["thumbnail"], width=200)
                with h_col2:
                    st.write(f"Inference Latency: **{item['latency_ms']} ms**")
                    if item["details"]:
                        st.table(item["details"])
                    else:
                        st.write("No items detected.")

with tab_about:
    st.title("Model Architecture & Guidelines")
    st.markdown("""
    ### System Details
    - **Engine:** Ultralytics YOLOv8
    - **Trained on:** Roboflow Indian Municipal Waste Dataset
    - **Inference Pipeline:** Non-Maximum Suppression (NMS) with custom IoU thresholds
    
    ### Guidelines for Best Results
    1. Ensure direct, uniform lighting over waste items.
    2. Maintain a distance of 30 cm to 1 meter between the camera and items.
    3. Separate overlapping objects to reduce bounding box occlusion.
    """)
