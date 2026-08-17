import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import time

# -----------------------------------------------------------------------------
# 1. Page & UI Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Vision | Waste Segregation Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Elite, High-End Custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0A0A0A;
        color: #EDEDED;
        font-family: "Inter", sans-serif;
    }
    h1, h2, h3 {
        color: #FFFFFF;
        font-weight: 300;
        letter-spacing: -0.02em;
    }
    .guide-card {
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 6px solid;
        background-color: #171717;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .guide-blue { border-color: #3B82F6; }
    .guide-green { border-color: #10B981; }
    .guide-black { border-color: #3F3F46; }
    .guide-yellow { border-color: #EAB308; }
    
    .guide-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 300;
        color: #FFFFFF;
    }
    div[data-testid="stMetricLabel"] {
        color: #A1A1AA;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Waste Classification Mapping
# -----------------------------------------------------------------------------

# RGB Color Definitions for OpenCV (Format: B, G, R)
BIN_COLORS = {
    "Blue": (246, 130, 59),    # #3B82F6
    "Green": (129, 185, 16),   # #10B981
    "Black": (100, 100, 100),  # #646464 (Lighter black/grey for visibility)
    "Yellow": (8, 179, 234)    # #EAB308
}

# Hex Colors for UI elements
UI_COLORS = {
    "Blue": "#3B82F6",
    "Green": "#10B981",
    "Black": "#71717A",
    "Yellow": "#EAB308"
}

# IMPORTANT: Map your exact Roboflow class names to the correct bin color here.
# Change the keys ("plastic", "paper", etc.) to match your model's exact labels.
# IMPORTANT: Map your exact Roboflow class names to the correct bin color here.
CLASS_TO_BIN = {
    # Blue Bin (Dry Recyclables)
    "cardboard": "Blue",
    "cloth": "Blue",
    "dairy packets": "Blue",
    "glass bottle": "Blue",
    "metal can": "Blue",
    "packaging box": "Blue",
    "paper": "Blue",
    "paper bag": "Blue",
    "paper cup": "Blue",
    "paper utensils": "Blue",
    "plastic bag": "Blue",
    "plastic bits": "Blue",
    "plastic bottle": "Blue",
    "plastic box": "Blue",
    "plastic cup": "Blue",
    "plastic packet": "Blue",
    "plastic straw": "Blue",
    "plastic utensils": "Blue",
    "synthetic bag": "Blue",
    "thermocol": "Blue",
    
    # Green Bin (Organic / Compostable)
    "coconut shell": "Green",
    "wood materials": "Green",
    
    # Black Bin (Sanitary / Hazardous / Non-recyclable Domestic)
    "brick": "Black",
    "broken glass": "Black",
    "cigarette": "Black",
    "footwear": "Black",
    "mask": "Black",
    "sanitary": "Black",
    "tile": "Black",
    "tobacco packet": "Black",
    
    # Yellow Bin (Biomedical / Clinical)
    "medical waste": "Yellow"
}


# -----------------------------------------------------------------------------
# 3. Core Engine & Drawing Functions
# -----------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

@st.cache_resource(show_spinner=False)
def load_model(weights_path="best.pt"):
    try:
        return YOLO(weights_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()

def draw_colored_boxes(image_pil, boxes, class_names):
    """Draws custom colored bounding boxes and masks based on bin classification."""
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    overlay = img_cv.copy()
    
    detections_data = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = class_names.get(cls_id, f"Class {cls_id}")
        
        # Determine Bin Category (Default to Black if not found in dictionary)
        bin_category = CLASS_TO_BIN.get(label.lower(), "Black")
        color = BIN_COLORS[bin_category]
        ui_color = UI_COLORS[bin_category]
        
        # Draw translucent mask
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        
        # Draw sharp bounding box
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
        
        # Draw Label Background and Text
        text = f"{label.capitalize()} ({conf:.2f})"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_cv, (x1, y1 - 20), (x1 + tw + 5, y1), color, -1)
        cv2.putText(img_cv, text, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        detections_data.append({
            "Detected Item": label.capitalize(),
            "Bin Assignment": bin_category,
            "Confidence": f"{conf * 100:.1f}%",
            "Color": ui_color
        })

    # Apply alpha blending for the mask effect
    cv2.addWeighted(overlay, 0.3, img_cv, 0.7, 0, img_cv)
    return cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), detections_data

# -----------------------------------------------------------------------------
# 4. Interface Architecture
# -----------------------------------------------------------------------------

st.title("Waste Segregation Engine")
st.markdown("Advanced computer vision system for Indian municipal and domestic waste classification.")

# Navigation Tabs
tab_scanner, tab_guide, tab_history = st.tabs(["Active Scanner", "Disposal Guide", "Scan History"])

with tab_scanner:
    with st.expander("System Initialization & Usage Instructions", expanded=True):
        st.markdown("""
        **How to use this tool:**
        1. **Select Input Method:** Choose between uploading a local image file or activating your device camera.
        2. **Capture/Upload:** Ensure the waste material is well-lit and clearly visible in the frame.
        3. **Process:** The YOLOv8 AI model will automatically analyze the image.
        4. **Review:** The system will draw a colored mask over the item, corresponding to the correct disposal bin (Blue, Green, Black, or Yellow).
        """)
        
    input_mode = st.radio("Stream Selection", ["Image Upload", "Camera Capture"], horizontal=True, label_visibility="collapsed")
    
    raw_image = None
    if input_mode == "Image Upload":
        uploaded_file = st.file_uploader("Drop image file here", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_file:
            raw_image = Image.open(uploaded_file).convert("RGB")
    else:
        camera_file = st.camera_input("Initialize Camera")
        if camera_file:
            raw_image = Image.open(camera_file).convert("RGB")
            
    if raw_image is not None and model is not None:
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h3 style='font-size: 1.2rem; color: #A1A1AA;'>Source Input</h3>", unsafe_allow_html=True)
            st.image(raw_image, use_container_width=True)
            
        with col2:
            st.markdown("<h3 style='font-size: 1.2rem; color: #A1A1AA;'>Processed Analysis</h3>", unsafe_allow_html=True)
            with st.spinner("Executing neural network..."):
                start_time = time.time()
                
                # Run YOLO Inference (suppress drawing, we do it manually)
                results = model.predict(source=np.array(raw_image), conf=0.35, iou=0.45, verbose=False)
                latency = round((time.time() - start_time) * 1000, 2)
                
                res = results[0]
                
                # Apply custom drawing logic
                if len(res.boxes) > 0:
                    annotated_image, detection_details = draw_colored_boxes(raw_image, res.boxes, model.names)
                else:
                    annotated_image = np.array(raw_image)
                    detection_details = []
                
                st.image(annotated_image, use_container_width=True)
                
                # Store History
                st.session_state.history.append({
                    "timestamp": time.strftime("%H:%M:%S"),
                    "thumbnail": annotated_image.copy(),
                    "details": detection_details,
                    "latency": latency
                })

        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Objects Identified", len(detection_details))
        m2.metric("Inference Latency", f"{latency} ms")
        m3.metric("System Status", "Optimal")
        
        if detection_details:
            st.markdown("<h3 style='font-size: 1.2rem; margin-top: 1rem;'>Segregation Instructions</h3>", unsafe_allow_html=True)
            for item in detection_details:
                st.markdown(
                    f"<div style='padding: 10px; border-left: 4px solid {item['Color']}; background-color: #171717; margin-bottom: 5px; border-radius: 4px;'>"
                    f"<strong style='color: #FFFFFF;'>{item['Detected Item']}</strong> "
                    f"<span style='color: #A1A1AA;'>({item['Confidence']})</span> — "
                    f"Route to <strong style='color: {item['Color']};'>{item['Bin Assignment']} Bin</strong>"
                    f"</div>", 
                    unsafe_allow_html=True
                )

with tab_guide:
    st.markdown("### Standard Segregation Protocol")
    st.markdown("The system automatically color-codes detections to match the following municipal standards:")
    
    st.markdown("""
    <div class="guide-card guide-blue">
        <div class="guide-title" style="color: #3B82F6;">Blue Bin (Dry Recyclables)</div>
        <div style="color: #D4D4D8;">Designated for recyclable dry waste. This includes plastics, paper, cardboard, glass, and metals. Items should be rinsed and free of heavy food residue before disposal.</div>
    </div>
    
    <div class="guide-card guide-green">
        <div class="guide-title" style="color: #10B981;">Green Bin (Organic Waste)</div>
        <div style="color: #D4D4D8;">Designated for compostable, organic, kitchen, and garden waste. This includes fruit peels, vegetable scraps, leftover food, and dead leaves.</div>
    </div>
    
    <div class="guide-card guide-black">
        <div class="guide-title" style="color: #A1A1AA;">Black Bin (Domestic/Hazardous)</div>
        <div style="color: #D4D4D8;">Designated for sanitary, hazardous, and domestic waste that cannot be recycled or composted. This includes diapers, sanitary napkins, e-waste, batteries, and chemically contaminated materials.</div>
    </div>
    
    <div class="guide-card guide-yellow">
        <div class="guide-title" style="color: #EAB308;">Yellow Bin (Biomedical Waste)</div>
        <div style="color: #D4D4D8;">Designated for biomedical and clinical waste. This includes used syringes, bandages, expired medicines, and any materials exposed to bodily fluids.</div>
    </div>
    """, unsafe_allow_html=True)

with tab_history:
    st.markdown("### Session Scan History")
    if not st.session_state.history:
        st.markdown("<p style='color: #A1A1AA;'>No scans processed in the current session.</p>", unsafe_allow_html=True)
    else:
        for idx, item in enumerate(reversed(st.session_state.history)):
            with st.expander(f"Scan {len(st.session_state.history) - idx} at {item['timestamp']}"):
                h_col1, h_col2 = st.columns([1, 2])
                with h_col1:
                    st.image(item["thumbnail"], width=250)
                with h_col2:
                    st.markdown(f"**Processing Time:** {item['latency']} ms")
                    if item["details"]:
                        for det in item["details"]:
                            st.markdown(f"- **{det['Detected Item']}** → {det['Bin Assignment']} Bin")
                    else:
                        st.write("No objects identified.")
