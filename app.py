import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import time
from datetime import datetime

# -----------------------------------------------------------------------------
# 1. Luxury Page & UI Architecture (Advanced CSS + Mobile Responsive Fixes)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Vantedge AI | Waste Segmentation",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap');

    .stApp {
        background-color: #000000;
        color: #E0E0E0;
        font-family: 'Poppins', sans-serif;
    }
    
    .brand-container {
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: 5px;
        padding-top: 1rem;
    }
    .vantedge-icon {
        width: 45px;
        height: 45px;
        filter: drop-shadow(0px 4px 6px rgba(230, 198, 87, 0.2));
    }
    .luxury-head {
        text-transform: uppercase;
        letter-spacing: 0.05em;
        background: linear-gradient(135deg, #E6C657 0%, #B6922E 50%, #E6C657 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600;
        font-size: 2.2rem;
        margin: 0;
    }
    .luxury-subhead {
        color: #A1A1AA;
        font-size: 1rem;
        letter-spacing: 0.05em;
        margin-top: 5px;
        margin-bottom: 2rem;
        border-bottom: 1px solid #1A1A1A;
        padding-bottom: 1rem;
    }
    .highlight-brand { color: #E6C657; font-weight: 500; }

    .integrated-ref-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 2rem;
    }
    .ref-card {
        padding: 1.2rem;
        border-radius: 8px;
        border-top: 4px solid;
        background: linear-gradient(180deg, #0A0A0A 0%, #050505 100%);
        border: 1px solid #1A1A1A;
        box-shadow: 0 4px 10px rgba(0,0,0,0.5);
    }
    .ref-title { font-weight: 600; font-size: 1rem; margin-bottom: 0.3rem; }
    .ref-desc { font-size: 0.75rem; color: #A1A1AA; line-height: 1.5; }

    /* Mobile Responsive Fixes */
    @media (max-width: 900px) {
        .luxury-head { font-size: 1.5rem; }
        .integrated-ref-grid { grid-template-columns: repeat(2, 1fr); }
    }
    @media (max-width: 500px) {
        .luxury-head { font-size: 1.2rem; }
        .integrated-ref-grid { grid-template-columns: 1fr; }
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Logic & Engine (Everything retained from your code)
# -----------------------------------------------------------------------------
if "history" not in st.session_state: st.session_state.history = []

BIN_COLOR_MAP = {
    "Blue": {"bright": (246, 130, 59), "dark": (150, 60, 10), "hex": "#3B82F6"},
    "Green": {"bright": (129, 185, 16), "dark": (20, 100, 10), "hex": "#10B981"},
    "Black": {"bright": (180, 180, 180), "dark": (30, 30, 30), "hex": "#A1A1AA"},
    "Yellow": {"bright": (8, 179, 234), "dark": (10, 100, 130), "hex": "#EAB308"}
}

CLASS_TO_BIN = {
    "cardboard": "Blue", "cloth": "Blue", "dairy packets": "Blue", "glass bottle": "Blue",
    "metal can": "Blue", "packaging box": "Blue", "paper": "Blue", "paper bag": "Blue",
    "paper cup": "Blue", "paper utensils": "Blue", "plastic bag": "Blue", "plastic bits": "Blue",
    "plastic bottle": "Blue", "plastic box": "Blue", "plastic cup": "Blue", "plastic packet": "Blue",
    "plastic straw": "Blue", "plastic utensils": "Blue", "synthetic bag": "Blue", "thermocol": "Blue",
    "coconut shell": "Green", "wood materials": "Green",
    "brick": "Black", "broken glass": "Black", "cigarette": "Black", "footwear": "Black",
    "mask": "Black", "sanitary": "Black", "tile": "Black", "tobacco packet": "Black",
    "medical waste": "Yellow"
}

@st.cache_resource(show_spinner=False)
def load_engine(weights="best.pt"):
    try: return YOLO(weights)
    except: return None

engine = load_engine()

def draw_luxury_segmentation(image_pil, results_object):
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    h, w, _ = img_cv.shape
    mask_canvas = np.zeros((h, w, 3), dtype=np.uint8)
    detection_details = []
    boxes = results_object.boxes
    masks = results_object.masks

    if len(boxes) > 0:
        for i, box in enumerate(boxes):
            label_id = int(box.cls[0]); label_name = engine.names.get(label_id, "Item")
            conf = float(box.conf[0]); bin_category = CLASS_TO_BIN.get(label_name.lower(), "Black")
            color_scheme = BIN_COLOR_MAP[bin_category]; x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            if masks is not None and len(masks) > i:
                m_data = masks[i].data[0].cpu().numpy()
                m_scaled = cv2.resize(m_data, (w, h))
                mask_canvas[m_scaled > 0.5] = color_scheme['dark']
            else:
                cv2.rectangle(mask_canvas, (x1, y1), (x2, y2), color_scheme['dark'], -1)
            
            detection_details.append({"label": label_name, "conf": conf, "bin": bin_category, "hex": color_scheme['hex'], "coords": (x1, y1, x2, y2), "bright": color_scheme['bright']})

        cv2.addWeighted(mask_canvas, 0.65, img_cv, 0.35, 0, img_cv)
        for det in detection_details:
            x1, y1, x2, y2 = det['coords']
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), det['bright'], 2)
    return cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), detection_details

# -----------------------------------------------------------------------------
# 3. Branding & Interface
# -----------------------------------------------------------------------------
st.markdown("""
<div class="brand-container">
    <svg class="vantedge-icon" viewBox="0 0 100 100"><polygon points="50,90 10,20 35,20 50,55 65,20 90,20" fill="#E6C657"/></svg>
    <h1 class='luxury-head'>VANTEDGE</h1>
</div>
<p class='luxury-subhead'>Waste Segmentation Engine</p>
""", unsafe_allow_html=True)

# 4-Bin Reference Grid
st.markdown("""
<div class="integrated-ref-grid">
    <div class="ref-card" style="border-top-color: #3B82F6;"><div class="ref-title" style="color: #3B82F6;">BLUE BIN</div><div class="ref-desc">Dry Recyclables</div></div>
    <div class="ref-card" style="border-top-color: #10B981;"><div class="ref-title" style="color: #10B981;">GREEN BIN</div><div class="ref-desc">Organic Waste</div></div>
    <div class="ref-card" style="border-top-color: #A1A1AA;"><div class="ref-title" style="color: #A1A1AA;">BLACK BIN</div><div class="ref-desc">Domestic/Hazardous</div></div>
    <div class="ref-card" style="border-top-color: #EAB308;"><div class="ref-title" style="color: #EAB308;">YELLOW BIN</div><div class="ref-desc">Biomedical</div></div>
</div>
""", unsafe_allow_html=True)

# Scanner Tab with Camera Toggle
tab1, tab2 = st.tabs(["ACTIVE SCANNER", "SESSION HISTORY"])
with tab1:
    input_mode = st.radio("Mode", ["Upload", "Camera"], horizontal=True)
    if input_mode == "Camera":
        # iPad Camera Switch Toggle
        cam_dir = st.radio("Lens:", ["Back (Environment)", "Front (User)"], horizontal=True)
        facing = "environment" if "Back" in cam_dir else "user"
        raw_img = st.camera_input("Optical Sensor", facing_mode=facing)
    else:
        raw_img = st.file_uploader("Upload", type=["jpg", "png"])
        if raw_img: raw_img = Image.open(raw_img)

    if raw_img:
        res = engine.predict(np.array(raw_img), conf=0.4, verbose=False)
        processed, dets = draw_luxury_segmentation(raw_img, res[0])
        st.image(processed, use_container_width=True)
