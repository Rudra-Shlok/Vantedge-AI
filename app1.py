import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import time
from datetime import datetime

# -----------------------------------------------------------------------------
# 1. Luxury Page & UI Architecture (Advanced CSS)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Vantedge AI | Waste Segmentation",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Deep Obsidian, Gold, & Rich Metal Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap');

    .stApp {
        background-color: #000000;
        color: #E0E0E0;
        font-family: 'Poppins', sans-serif;
    }
    
    /* Vantedge Branding & Header */
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
        letter-spacing: 0.12em;
        background: linear-gradient(135deg, #E6C657 0%, #B6922E 50%, #E6C657 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600;
        font-size: 2.2rem;
        margin: 0;
        padding: 0;
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
    .highlight-brand {
        color: #E6C657;
        font-weight: 500;
    }

    /* Reference Cards - Integrated on Main Page */
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
        border-left: 1px solid #1A1A1A;
        border-right: 1px solid #1A1A1A;
        border-bottom: 1px solid #1A1A1A;
        box-shadow: 0 4px 10px rgba(0,0,0,0.5);
    }
    .ref-title { font-weight: 600; font-size: 1.05rem; margin-bottom: 0.3rem; letter-spacing: 0.05em; }
    .ref-desc { font-size: 0.8rem; color: #A1A1AA; line-height: 1.5; }

    /* Button Styling */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #262626 0%, #1A1A1A 100%);
        color: #E6C657;
        border: 1px solid #E6C657;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #E6C657 0%, #B6922E 100%);
        color: #000000;
        border-color: #000000;
        box-shadow: 0 0 15px rgba(230, 198, 87, 0.3);
    }
    
    /* Guide & History Elements */
    .guide-step {
        background-color: #0A0A0A;
        border-left: 3px solid #E6C657;
        padding: 15px;
        margin-bottom: 10px;
        border-radius: 0 8px 8px 0;
    }
    .history-card {
        background-color: #0A0A0A;
        border: 1px solid #1A1A1A;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. State & Engine Initialization
# -----------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

BIN_COLOR_MAP = {
    "Blue":   {"bright": (246, 130, 59),  "dark": (150, 60, 10),   "hex": "#3B82F6"},
    "Green":  {"bright": (129, 185, 16),  "dark": (20, 100, 10),   "hex": "#10B981"},
    "Black":  {"bright": (180, 180, 180), "dark": (30, 30, 30),    "hex": "#A1A1AA"},
    "Yellow": {"bright": (8, 179, 234),   "dark": (10, 100, 130),  "hex": "#EAB308"}
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
    try:
        return YOLO(weights)
    except Exception as e:
        st.error(f"Failed to load Vantedge AI engine: {e}")
        return None

engine = load_engine()

def draw_luxury_segmentation(image_pil, results_object):
    """Hybrid drawing: Renders polygon masks if available, otherwise rich tinted boxes."""
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    h, w, _ = img_cv.shape
    mask_canvas = np.zeros((h, w, 3), dtype=np.uint8)
    
    detection_details = []
    boxes = results_object.boxes
    masks = results_object.masks

    if len(boxes) > 0:
        for i, box in enumerate(boxes):
            label_id = int(box.cls[0])
            label_name = engine.names.get(label_id, f"Class {label_id}")
            conf = float(box.conf[0])
            
            bin_category = CLASS_TO_BIN.get(label_name.lower(), "Black")
            color_scheme = BIN_COLOR_MAP[bin_category]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            if masks is not None and len(masks) > i:
                m_data = masks[i].data[0].cpu().numpy()
                m_scaled = cv2.resize(m_data, (w, h))
                bool_mask = m_scaled > 0.5
                mask_canvas[bool_mask] = color_scheme['dark']
            else:
                cv2.rectangle(mask_canvas, (x1, y1), (x2, y2), color_scheme['dark'], -1)
            
            detection_details.append({
                "label": label_name,
                "conf": conf,
                "bin": bin_category,
                "bright_color": color_scheme['bright'],
                "hex": color_scheme['hex'],
                "coords": (x1, y1, x2, y2)
            })

        cv2.addWeighted(mask_canvas, 0.65, img_cv, 0.35, 0, img_cv)

        for det in detection_details:
            x1, y1, x2, y2 = det['coords']
            bright_color = det['bright_color']
            
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), bright_color, 2)
            txt = f"{det['label'].capitalize()} | {det['bin']} Bin ({int(det['conf']*100)}%)"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img_cv, (x1, y1 - 22), (x1 + tw + 6, y1), bright_color, -1)
            cv2.putText(img_cv, txt, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    return cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), detection_details

# -----------------------------------------------------------------------------
# 3. High-Tech Interface Layout & Branding
# -----------------------------------------------------------------------------

# Vantedge AI Custom SVG Logo & Header
st.markdown("""
<div class="brand-container">
    <svg class="vantedge-icon" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
        <polygon points="50,90 10,20 35,20 50,55 65,20 90,20" fill="url(#goldGradient)"/>
        <defs>
            <linearGradient id="goldGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#E6C657;stop-opacity:1" />
                <stop offset="50%" style="stop-color:#B6922E;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#E6C657;stop-opacity:1" />
            </linearGradient>
        </defs>
    </svg>
    <h1 class='luxury-head'>Waste Detection And Segmentation</h1>
</div>
<p class='luxury-subhead'>Engineered by <span class="highlight-brand">Vantedge AI</span> • Proprietary Classification Model</p>
""", unsafe_allow_html=True)

# Main Application Container
with st.container():
    # -------------------------------------------------
    # INTEGRATED REFERENCE GUIDE
    # -------------------------------------------------
    st.markdown("""
    <div class="integrated-ref-grid">
        <div class="ref-card" style="border-top-color: #3B82F6;">
            <div class="ref-title" style="color: #3B82F6;">BLUE BIN</div>
            <div class="ref-desc">Recyclable dry waste: plastics, paper, glass, cardboard, metals. Clean residue.</div>
        </div>
        <div class="ref-card" style="border-top-color: #10B981;">
            <div class="ref-title" style="color: #10B981;">GREEN BIN</div>
            <div class="ref-desc">Organic and compostable waste: food scraps, peels, kitchen waste, garden leaves.</div>
        </div>
        <div class="ref-card" style="border-top-color: #A1A1AA;">
            <div class="ref-title" style="color: #A1A1AA;">BLACK BIN</div>
            <div class="ref-desc">Sanitary, hazardous, non-recyclable domestic waste: diapers, e-waste, chemicals.</div>
        </div>
        <div class="ref-card" style="border-top-color: #EAB308;">
            <div class="ref-title" style="color: #EAB308;">YELLOW BIN</div>
            <div class="ref-desc">Biomedical and clinical waste: used syringes, bandages, expired medicine, clinical items.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # -------------------------------------------------
    # TABS: SCANNER | HISTORY | USER GUIDE
    # -------------------------------------------------
    tab_scan, tab_history, tab_guide = st.tabs(["ACTIVE SCANNER", "SESSION HISTORY", "USER GUIDE"])
    
    # --- TAB 1: SCANNER ---
    with tab_scan:
        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2 = st.columns([1, 1])
        with m1:
            input_mode = st.radio("Initializing Sensor Array", ["Static Image Upload", "Live Camera Feed"], horizontal=True, label_visibility="collapsed")
        
        raw_img = None
        if input_mode == "Static Image Upload":
            up_file = st.file_uploader("Insert Image Data", type=["jpg", "jpeg", "png", "webp"])
            if up_file:
                raw_img = Image.open(up_file).convert("RGB")
        else:
            cam_file = st.camera_input("Activate Optical Sensor")
            if cam_file:
                raw_img = Image.open(cam_file).convert("RGB")
                
        if raw_img is not None and engine is not None:
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<p style='font-size: 0.8rem; color: #A1A1AA; text-transform: uppercase; letter-spacing: 1px;'>Raw Optical Input</p>", unsafe_allow_html=True)
                st.image(raw_img, use_container_width=True)
                
            with col2:
                st.markdown("<p style='font-size: 0.8rem; color: #A1A1AA; text-transform: uppercase; letter-spacing: 1px;'>Vantedge AI Analysis (Masks Active)</p>", unsafe_allow_html=True)
                with st.spinner("Vantedge Neural Engine Processing..."):
                    start_t = time.time()
                    
                    # YOLO Inference
                    y_results = engine.predict(source=np.array(raw_img), conf=0.40, verbose=False)
                    latency = round((time.time() - start_t) * 1000, 1)
                    
                    # Core Masking/Boxing
                    processed_img, det_list = draw_luxury_segmentation(raw_img, y_results[0])
                    st.image(processed_img, use_container_width=True)
                    
                    # Log to History
                    st.session_state.history.append({
                        "id": datetime.now().strftime("%H:%M:%S"),
                        "thumb": processed_img,
                        "latency": latency,
                        "details": det_list
                    })

            st.markdown("---")
            if det_list:
                st.markdown("<h3 style='font-size: 1.2rem; color: #E6C657; font-weight: 400;'>Segregation Directives Executed</h3>", unsafe_allow_html=True)
                for det in det_list:
                    st.markdown(
                        f"<div style='padding: 12px; border-radius: 6px; border-left: 4px solid {det['hex']}; background-color: #0A0A0A; margin-bottom: 8px; color: #E0E0E0;'>"
                        f"Item identified as <strong style='color: #FFFFFF;'>{det['label'].capitalize()}</strong> "
                        f"— Authorized for <strong style='color: {det['hex']};'>{det['bin'].upper()} BIN</strong> disposal."
                        f"</div>", 
                        unsafe_allow_html=True
                    )
            else:
                st.info("Zero objects detected matching current confidence thresholds.")

    # --- TAB 2: HISTORY ---
    with tab_history:
        st.markdown("<br>", unsafe_allow_html=True)
        if not st.session_state.history:
            st.markdown("<p style='color: #A1A1AA;'>No inference tasks executed in the current session.</p>", unsafe_allow_html=True)
        else:
            for item in reversed(st.session_state.history):
                with st.container():
                    st.markdown(f"<div class='history-card'>", unsafe_allow_html=True)
                    h_col1, h_col2 = st.columns([1, 3])
                    with h_col1:
                        st.image(item['thumb'], use_container_width=True)
                    with h_col2:
                        st.markdown(f"<strong style='color: #E6C657;'>Scan ID:</strong> {item['id']}", unsafe_allow_html=True)
                        st.markdown(f"<strong style='color: #E6C657;'>Compute Latency:</strong> {item['latency']} ms", unsafe_allow_html=True)
                        st.markdown(f"<strong style='color: #E6C657;'>Items Detected:</strong> {len(item['details'])}", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

    # --- TAB 3: USER GUIDE ---
    with tab_guide:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: #E6C657; font-size: 1.3rem; margin-bottom: 15px;'>Vantedge AI Operational Protocol</h3>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="guide-step">
            <strong style="color: #FFFFFF;">1. Initialization:</strong> 
            <span style="color: #A1A1AA;">Select between static image upload or live optical camera feed via the radio buttons in the Active Scanner tab.</span>
        </div>
        <div class="guide-step">
            <strong style="color: #FFFFFF;">2. Environmental Controls:</strong> 
            <span style="color: #A1A1AA;">Ensure the target waste material is well-lit. Avoid extreme shadows or heavy occlusion (stacking items on top of one another).</span>
        </div>
        <div class="guide-step">
            <strong style="color: #FFFFFF;">3. Distance Parameters:</strong> 
            <span style="color: #A1A1AA;">Maintain a standard distance of 30cm to 100cm between the lens and the subject for optimal feature extraction.</span>
        </div>
        <div class="guide-step">
            <strong style="color: #FFFFFF;">4. Execution:</strong> 
            <span style="color: #A1A1AA;">Upon capturing the image, the Vantedge engine will autonomously generate localized bounding boxes, rich segmentation masks, and precise bin routing directives based on municipal regulations.</span>
        </div>
        """, unsafe_allow_html=True)
