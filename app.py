import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import time

# -----------------------------------------------------------------------------
# 1. Luxury Page & UI Architecture (Responsive CSS)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="VANTEDGE | Vision Console",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Deep Obsidian, Gold, & Rich Metal Theme with Mobile Responsiveness
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap');

    .stApp {
        background-color: #000000;
        color: #E0E0E0;
        font-family: 'Poppins', sans-serif;
    }
    
    /* VANTEDGE Header Customization */
    h1, h2, h3 {
        color: #FFFFFF;
        font-weight: 300;
        letter-spacing: -0.01em;
    }
    h1.luxury-head {
        text-transform: uppercase;
        letter-spacing: 0.15em;
        background: linear-gradient(135deg, #E6C657 0%, #B6922E 50%, #E6C657 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600;
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
        line-height: 1.2;
    }
    .luxury-subhead {
        color: #A1A1AA;
        font-size: 1rem;
        margin-top: 0;
        margin-bottom: 2rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* Container Styling */
    .element-container, div[data-testid="stVerticalBlock"] > div {
        background: transparent;
    }
    
    /* Reference Cards Grid */
    .integrated-ref-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 2rem;
    }
    
    .ref-card {
        padding: 1rem;
        border-radius: 8px;
        border-top: 4px solid;
        background-color: #0A0A0A;
        border-left: 1px solid #1A1A1A;
        border-right: 1px solid #1A1A1A;
        border-bottom: 1px solid #1A1A1A;
    }
    
    .ref-title { font-weight: 600; font-size: 1rem; margin-bottom: 0.25rem; }
    .ref-desc { font-size: 0.75rem; color: #D4D4D8; line-height: 1.4; }

    /* Mobile Responsiveness Rules */
    @media (max-width: 1024px) {
        .integrated-ref-grid { grid-template-columns: repeat(2, 1fr); }
    }
    @media (max-width: 768px) {
        h1.luxury-head { 
            font-size: 2rem; 
            letter-spacing: 0.1em;
        }
        .luxury-subhead { font-size: 0.85rem; }
    }
    @media (max-width: 480px) {
        h1.luxury-head { font-size: 1.6rem; }
        .integrated-ref-grid { grid-template-columns: 1fr; }
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Rich Color Palette & Class Mapping
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# 3. Hybrid AI Engine (Supports Both Segmentation & Bounding Boxes)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_engine(weights="best.pt"):
    try:
        return YOLO(weights)
    except Exception as e:
        st.error(f"Failed to load neural engine: {e}")
        return None

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
            label_id = int(box.cls[0])
            label_name = engine.names.get(label_id, f"Class {label_id}")
            conf = float(box.conf[0])
            
            bin_category = CLASS_TO_BIN.get(label_name.lower(), "Black")
            color_scheme = BIN_COLOR_MAP[bin_category]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # If segmentation masks are available, use pixel-accurate polygon mask
            if masks is not None and len(masks) > i:
                m_data = masks[i].data[0].cpu().numpy()
                m_scaled = cv2.resize(m_data, (w, h))
                bool_mask = m_scaled > 0.5
                mask_canvas[bool_mask] = color_scheme['dark']
            else:
                # Fallback: fill the bounding box region with a dark tinted mask
                cv2.rectangle(mask_canvas, (x1, y1), (x2, y2), color_scheme['dark'], -1)
            
            detection_details.append({
                "label": label_name,
                "conf": conf,
                "bin": bin_category,
                "bright_color": color_scheme['bright'],
                "hex": color_scheme['hex'],
                "coords": (x1, y1, x2, y2)
            })

        # Blend the rich dark mask layer over the original image
        cv2.addWeighted(mask_canvas, 0.65, img_cv, 0.35, 0, img_cv)

        # Draw sharp bounding boxes and modern badges over top
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
# 4. Integrated High-Tech Interface Layout
# -----------------------------------------------------------------------------

# VANTEDGE Branding
st.markdown("<h1 class='luxury-head'>VANTEDGE</h1>", unsafe_allow_html=True)
st.markdown("<p class='luxury-subhead'>Vision Waste Segregation Engine</p>", unsafe_allow_html=True)

with st.container():
    st.write("---")
    
    # Responsive Guide Grid
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
    
    # Input Logic with Camera Toggle
    m1, m2 = st.columns([1, 1])
    with m1:
        input_mode = st.radio("Initializing Input Sensor", ["Static File", "Live Stream"], horizontal=True, label_visibility="collapsed")
    
    raw_img = None
    if input_mode == "Static File":
        up_file = st.file_uploader("Insert Image", type=["jpg", "jpeg", "png", "webp"])
        if up_file:
            raw_img = Image.open(up_file).convert("RGB")
    else:
        # Front / Back Camera Switch for iPad & Mobile
        cam_facing = st.radio("Camera Source", ["Back Camera (Environment)", "Front Camera (User)"], horizontal=True)
        mode = "environment" if "Back" in cam_facing else "user"
        
        cam_file = st.camera_input("Activate Sensor", facing_mode=mode)
        if cam_file:
            raw_img = Image.open(cam_file).convert("RGB")
            
    if raw_img is not None and engine is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<p style='font-size: 0.8rem; color: #A1A1AA; text-transform: uppercase;'>Sensor Input</p>", unsafe_allow_html=True)
            st.image(raw_img, use_container_width=True)
            
        with col2:
            st.markdown("<p style='font-size: 0.8rem; color: #A1A1AA; text-transform: uppercase;'>Processed Analysis</p>", unsafe_allow_html=True)
            with st.spinner("Processing via Vantedge AI..."):
                start_t = time.time()
                y_results = engine.predict(source=np.array(raw_img), conf=0.45, verbose=False)
                
                processed_img, det_list = draw_luxury_segmentation(raw_img, y_results[0])
                st.image(processed_img, use_container_width=True)
                
        # Segregation Directive
        st.markdown("---")
        if det_list:
            st.markdown("<h3 style='font-size: 1.1rem; color: #FFFFFF;'>Segregation Directive</h3>", unsafe_allow_html=True)
            for det in det_list:
                st.markdown(
                    f"<div style='padding: 12px; border-radius: 8px; border: 1px solid #1A1A1A; background-color: #0A0A0A; margin-bottom: 8px; color: #E0E0E0;'>"
                    f"Route detected <strong style='color: #FFFFFF;'>{det['label'].capitalize()}</strong> "
                    f"to <strong style='color: {det['hex']}; border-bottom: 1px solid {det['hex']};'>{det['bin']} Bin</strong>"
                    f"</div>", 
                    unsafe_allow_html=True
                )
        else:
            st.info("No objects identified above confidence threshold.")
