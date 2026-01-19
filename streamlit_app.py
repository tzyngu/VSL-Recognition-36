import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import pickle
import json
import tempfile
import os
import base64

# ===== 1. CẤU HÌNH TRANG =====
st.set_page_config(
    page_title="VSL AI - Professional",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===== 2. HÀM HỖ TRỢ HIỂN THỊ ẢNH TRONG HTML (CHO BANNER TRẮNG) =====
def get_base64_of_bin_file(bin_file):
    """Chuyển đổi ảnh sang base64 để nhúng vào HTML"""
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except:
        return ""

# ===== 3. CSS=====
# ===== 3. CSS (LUXURY & PROFESSIONAL THEME) =====
st.markdown("""
<style>
    /* NHÚNG FONT CHỮ HIỆN ĐẠI */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@300;400;600&display=swap');

    /* === 1. NỀN & CẤU TRÚC CHUNG === */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top left, #1a1a1a, #050505 80%);
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }
    
    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0);
        visibility: hidden; /* Ẩn header mặc định */
    }

    /* === 2. HEADER CUSTOM (GLASSMORPHISM) === */
    .glass-header {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        padding: 1.5rem 3rem;
        border-radius: 0 0 20px 20px;
        margin-bottom: 3rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
    }
    
    .header-branding {
        display: flex;
        align-items: center;
        gap: 1.5rem;
    }
    
    .header-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 2rem;
        background: linear-gradient(90deg, #ffffff, #a0a0a0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 1px;
        margin: 0;
    }
    
    .header-badge {
        background: rgba(255, 215, 0, 0.1);
        color: #ffd700;
        border: 1px solid rgba(255, 215, 0, 0.2);
        padding: 0.3rem 0.8rem;
        border-radius: 50px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    /* === 3. TYPOGRAPHY & ELEMENTS === */
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    p, .stMarkdown, .stText {
        color: #cccccc !important;
        font-weight: 300;
        line-height: 1.6;
    }

    /* Input Labels */
    .stFileUploader label {
        color: #ffffff !important;
        font-family: 'Outfit', sans-serif;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
    }

    /* === 4. NÚT BẤM (BUTTONS) === */
    div.stButton > button {
        background: linear-gradient(135deg, #ffffff, #dedede);
        color: #000000 !important;
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 8px;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 255, 255, 0.1);
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(255, 255, 255, 0.2);
        background: #ffffff;
    }
    
    div.stButton > button:disabled {
        background: #333333;
        color: #666666 !important;
        box-shadow: none;
    }

    /* === 5. KẾT QUẢ (RESULT CARD) === */
    .result-container {
        background: linear-gradient(160deg, rgba(30, 30, 30, 0.6), rgba(10, 10, 10, 0.8));
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 2.5rem;
        text-align: center;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    /* Hiệu ứng glow nhẹ ở nền */
    .result-container::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.03) 0%, transparent 60%);
        pointer-events: none;
    }

    .prediction-label {
        font-family: 'Outfit', sans-serif;
        font-size: 3.5rem;
        font-weight: 800;
        margin: 1rem 0;
        background: linear-gradient(135deg, #fff, #b4b4b4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(255, 255, 255, 0.1);
    }
    
    .confidence-badge {
        display: inline-block;
        background: rgba(0, 230, 118, 0.1);
        color: #00e676;
        border: 1px solid rgba(0, 230, 118, 0.2);
        padding: 0.4rem 1.2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.9rem;
        letter-spacing: 0.5px;
    }

    /* === 6. THANH TIẾN TRÌNH === */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00e676, #00b0ff);
        border-radius: 10px;
    }

    /* === 7. FOOTER === */
    .footer {
        text-align: center;
        padding-top: 3rem;
        padding-bottom: 2rem;
        color: #444;
        font-size: 0.85rem;
        font-weight: 300;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# ===== 4. LOGIC XỬ LÝ (GIỮ NGUYÊN) =====
mp_hands = None
MEDIAPIPE_AVAILABLE = False
try:
    import mediapipe as mp
    if hasattr(mp, 'solutions'):
        mp_hands = mp.solutions.hands
        MEDIAPIPE_AVAILABLE = True
except Exception:
    pass

def extract_hand_keypoints(results):
    left_hand = np.zeros(21 * 3)
    right_hand = np.zeros(21 * 3)
    hands_detected = 0
    if results.multi_hand_landmarks:
        hands_detected = len(results.multi_hand_landmarks)
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[idx].classification[0].label
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            if handedness == 'Left': right_hand = landmarks 
            else: left_hand = landmarks
    return np.concatenate([left_hand, right_hand]), hands_detected

@st.cache_resource
def load_model():
    try:
        try: model = tf.keras.models.load_model('models/vsl_model.h5', compile=False)
        except: 
            try: model = tf.keras.models.load_model('models/vsl_model.keras', compile=False)
            except: model = tf.keras.models.load_model('models/best_model.keras', compile=False)
        
        scaler = None
        try:
            with open('models/scaler.pkl', 'rb') as f: scaler = pickle.load(f)
        except: pass
        
        with open('models/label_encoder.pkl', 'rb') as f: label_encoder = pickle.load(f)
        with open('models/config.json', 'r') as f: config = json.load(f)
        return model, scaler, label_encoder, config, None
    except Exception as e: return None, None, None, None, str(e)

def process_video_with_debug(video_path, sequence_length, feature_size, scaler):
    debug_info = {"frames_processed": 0, "hands_detected_frames": 0, "normalized": False}
    if not MEDIAPIPE_AVAILABLE: return np.zeros((sequence_length, feature_size)), debug_info
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1: return None, debug_info
    
    frame_indices = np.linspace(0, total_frames - 1, sequence_length, dtype=int) if total_frames >= sequence_length else list(range(total_frames))
    keypoints_sequence = []
    
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3, min_tracking_confidence=0.3) as hands:
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret: 
                keypoints_sequence.append(np.zeros(feature_size))
                continue
            
            debug_info["frames_processed"] += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            kps, cnt = extract_hand_keypoints(results)
            keypoints_sequence.append(kps)
            if cnt > 0: debug_info["hands_detected_frames"] += 1
            
    cap.release()
    
    while len(keypoints_sequence) < sequence_length:
        keypoints_sequence.append(keypoints_sequence[-1] if keypoints_sequence else np.zeros(feature_size))
    
    sequence = np.array(keypoints_sequence[:sequence_length])
    
    if scaler:
        sequence = scaler.transform(sequence.reshape(-1, feature_size)).reshape(sequence_length, feature_size)
        debug_info["normalized"] = True
        
    return sequence, debug_info

# ===== 5. UI CHÍNH (MAIN) =====
def main():
    # --- HEADER SECTION ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(current_dir, "logo.png")
    
    # Logo Handling
    logo_html = ""
    if os.path.exists(logo_path):
        b64_logo = get_base64_of_bin_file(logo_path)
        logo_html = f'<img src="data:image/jpeg;base64,{b64_logo}" style="height: 50px; width: auto; opacity: 0.9;">'
    
    st.markdown(f"""
        <div class="glass-header">
            <div class="header-branding">
                {logo_html}
                <h1 class="header-title">VSL RECOGNITION</h1>
            </div>
            <div class="header-badge">AI CORE v2.0</div>
        </div>
    """, unsafe_allow_html=True)

    # Load Model
    model, scaler, label_encoder, config, error = load_model()
    if model is None:
        st.error(f"⚠️ SYSTEM ERROR: Unable to load AI Model.\nDetails: {error}")
        st.stop()

    # --- MAIN GRID ---
    # Sử dụng 3 cột để tạo padding 2 bên nếu cần, hoặc layout 6:6
    col1, col2 = st.columns([1.2, 1], gap="large")

    # --- LEFT COLUMN: INPUT ---
    with col1:
        st.markdown("###INPUT SOURCE")
        st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True) # Spacer
        
        uploaded = st.file_uploader("Upload Video File", type=['mp4', 'avi', 'mov', 'webm'], help="Supports MP4, AVI, MOV", label_visibility="visible")
        
        if uploaded:
            st.session_state['uploaded_file'] = True
            # Hiển thị video với bo góc
            st.markdown('<div style="border-radius: 12px; overflow: hidden; border: 1px solid rgba(255,255,255,0.1);">', unsafe_allow_html=True)
            st.video(uploaded)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Placeholder Box nếu user muốn thêm ảnh minh họa ở đây sau này
            st.markdown("""
            <div style="
                border: 2px dashed rgba(255,255,255,0.1); 
                border-radius: 12px; 
                padding: 40px; 
                text-align: center; 
                color: #555;">
                <p>Waiting for video stream...</p>
                <small>Select a file to begin analysis</small>
            </div>
            """, unsafe_allow_html=True)

    # --- RIGHT COLUMN: ANALYSIS ---
    with col2:
        st.markdown("### REAL-TIME ANALYSIS")
        st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True) # Spacer

        # Button Group
        start_col, _ = st.columns([1, 1])
        with start_col:
            predict_btn = st.button("INITIALIZE ANALYSIS", disabled=(uploaded is None), use_container_width=True)

        if uploaded and predict_btn:
            with st.spinner("Processing biometric data..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name
                
                seq, debug = process_video_with_debug(
                    tmp_path, config.get('sequence_length', 20), config.get('feature_size', 126), scaler
                )
                
                if seq is not None:
                    # Logic dự đoán
                    if debug.get('hands_detected_frames', 0) == 0:
                        st.warning("⚠️ No hand landmarks detected in sequence.")
                    
                    pred = model.predict(np.expand_dims(seq, axis=0), verbose=0)
                    idx = np.argmax(pred)
                    label = label_encoder.inverse_transform([idx])[0]
                    confidence = pred[0][idx]
                    
                    # --- DISPLAY RESULTS (NEW CARD) ---
                    st.markdown(f"""
                    <div class="result-container">
                        <p style="color:#888; letter-spacing: 2px; font-size: 0.8rem; text-transform: uppercase;">Predicted Gesture</p>
                        <div class="prediction-label">{label.upper()}</div>
                        <div class="confidence-badge">CONFIDENCE: {confidence:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # --- TOP CANDIDATES ---
                    st.markdown("<div style='margin-top: 2rem; margin-bottom: 1rem; font-size: 0.9rem; color: #888;'>ALTERNATIVE PREDICTIONS</div>", unsafe_allow_html=True)
                    
                    top3 = np.argsort(pred[0])[-3:][::-1]
                    for i in top3:
                        if i == idx: continue
                        lbl_name = label_encoder.inverse_transform([i])[0]
                        score = float(pred[0][i])
                        
                        # Custom bar layout
                        c1, c2 = st.columns([3, 1])
                        with c1:
                            st.progress(score)
                        with c2:
                            st.caption(f"{lbl_name} ({score:.0%})")

                # Debug Info (Tối giản hơn)
                with st.expander("Technical Telemetry"):
                    st.json(debug)
    
    # --- FOOTER ---
    st.markdown("""
        <div class="footer">
            VSL INTELLIGENCE SYSTEM &copy; 2026<br>
            <span style="opacity: 0.5; font-size: 0.7rem;">Powered by TensorFlow & MediaPipe</span>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()