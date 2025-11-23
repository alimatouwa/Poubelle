import streamlit as st
import os
import cv2
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# -----------------------
# Config Streamlit
# -----------------------
st.set_page_config(
    page_title="SmartBin Pro",
    page_icon="🗑️",
    layout="wide"
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_FILENAME = "poubelle_modell.h5"  # chemin local vers ton modèle
CLASSES = ["poubelle_vide", "poubelle_pleine"]

# -----------------------
# Charger ou construire modèle
# -----------------------
def build_mobilenet_model(num_classes=2):
    base = MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')
    base.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(base.input, out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if os.path.exists(MODEL_FILENAME):
    model = load_model(MODEL_FILENAME)
else:
    model = build_mobilenet_model()

# -----------------------
# Fonctions de prédiction
# -----------------------
def predict_image_file(path):
    from tensorflow.keras.preprocessing import image
    img = image.load_img(path, target_size=(224,224))
    arr = image.img_to_array(img)/255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr)[0]
    idx = int(np.argmax(preds))
    return CLASSES[idx], float(preds[idx])

def predict_frame(frame):
    frm = cv2.resize(frame, (224,224))
    arr = frm.astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr)[0]
    idx = int(np.argmax(preds))
    return CLASSES[idx], float(preds[idx])

def predict_video_file(path, sample_rate=5):
    cap = cv2.VideoCapture(path)
    results = []
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % sample_rate == 0:
            cls, conf = predict_frame(frame)
            results.append((cls, conf))
        i += 1
    cap.release()
    if not results:
        return "poubelle_vide", 0.0
    classes, confs = zip(*results)
    final = max(set(classes), key=classes.count)
    avg_conf = float(np.mean(confs))
    return final, avg_conf

# -----------------------
# Historique
# -----------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------
# Email (SMTP)
# -----------------------
def send_email_alert(subject, body, recipient):
    try:
        sender = os.environ.get("MAIL_USERNAME")
        password = os.environ.get("MAIL_PASSWORD")
        smtp_server = os.environ.get("MAIL_SERVER", "smtp.gmail.com")
        smtp_port = int(os.environ.get("MAIL_PORT", 587))
        
        msg = MIMEMultipart()
        msg['From'] = sender
        msg['To'] = recipient
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)
        server.quit()
    except Exception as e:
        st.warning(f"Impossible d'envoyer l'email: {e}")

# -----------------------
# CSS pour design
# -----------------------
st.markdown("""
<style>
.header {
    background-color: #2E8B57;
    padding: 15px;
    border-radius: 10px;
    color: white;
    text-align: center;
}
.card {
    background-color: #f5f5f5;
    border-radius: 10px;
    padding: 15px;
    text-align: center;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
}
.alert-red {
    background-color: #FF6347;
    color: white;
    padding: 10px;
    border-radius: 10px;
    font-weight: bold;
}
.alert-green {
    background-color: #32CD32;
    color: white;
    padding: 10px;
    border-radius: 10px;
    font-weight: bold;
}
</style>
<div class="header">
    <h1>🗑️ SmartBin Pro</h1>
    <p>Détection intelligente des poubelles pleines et vides</p>
</div>
""", unsafe_allow_html=True)

# -----------------------
# Upload images ou vidéos
# -----------------------
st.subheader("📤 Upload images ou vidéos")
uploaded_files = st.file_uploader(
    "Sélectionnez des fichiers", accept_multiple_files=True, type=["jpg","jpeg","png","mp4"]
)
recipient_email = st.text_input("Email pour alertes (poubelle pleine)", "")

if uploaded_files:
    for f in uploaded_files:
        path = os.path.join(UPLOAD_FOLDER, f.name)
        with open(path,"wb") as out:
            out.write(f.read())
        
        if f.type.startswith("image"):
            cls, conf = predict_image_file(path)
            ftype = "Image"
            st.image(path, caption=f.name, use_column_width=True)
        elif f.type.startswith("video"):
            cls, conf = predict_video_file(path)
            ftype = "Vidéo"
            st.video(path)
        else:
            continue

        st.session_state.history.append({
            "filename": f.name,
            "type": ftype,
            "result": cls,
            "confidence": conf,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        if cls == "poubelle_pleine":
            st.markdown(f'<div class="alert-red">{ftype} {f.name} → {cls} ({conf*100:.1f}%)</div>', unsafe_allow_html=True)
            if recipient_email:
                send_email_alert(
                    "Alerte SmartBin: Poubelle pleine",
                    f"La poubelle est pleine pour le fichier {f.name} (confiance {conf*100:.1f}%)",
                    recipient_email
                )
        else:
            st.markdown(f'<div class="alert-green">{ftype} {f.name} → {cls} ({conf*100:.1f}%)</div>', unsafe_allow_html=True)

# -----------------------
# Télécharger modèle
# -----------------------
st.subheader("⬇️ Télécharger le modèle")
if os.path.exists(MODEL_FILENAME):
    with open(MODEL_FILENAME, "rb") as f:
        model_bytes = f.read()
    st.download_button(
        label="Télécharger le modèle MobileNetV2",
        data=model_bytes,
        file_name="model_MobileNetV2.h5",
        mime="application/octet-stream"
    )
else:
    st.warning("Le fichier modèle n'existe pas.")

# -----------------------
# Statistiques et historique
# -----------------------
if st.session_state.history:
    total = len(st.session_state.history)
    pleines = sum(1 for h in st.session_state.history if h["result"]=="poubelle_pleine")
    vides = total - pleines

    st.subheader("📊 Statistiques")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total fichiers", total)
    col2.metric("Poubelles Pleines", pleines)
    col3.metric("Poubelles Vides", vides)

    st.subheader("📝 Historique")
    for h in st.session_state.history[::-1]:
        color = "#FF6347" if h["result"]=="poubelle_pleine" else "#32CD32"
        st.markdown(f'<div style="background-color:{color};color:white;padding:5px;border-radius:5px;margin-bottom:3px;">{h["timestamp"]} - {h["type"]} {h["filename"]} → {h["result"]} ({h["confidence"]*100:.1f}%)</div>', unsafe_allow_html=True)
