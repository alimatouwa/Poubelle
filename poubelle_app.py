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
import pandas as pd

# -----------------------
# Config Streamlit
# -----------------------
st.set_page_config(
    page_title="SmartBin Poubelles",
    page_icon="🗑️",
    layout="wide"
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_FILENAME = "poubelle_modell.h5"
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
# CSS pour design dashboard bleu
# -----------------------
st.markdown("""
<style>
.header {
    background: linear-gradient(90deg, #1E3C72, #2A5298);
    padding: 25px;
    border-radius: 15px;
    color: white;
    text-align: center;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    box-shadow: 0 6px 15px rgba(0,0,0,0.2);
}
.card {
    background-color: #f0f4f8;
    border-radius: 15px;
    padding: 20px;
    margin: 10px 0;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
.alert-red {
    background-color: #FF4B4B;
    color: white;
    padding: 12px;
    border-radius: 10px;
    font-weight: bold;
    text-align: center;
    margin: 10px 0;
}
.alert-green {
    background-color: #4BB543;
    color: white;
    padding: 12px;
    border-radius: 10px;
    font-weight: bold;
    text-align: center;
    margin: 10px 0;
}
.download-btn {
    background-color: #1E3C72;
    color: white;
    font-weight: bold;
    border-radius: 8px;
    padding: 8px 16px;
}
</style>
<div class="header">
    <h1>🗑️ SmartBin Pro</h1>
    <p>Détection intelligente des poubelles pleines et vides</p>
</div>
""", unsafe_allow_html=True)

# -----------------------
# Upload
# -----------------------
st.subheader("📤 Upload images ou vidéos")
uploaded_files = st.file_uploader(
    "Sélectionnez vos fichiers", accept_multiple_files=True, type=["jpg","jpeg","png","mp4"]
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
        label="Télécharger le modèle",
        data=model_bytes,
        file_name="model_MobileNetV2.h5",
        mime="application/octet-stream"
    )
else:
    st.warning("Le fichier modèle n'existe pas.")

# -----------------------
# Statistiques
# -----------------------
if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    total = len(df)
    pleines = df[df["result"]=="poubelle_pleine"].shape[0]
    vides = total - pleines

    st.subheader("📊 Statistiques")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total fichiers", total)
    col2.metric("Poubelles Pleines", pleines)
    col3.metric("Poubelles Vides", vides)

    st.subheader("📝 Historique")
    df_display = df.copy()
    df_display["Confidence"] = df_display["confidence"].apply(lambda x: f"{x*100:.1f}%")
    df_display = df_display.rename(columns={"filename": "Fichier", "type": "Type", "result": "Résultat", "timestamp": "Horodatage"})
    
    st.dataframe(df_display[["Horodatage","Type","Fichier","Résultat","Confidence"]].sort_values(by="Horodatage", ascending=False), use_container_width=True)
