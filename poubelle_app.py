import streamlit as st
import os
import cv2
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# -----------------------
# Config Streamlit
# -----------------------
st.set_page_config(
    page_title="Poubelle Detection",
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
# CSS moderne
# -----------------------
st.markdown("""
<style>
.header { text-align: center; background-color: #1E40AF; color: white; padding: 30px; border-radius: 10px; margin-bottom: 20px; }
.upload-area { border: 2px dashed #1E40AF; border-radius: 10px; padding: 50px; text-align: center; margin-bottom: 20px; background-color: #eff6ff; font-size: 18px; color: #1E3A8A;}
.button-row button { margin-right: 10px; }
.card { background-color: #eff6ff; padding: 15px; border-radius: 10px; margin-bottom: 10px; }
.alert-red { background-color: #EF4444; color:white; padding:10px; border-radius:10px; }
.alert-blue { background-color: #3B82F6; color:white; padding:10px; border-radius:10px; }
.footer { text-align:center; color:gray; margin-top:30px; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="header"><h1>Poubelle Detection</h1><p>Gestion intelligente des poubelles</p></div>', unsafe_allow_html=True)

# Upload area
uploaded_files = st.file_uploader(
    "Glisser / Déposer vos fichiers ici ou cliquer pour sélectionner", 
    accept_multiple_files=True, type=["jpg","jpeg","png","mp4"], key="uploader"
)

# Boutons
col1, col2, col3 = st.columns([1,1,1])
predict_btn = col1.button("Prédire")
reset_btn = col2.button("Réinitialiser")
download_btn = col3.button("Télécharger le modèle")

recipient_email = st.text_input("Email pour alertes (poubelle pleine)", "")

# Réinitialiser
if reset_btn:
    st.session_state.history = []
    st.success("Historique réinitialisé")

# Télécharger le modèle
if download_btn:
    if os.path.exists(MODEL_FILENAME):
        with open(MODEL_FILENAME, "rb") as f:
            model_bytes = f.read()
        st.download_button("Télécharger le modèle", data=model_bytes, file_name="model_Poubelle.h5")
    else:
        st.warning("Le fichier modèle n'existe pas.")

# Prédire
if predict_btn and uploaded_files:
    for f in uploaded_files:
        path = os.path.join(UPLOAD_FOLDER, f.name)
        with open(path,"wb") as out:
            out.write(f.read())

        if f.type.startswith("image"):
            cls, conf = predict_image_file(path)
            ftype = "image"
            st.image(path, caption=f.name, use_column_width=True)
        elif f.type.startswith("video"):
            cls, conf = predict_video_file(path)
            ftype = "video"
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

# Statistiques
if st.session_state.history:
    total = len(st.session_state.history)
    pleines = sum(1 for h in st.session_state.history if h["result"]=="poubelle_pleine")
    vides = total - pleines

    st.subheader("Statistiques")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total", total)
    col2.metric("Pleines", pleines)
    col3.metric("Vides", vides)

    st.bar_chart({"Pleines": [pleines], "Vides": [vides]})

    # Historique
    st.subheader("Historique des prédictions")
    for h in st.session_state.history[::-1]:
        color = "#EF4444" if h["result"]=="poubelle_pleine" else "#3B82F6"
        st.markdown(f"""
        <div class="card" style="border-left:5px solid {color};">
            <b>{h['filename']}</b><br>
            Résultat: {h['result']}<br>
            Confiance: {h['confidence']*100:.2f}%<br>
            Type: {h['type']}<br>
            Fichier: {h['filename']}
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">Poubelle Detection © 2025</div>', unsafe_allow_html=True)
