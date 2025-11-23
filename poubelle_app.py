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

MODEL_FILENAME = "poubelle_modell1.h5"
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
.card { background-color: #eff6ff; padding: 15px; border-radius: 10px; margin-bottom: 10px; box-shadow: 2px 2px 8px rgba(0,0,0,0.1);}
.alert-red { background-color: #EF4444; color:white; padding:10px; border-radius:10px; }
.alert-blue { background-color: #3B82F6; color:white; padding:10px; border-radius:10px; }
.footer { text-align:center; color:gray; margin-top:30px; }
.button-style { background-color:#1E40AF; color:white; border:none; padding:10px 20px; border-radius:5px; font-weight:bold; cursor:pointer; margin-right:5px;}
.button-style:hover { background-color:#2563EB; }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Sidebar Navigation
# -----------------------
page = st.sidebar.selectbox("Navigation", ["Accueil", "Statistiques", "Télécharger le modèle"])

# -----------------------
# Page Accueil
# -----------------------
if page == "Accueil":
    st.markdown('<div class="header"><h1>Poubelle Detection</h1><p>Gestion intelligente des poubelles</p></div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Glisser / Déposer vos fichiers ici ou cliquer pour sélectionner", 
        accept_multiple_files=True, type=["jpg","jpeg","png","mp4"]
    )

    col1, col2 = st.columns([1,1])
    predict_btn = col1.button("🖼️ Prédire")
    reset_btn = col2.button("♻️ Réinitialiser")

    recipient_email = st.text_input("Email pour alertes (poubelle pleine)", "")

    if reset_btn:
        st.session_state.history = []
        st.success("Historique réinitialisé")

    if predict_btn and uploaded_files:
        for f in uploaded_files:
            path = os.path.join(UPLOAD_FOLDER, f.name)
            with open(path,"wb") as out:
                out.write(f.read())

            if f.type.startswith("image"):
                cls, conf = predict_image_file(path)
                ftype = "image"
            elif f.type.startswith("video"):
                cls, conf = predict_video_file(path)
                ftype = "video"
            else:
                continue

            st.session_state.history.append({
                "filename": f.name,
                "type": ftype,
                "result": cls,
                "confidence": conf,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

            # Affichage côte à côte
            col_img, col_res = st.columns([2,3])
            with col_img:
                if ftype=="image":
                    st.image(path, use_column_width=True)
                else:
                    st.video(path)
            with col_res:
                color = "#EF4444" if cls=="poubelle_pleine" else "#3B82F6"
                st.markdown(f"""
                <div class="card" style="border-left:5px solid {color};">
                    <b>{f.name}</b><br>
                    Résultat: {cls}<br>
                    Confiance: {conf*100:.2f}%<br>
                    Type: {ftype}<br>
                    Fichier: {f.name}
                </div>
                """, unsafe_allow_html=True)

# -----------------------
# Page Statistiques
# -----------------------
elif page == "Statistiques":
    st.subheader("📊 Statistiques")
    if st.session_state.history:
        total = len(st.session_state.history)
        pleines = sum(1 for h in st.session_state.history if h["result"]=="poubelle_pleine")
        vides = total - pleines

        col1, col2, col3 = st.columns(3)
        col1.metric("Total", total)
        col2.metric("Pleines", pleines)
        col3.metric("Vides", vides)

        # Bar chart simple
        st.bar_chart({"Pleines": [pleines], "Vides": [vides]})

        # Confiance moyenne
        if pleines>0:
            conf_pleines = np.mean([h["confidence"] for h in st.session_state.history if h["result"]=="poubelle_pleine"])
        else: conf_pleines = 0.0
        if vides>0:
            conf_vides = np.mean([h["confidence"] for h in st.session_state.history if h["result"]=="poubelle_vide"])
        else: conf_vides = 0.0

        st.subheader("Confiance moyenne (%)")
        col1, col2 = st.columns(2)
        col1.metric("Poubelles Pleines", f"{conf_pleines*100:.2f}%")
        col2.metric("Poubelles Vides", f"{conf_vides*100:.2f}%")

        # Historique trié par plus récent
        st.subheader("Historique des prédictions")
        for h in sorted(st.session_state.history, key=lambda x:x["timestamp"], reverse=True):
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
    else:
        st.info("Aucune prédiction pour le moment.")

# -----------------------
# Page Télécharger le modèle
# -----------------------
elif page == "Télécharger le modèle":
    st.subheader("⬇️ Télécharger le modèle")
    if os.path.exists(MODEL_FILENAME):
        with open(MODEL_FILENAME, "rb") as f:
            model_bytes = f.read()
        st.download_button("💾 Télécharger le modèle", data=model_bytes, file_name="model_Poubelle.h5")
    else:
        st.warning("Le fichier modèle n'existe pas.")

# Footer
st.markdown('<div class="footer">Poubelle Detection © 2025</div>', unsafe_allow_html=True)
