import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
import atexit
from datetime import datetime
import pandas as pd
from ultralytics import YOLO
from fast_plate_ocr import LicensePlateRecognizer
import logging
from pathlib import Path
import json
from difflib import SequenceMatcher

# Tentative d'importation de streamlit_webrtc avec gestion d'erreur
try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, VideoFrame
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    st.warning("⚠️ streamlit-webrtc n'est pas disponible. La détection en temps réel sera désactivée.")

# ===============================
# Constantes pour les plaques algériennes
# ===============================

WILAYAS = {
    "01": "Adrar", "02": "Chlef", "03": "Laghouat", "04": "Oum El Bouaghi",
    "05": "Batna", "06": "Béjaïa", "07": "Biskra", "08": "Béchar",
    "09": "Blida", "10": "Bouira", "11": "Tamanrasset", "12": "Tébessa",
    "13": "Tlemcen", "14": "Tiaret", "15": "Tizi Ouzou", "16": "Alger",
    "17": "Djelfa", "18": "Jijel", "19": "Sétif", "20": "Saïda",
    "21": "Skikda", "22": "Sidi Bel Abbès", "23": "Annaba", "24": "Guelma",
    "25": "Constantine", "26": "Médéa", "27": "Mostaganem", "28": "M\u2019Sila",
    "29": "Mascara", "30": "Ouargla", "31": "Oran", "32": "El Bayadh",
    "33": "Illizi", "34": "Bordj Bou Arréridj", "35": "Boumerdès", "36": "El Tarf",
    "37": "Tindouf", "38": "Tissemsilt", "39": "El Oued", "40": "Khenchela",
    "41": "Souk Ahras", "42": "Tipaza", "43": "Mila", "44": "Aïn Defla",
    "45": "Naama", "46": "Aïn Témouchent", "47": "Ghardaïa", "48": "Relizane",
    "49": "Timimoun", "50": "Bordj Badji Mokhtar", "51": "Ouled Djellal",
    "52": "Béni Abbès", "53": "In Salah", "54": "In Guezzam", "55": "Touggourt",
    "56": "Djanet", "57": "El M\u2019Ghair", "58": "El Menia"
}

VEHICLE_CATEGORIES = {
    "1": "véhicules de tourisme (véhicules particuliers)",
    "2": "camions",
    "3": "camionnettes",
    "4": "autocars et autobus",
    "5": "tracteurs routiers",
    "6": "autres tracteurs",
    "7": "véhicules spéciaux",
    "8": "remorques et semi-remorques",
    "9": "motocyclettes (deux (2) roues ou plus)"
}

# ===============================
# Fonctions utilitaires pour les plaques
# ===============================

def clean_plate_text(plate_text: str) -> str:
    """Remplace les tirets par des espaces dans le texte de la plaque."""
    return plate_text.replace("-", " ")

def parse_plate_info(plate_text: str) -> dict:
    """Extrait les informations (numéro de série, catégorie, année, wilaya) d'une plaque algérienne.
    La plaque doit être au format continu (sans tirets) et de 10 ou 11 caractères.
    Ex: '1234511816' (5 chiffres pour le numéro de série) ou '12345621825' (6 chiffres).
    """
    info = {
        "serial_number": None,
        "category": None,
        "year": None,
        "wilaya": None,
        "is_valid": False,
        "error": ""
    }

    # Supprimer les tirets si l'OCR les a inclus par erreur
    plate_text = plate_text.replace("-", "").strip()

    if not plate_text.isdigit():
        info["error"] = "La plaque ne contient pas que des chiffres."
        return info

    if len(plate_text) == 10:  # Format: SSSSSCYW (5 chiffres série, 1 catégorie, 2 année, 2 wilaya)
        serial_number_str = plate_text[0:5]
        category_year_str = plate_text[5:8]
        wilaya_str = plate_text[8:10]
    elif len(plate_text) == 11: # Format: SSSSSSCCYW (6 chiffres série, 1 catégorie, 2 année, 2 wilaya)
        serial_number_str = plate_text[0:6]
        category_year_str = plate_text[6:9]
        wilaya_str = plate_text[9:11]
    else:
        info["error"] = f"Longueur de plaque invalide: {len(plate_text)} caractères. Attendu 10 ou 11."
        return info

    # Validation et extraction
    if (len(serial_number_str) == 5 or len(serial_number_str) == 6) and serial_number_str.isdigit():
        info["serial_number"] = serial_number_str
    else:
        info["error"] = "Format du numéro de série invalide."
        return info

    if len(category_year_str) == 3 and category_year_str.isdigit():
        info["category"] = category_year_str[0]
        info["year"] = category_year_str[1:]
    else:
        info["error"] = "Format de la catégorie/année invalide."
        return info

    if len(wilaya_str) == 2 and wilaya_str.isdigit():
        info["wilaya"] = wilaya_str
    else:
        info["error"] = "Format de la wilaya invalide."
        return info

    # Vérification des valeurs extraites avec les listes fournies
    if info["category"] in VEHICLE_CATEGORIES and info["wilaya"] in WILAYAS:
        info["is_valid"] = True
    else:
        info["error"] = "Catégorie ou Wilaya non reconnue."

    return info

def get_wilaya_name(wilaya_code: str) -> str:
    """Retourne le nom de la wilaya à partir de son code."""
    return WILAYAS.get(wilaya_code, "Inconnu")

def get_vehicle_category_name(category_code: str) -> str:
    """Retourne le nom de la catégorie de véhicule à partir de son code."""
    return VEHICLE_CATEGORIES.get(category_code, "Inconnu")

def similarity(a, b):
    """Calcule la similarité entre deux chaînes de caractères."""
    return SequenceMatcher(None, a, b).ratio()

def deduplicate_plates(detections, similarity_threshold=0.8):
    """
    Déduplique les plaques similaires et garde la meilleure détection pour chaque plaque unique.
    
    Args:
        detections: Liste des détections
        similarity_threshold: Seuil de similarité pour considérer deux plaques comme identiques
    
    Returns:
        Liste des meilleures détections uniques
    """
    if not detections:
        return []
    
    # Grouper les détections similaires
    groups = []
    
    for detection in detections:
        plate_text = detection["text"]
        added_to_group = False
        
        # Chercher un groupe existant avec une plaque similaire
        for group in groups:
            for existing_detection in group:
                if similarity(plate_text, existing_detection["text"]) >= similarity_threshold:
                    group.append(detection)
                    added_to_group = True
                    break
            if added_to_group:
                break
        
        # Si aucun groupe trouvé, créer un nouveau groupe
        if not added_to_group:
            groups.append([detection])
    
    # Pour chaque groupe, sélectionner la meilleure détection
    best_detections = []
    
    for group in groups:
        # Critères de sélection (par ordre de priorité) :
        # 1. Plaque valide (format correct)
        # 2. Confiance la plus élevée
        # 3. Longueur de texte appropriée (10 ou 11 caractères)
        
        valid_detections = [d for d in group if d["parsed_info"]["is_valid"]]
        
        if valid_detections:
            # Prendre la détection valide avec la plus haute confiance
            best_detection = max(valid_detections, key=lambda x: x["confidence"])
        else:
            # Si aucune détection valide, prendre celle avec la plus haute confiance
            # et la longueur la plus proche de 10-11 caractères
            def score_detection(d):
                conf_score = d["confidence"]
                length_score = 1.0 if len(d["text"]) in [10, 11] else 0.5
                return conf_score * length_score
            
            best_detection = max(group, key=score_detection)
        
        best_detections.append(best_detection)
    
    return best_detections

# ===============================
# Configuration et initialisation de l'application Streamlit
# ===============================

# Configuration de la page
st.set_page_config(
    page_title="🚗 Détecteur de Plaques Algériennes",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration des chemins (à adapter selon votre environnement)
CONFIG = {
    "yolo_model_path": "best.pt",
    "onnx_model_path": "last.onnx",
    "plate_config_path": "algerian_plates.yml",
    "confidence_threshold": 0.5,
    "max_video_size_mb": 100,  # Limite de taille vidéo en MB
}

# ===============================
# Fonctions de traitement de l'application
# ===============================

@st.cache_resource
def load_models():
    """Charger les modèles avec cache pour éviter le rechargement"""
    try:
        yolo_model = YOLO(CONFIG["yolo_model_path"])
        plate_recognizer = LicensePlateRecognizer(
            onnx_model_path=CONFIG["onnx_model_path"],
            plate_config_path=CONFIG["plate_config_path"],
        )
        logger.info("Modèles chargés avec succès")
        return yolo_model, plate_recognizer
    except Exception as e:
        logger.error(f"Erreur lors du chargement des modèles: {e}")
        st.error(f"❌ Erreur lors du chargement des modèles: {e}")
        return None, None

# Classe pour le traitement vidéo en temps réel (seulement si webrtc est disponible)
if WEBRTC_AVAILABLE:
    class PlateDetectionProcessor(VideoProcessorBase):
        def __init__(self, yolo_model, plate_recognizer, confidence_threshold):
            self.yolo_model = yolo_model
            self.plate_recognizer = plate_recognizer
            self.confidence_threshold = confidence_threshold
            self.all_detections_in_session = [] # Pour stocker toutes les détections du flux

        def recv(self, frame: VideoFrame) -> VideoFrame:
            img = frame.to_ndarray(format="bgr24")

            processed_img, detections = process_detection(
                img.copy(), self.yolo_model, self.plate_recognizer, self.confidence_threshold
            )
            
            # Sauvegarder les détections pour traitement ultérieur (déduplication)
            if detections:
                self.all_detections_in_session.extend(detections)
                # Note: save_results est appelée après l'arrêt du streamer pour la déduplication

            return VideoFrame.from_ndarray(processed_img, format="bgr24")

def process_detection(frame, yolo_model, plate_recognizer, confidence_threshold):
    """Traiter la détection sur une image (utilisé par tous les modes)"""
    detections = []
    
    try:
        results = yolo_model(frame, conf=confidence_threshold)
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    
                    if conf > confidence_threshold:
                        # Extraire la plaque
                        cropped_plate = frame[y1:y2, x1:x2]
                        
                        # Vérifier que la plaque n'est pas vide
                        if cropped_plate.size == 0:
                            continue
                            
                        # Prétraitement pour l'OCR
                        gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
                        resized = cv2.resize(gray, (128, 64))
                        input_img = np.expand_dims(resized, axis=(0, -1))
                        
                        # Reconnaissance de texte
                        plate_text = plate_recognizer.run(input_img)
                        if isinstance(plate_text, list):
                            plate_text = "".join(plate_text)
                        
                        # Nettoyer le texte et extraire les informations
                        plate_text = str(plate_text).strip()
                        parsed_info = parse_plate_info(plate_text)
                        
                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': conf,
                            'text': plate_text,
                            'parsed_info': parsed_info,
                            'cropped_image': cropped_plate
                        })
                        
                        # Dessiner les annotations
                        # Afficher le texte formaté si valide, sinon le texte brut
                        if parsed_info["is_valid"]:
                            display_text = f"{parsed_info['serial_number']} {parsed_info['category']}{parsed_info['year']} {parsed_info['wilaya']}"
                        else:
                            display_text = plate_text

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Texte avec arrière-plan
                        text = f"{display_text} ({conf:.2f})"
                        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        cv2.rectangle(frame, (x1, y1-30), (x1 + text_size[0], y1), (0, 255, 0), -1)
                        cv2.putText(frame, text, (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    except Exception as e:
        logger.error(f"Erreur lors du traitement: {e}")
        st.warning(f"⚠️ Erreur lors du traitement: {e}")
    
    return frame, detections

def save_results(detections, mode="image"):
    """Sauvegarder les résultats dans la session"""
    if 'detection_history' not in st.session_state:
        st.session_state.detection_history = []
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for detection in detections:
        st.session_state.detection_history.append({
            'timestamp': timestamp,
            'mode': mode,
            'text': detection['text'],
            'confidence': detection['confidence'],
            'parsed_info': detection['parsed_info']
        })

def validate_file_size(uploaded_file, max_size_mb):
    """Valider la taille du fichier uploadé"""
    if uploaded_file is not None:
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        if file_size_mb > max_size_mb:
            st.error(f"❌ Le fichier est trop volumineux ({file_size_mb:.1f} MB). Limite: {max_size_mb} MB")
            return False
    return True

# ===============================
# Interface utilisateur principale
# ===============================

def main():
    # Titre principal
    st.title("🚗 Détection & Reconnaissance de Plaques Algériennes")
    st.markdown("---")
    
    # Sidebar pour les paramètres
    with st.sidebar:
        st.header("⚙️ Paramètres")
        confidence_threshold = st.slider(
            "Seuil de confiance", 
            min_value=0.1, 
            max_value=1.0, 
            value=CONFIG["confidence_threshold"], 
            step=0.05
        )
        
        st.header("📊 Statistiques")
        if 'detection_history' in st.session_state:
            total_detections = len(st.session_state.detection_history)
            st.metric("Détections totales", total_detections)
            
            if total_detections > 0:
                avg_confidence = np.mean([d['confidence'] for d in st.session_state.detection_history])
                st.metric("Confiance moyenne", f"{avg_confidence:.2f}")
        
        # Bouton pour effacer l'historique
        if st.button("🗑️ Effacer l'historique"):
            st.session_state.detection_history = []
            st.success("Historique effacé!")
    
    # Chargement des modèles
    with st.spinner("🔄 Chargement des modèles..."):
        yolo_model, plate_recognizer = load_models()
    
    if yolo_model is None or plate_recognizer is None:
        st.stop()
    
    st.success("✅ Modèles chargés avec succès!")
    
    # Choix du mode (conditionnel pour la webcam)
    if WEBRTC_AVAILABLE:
        mode_options = ["🖼️ Image", "🎥 Vidéo (upload)", "📹 Temps réel (webcam)"]
    else:
        mode_options = ["🖼️ Image", "🎥 Vidéo (upload)"]
        st.info("ℹ️ Le mode temps réel nécessite l'installation de streamlit-webrtc")
    
    mode = st.radio(
        "📋 Choisir un mode :", 
        mode_options,
        horizontal=True
    )
    
    # ===============================
    # MODE IMAGE
    # ===============================
    if mode == "🖼️ Image":
        st.header("🖼️ Traitement d'image")
        
        uploaded_file = st.file_uploader(
            "Uploader une image", 
            type=["jpg", "jpeg", "png"],
            help="Formats supportés: JPG, JPEG, PNG"
        )
        
        if uploaded_file:
            # Validation de la taille
            if not validate_file_size(uploaded_file, 10):  # 10 MB pour les images
                return
            
            try:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                if frame is None:
                    st.error("❌ Impossible de lire l'image. Vérifiez le format.")
                    return
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Image originale")
                    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)
                
                with st.spinner("🔍 Traitement en cours..."):
                    processed_frame, detections = process_detection(
                        frame.copy(), yolo_model, plate_recognizer, confidence_threshold
                    )
                
                with col2:
                    st.subheader("Résultat de détection")
                    st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), use_column_width=True)
                
                # Afficher les résultats
                if detections:
                    st.success(f"✅ {len(detections)} plaque(s) détectée(s)!")
                    
                    for i, detection in enumerate(detections):
                        # Utiliser le texte formaté pour l'expander si valide, sinon le texte brut
                        expander_title = f"Plaque {i+1}: "
                        if detection['parsed_info']['is_valid']:
                            expander_title += f"{detection['parsed_info']['serial_number']} {detection['parsed_info']['category']}{detection['parsed_info']['year']} {detection['parsed_info']['wilaya']}"
                        else:
                            expander_title += detection['text']

                        with st.expander(expander_title):
                            col_info, col_crop = st.columns([2, 1])
                            with col_info:
                                st.write(f"**Texte OCR brut:** {detection['text']}")
                                if detection['parsed_info']['is_valid']:
                                    st.write(f"**Numéro de série:** {detection['parsed_info']['serial_number']}")
                                    st.write(f"**Catégorie de véhicule:** {get_vehicle_category_name(detection['parsed_info']['category'])} ({detection['parsed_info']['category']})")
                                    st.write(f"**Année:** 20{detection['parsed_info']['year']}")
                                    st.write(f"**Wilaya:** {get_wilaya_name(detection['parsed_info']['wilaya'])} ({detection['parsed_info']['wilaya']})")
                                else:
                                    st.error(f"❌ Format de plaque invalide: {detection['parsed_info']['error']}")
                                st.write(f"**Confiance:** {detection['confidence']:.2f}")
                                st.write(f"**Position:** {detection['bbox']}")
                            with col_crop:
                                st.image(cv2.cvtColor(detection['cropped_image'], cv2.COLOR_BGR2RGB))
                    
                    save_results(detections, "image")
                else:
                    st.warning("⚠️ Aucune plaque détectée. Essayez d'ajuster le seuil de confiance.")
                    
            except Exception as e:
                st.error(f"❌ Erreur lors du traitement de l'image: {e}")
    
    # ===============================
    # MODE VIDÉO
    # ===============================
    elif mode == "🎥 Vidéo (upload)":
        st.header("🎥 Traitement de vidéo")
        
        uploaded_file = st.file_uploader(
            "Uploader une vidéo", 
            type=["mp4", "avi", "mov"],
            help=f"Formats supportés: MP4, AVI, MOV (max {CONFIG['max_video_size_mb']} MB)"
        )
        
        if uploaded_file:
            if not validate_file_size(uploaded_file, CONFIG['max_video_size_mb']):
                return
            
            process_video = st.button("🚀 Commencer le traitement")
            
            if process_video:
                try:
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    temp_path = tfile.name
                    tfile.write(uploaded_file.read())
                    tfile.close()  # 🔑 CRUCIAL
                    cap = cv2.VideoCapture(temp_path)

                    if not cap.isOpened():
                        st.error("❌ Impossible d'ouvrir la vidéo")
                        return
                    
                    # Informations sur la vidéo
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = total_frames / fps if fps > 0 else 0
                    
                    st.info(f"📹 Vidéo: {total_frames} frames, {fps} FPS, {duration:.1f}s")
                    
                    stframe = st.empty()
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    frame_count = 0
                    all_detections = []
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        frame_count += 1
                        progress = frame_count / total_frames
                        
                        # Traiter seulement quelques frames pour éviter la surcharge
                        if frame_count % max(1, fps // 2) == 0:  # 2 FPS pour le traitement
                            processed_frame, detections = process_detection(
                                frame.copy(), yolo_model, plate_recognizer, confidence_threshold
                            )
                            all_detections.extend(detections)
                            stframe.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB")
                        
                        progress_bar.progress(progress)
                        status_text.text(f"Frame {frame_count}/{total_frames}")
                        
                        # Arrêt si l'utilisateur ferme l'app
                        if not ret:
                            break
                    
                    cap.release()
                    cap = None
                    os.remove(tfile.name)
                    
                    # Résumé final avec déduplication intelligente
                    if all_detections:
                        st.success(f"✅ Traitement terminé! {len(all_detections)} détections au total")
                        
                        # Appliquer la déduplication intelligente
                        unique_detections = deduplicate_plates(all_detections)
                        
                        st.info(f"🔍 Après déduplication: {len(unique_detections)} plaques uniques détectées")
                        
                        # Afficher les résultats dédupliqués
                        for i, detection in enumerate(unique_detections, 1):
                            with st.expander(f"🚗 Véhicule {i}"):
                                col_info, col_crop = st.columns([2, 1])
                                with col_info:
                                    # Afficher le texte formaté si valide
                                    if detection['parsed_info']['is_valid']:
                                        formatted_text = f"{detection['parsed_info']['serial_number']} {detection['parsed_info']['category']}{detection['parsed_info']['year']} {detection['parsed_info']['wilaya']}"
                                        st.write(f"**Matricule:** {formatted_text}")
                                        st.write(f"**Numéro de série:** {detection['parsed_info']['serial_number']}")
                                        st.write(f"**Catégorie de véhicule:** {get_vehicle_category_name(detection['parsed_info']['category'])} ({detection['parsed_info']['category']})")
                                        st.write(f"**Année:** 20{detection['parsed_info']['year']}")
                                        st.write(f"**Wilaya:** {get_wilaya_name(detection['parsed_info']['wilaya'])} ({detection['parsed_info']['wilaya']})")
                                    else:
                                        st.write(f"**Matricule (brut):** {detection['text']}")
                                        st.error(f"❌ Format invalide: {detection['parsed_info']['error']}")
                                    
                                    st.write(f"**Confiance:** {detection['confidence']:.2f}")
                                    
                                    # Compter les occurrences de cette plaque dans toutes les détections
                                    similar_count = sum(1 for d in all_detections 
                                                      if similarity(d['text'], detection['text']) >= 0.8)
                                    st.write(f"**Détections similaires:** {similar_count} fois")
                                    
                                with col_crop:
                                    st.image(cv2.cvtColor(detection['cropped_image'], cv2.COLOR_BGR2RGB))
                        
                        # Sauvegarder seulement les détections uniques
                        save_results(unique_detections, "video")
                        
                    else:
                        st.warning("⚠️ Aucune plaque détectée dans la vidéo")
                        
                except Exception as e:
                    st.error(f"❌ Erreur lors du traitement de la vidéo: {e}")
    
    # ===============================
    # MODE TEMPS RÉEL (seulement si webrtc disponible)
    # ===============================
    elif mode == "📹 Temps réel (webcam)" and WEBRTC_AVAILABLE:
        st.header("📹 Détection en temps réel")
        
        # Utilisation de streamlit-webrtc
        webrtc_ctx = webrtc_streamer(
            key="plate_detection",
            video_processor_factory=lambda: PlateDetectionProcessor(yolo_model, plate_recognizer, confidence_threshold),
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            media_stream_constraints={
                "video": True,
                "audio": False
            },
            async_processing=True,
        )

        if webrtc_ctx.state.playing:
            st.write("📹 Webcam en cours...")
            st.info("Appuyez sur 'STOP' pour arrêter et voir les résultats")
        elif webrtc_ctx.state.stopped:
            # Récupérer les détections du processeur après l'arrêt du flux
            if webrtc_ctx.video_processor and hasattr(webrtc_ctx.video_processor, 'all_detections_in_session'):
                all_detections_from_webcam = webrtc_ctx.video_processor.all_detections_in_session
                if all_detections_from_webcam:
                    st.success(f"✅ Session terminée! {len(all_detections_from_webcam)} détections au total")
                    unique_detections = deduplicate_plates(all_detections_from_webcam)
                    st.info(f"🔍 Après déduplication: {len(unique_detections)} plaques uniques détectées")
                    
                    for i, detection in enumerate(unique_detections, 1):
                        with st.expander(f"🚗 Véhicule {i}"):
                            col_info, col_crop = st.columns([2, 1])
                            with col_info:
                                if detection['parsed_info']['is_valid']:
                                    formatted_text = f"{detection['parsed_info']['serial_number']} {detection['parsed_info']['category']}{detection['parsed_info']['year']} {detection['parsed_info']['wilaya']}"
                                    st.write(f"**Matricule:** {formatted_text}")
                                    st.write(f"**Numéro de série:** {detection['parsed_info']['serial_number']}")
                                    st.write(f"**Catégorie de véhicule:** {get_vehicle_category_name(detection['parsed_info']['category'])} ({detection['parsed_info']['category']})")
                                    st.write(f"**Année:** 20{detection['parsed_info']['year']}")
                                    st.write(f"**Wilaya:** {get_wilaya_name(detection['parsed_info']['wilaya'])} ({detection['parsed_info']['wilaya']})")
                                else:
                                    st.write(f"**Matricule (brut):** {detection['text']}")
                                    st.error(f"❌ Format invalide: {detection['parsed_info']['error']}")
                                st.write(f"**Confiance:** {detection['confidence']:.2f}")
                                similar_count = sum(1 for d in all_detections_from_webcam 
                                                  if similarity(d['text'], detection['text']) >= 0.8)
                                st.write(f"**Détections similaires:** {similar_count} fois")
                            with col_crop:
                                st.image(cv2.cvtColor(detection['cropped_image'], cv2.COLOR_BGR2RGB))
                    
                    save_results(unique_detections, "webcam")
                else:
                    st.info("ℹ️ Aucune plaque détectée via la webcam.")

    # ===============================
    # HISTORIQUE DES DÉTECTIONS
    # ===============================
    if 'detection_history' in st.session_state and st.session_state.detection_history:
        st.markdown("---")
        st.header("📋 Historique des détections")
        
        df = pd.DataFrame(st.session_state.detection_history)
        
        # Filtres
        col1, col2, col3 = st.columns(3)
        with col1:
            mode_filter = st.selectbox("Filtrer par mode:", ["Tous"] + df['mode'].unique().tolist())
        with col2:
            min_conf = st.slider("Confiance minimale:", 0.0, 1.0, 0.0, 0.1)
        
        # Appliquer les filtres
        filtered_df = df.copy()
        if mode_filter != "Tous":
            filtered_df = filtered_df[filtered_df['mode'] == mode_filter]
        filtered_df = filtered_df[filtered_df['confidence'] >= min_conf]
        
        # Extraire les informations de la colonne 'parsed_info'
        if not filtered_df.empty:
            parsed_data = filtered_df['parsed_info'].apply(pd.Series)
            display_df = pd.concat([
                filtered_df[['timestamp', 'mode', 'text', 'confidence']],
                parsed_data[['serial_number', 'category', 'year', 'wilaya']]
            ], axis=1)

            # Renommer les colonnes pour un affichage plus clair
            display_df.rename(columns={
                'serial_number': 'Numéro de Série',
                'category': 'Catégorie',
                'year': 'Année',
                'wilaya': 'Wilaya'
            }, inplace=True)

            # Remplacer les codes par les noms correspondants
            display_df['Catégorie'] = display_df['Catégorie'].apply(get_vehicle_category_name)
            display_df['Wilaya'] = display_df['Wilaya'].apply(get_wilaya_name)
            display_df['Année'] = display_df['Année'].apply(lambda y: f"20{y}" if y else None)

            # Afficher le tableau amélioré
            st.dataframe(display_df)
        else:
            st.dataframe(filtered_df)
        
        # Télécharger l'historique
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Télécharger l'historique (CSV)",
            data=csv,
            file_name="detection_history.csv",
            mime="text/csv",
        )

# ===============================
# MÉTHODES ALTERNATIVES POUR LA WEBCAM
# ===============================

def alternative_webcam_method():
    """Méthode alternative utilisant OpenCV pour la webcam (sans streamlit-webrtc)"""
    st.header("📹 Détection en temps réel (méthode alternative)")
    st.warning("⚠️ Cette méthode utilise OpenCV directement. Fermez la fenêtre OpenCV pour revenir à Streamlit.")
    
    if st.button("🚀 Lancer la détection webcam"):
        with st.spinner("🔄 Chargement des modèles..."):
            yolo_model, plate_recognizer = load_models()
        
        if yolo_model is None or plate_recognizer is None:
            return
        
        confidence_threshold = st.sidebar.slider(
            "Seuil de confiance", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.5, 
            step=0.05
        )
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Impossible d'accéder à la webcam")
            return
        
        st.info("📹 Webcam lancée! Appuyez sur 'q' dans la fenêtre OpenCV pour arrêter.")
        all_detections = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame, detections = process_detection(
                frame.copy(), yolo_model, plate_recognizer, confidence_threshold
            )
            
            if detections:
                all_detections.extend(detections)
            
            cv2.imshow('Détection de plaques', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Traiter les résultats
        if all_detections:
            unique_detections = deduplicate_plates(all_detections)
            st.success(f"✅ Session terminée! {len(unique_detections)} plaques uniques détectées")
            
            for i, detection in enumerate(unique_detections, 1):
                with st.expander(f"🚗 Véhicule {i}"):
                    col_info, col_crop = st.columns([2, 1])
                    with col_info:
                        if detection['parsed_info']['is_valid']:
                            formatted_text = f"{detection['parsed_info']['serial_number']} {detection['parsed_info']['category']}{detection['parsed_info']['year']} {detection['parsed_info']['wilaya']}"
                            st.write(f"**Matricule:** {formatted_text}")
                            st.write(f"**Numéro de série:** {detection['parsed_info']['serial_number']}")
                            st.write(f"**Catégorie de véhicule:** {get_vehicle_category_name(detection['parsed_info']['category'])} ({detection['parsed_info']['category']})")
                            st.write(f"**Année:** 20{detection['parsed_info']['year']}")
                            st.write(f"**Wilaya:** {get_wilaya_name(detection['parsed_info']['wilaya'])} ({detection['parsed_info']['wilaya']})")
                        else:
                            st.write(f"**Matricule (brut):** {detection['text']}")
                            st.error(f"❌ Format invalide: {detection['parsed_info']['error']}")
                        st.write(f"**Confiance:** {detection['confidence']:.2f}")
                    with col_crop:
                        st.image(cv2.cvtColor(detection['cropped_image'], cv2.COLOR_BGR2RGB))
            
            save_results(unique_detections, "webcam_alternative")
        else:
            st.info("ℹ️ Aucune plaque détectée.")

# Ajouter la méthode alternative si webrtc n'est pas disponible
if not WEBRTC_AVAILABLE:
    st.markdown("---")
    alternative_webcam_method()

if __name__ == "__main__":
    main()
