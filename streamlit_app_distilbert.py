# streamlit_app.py

import streamlit as st
import torch
import re
from bs4 import BeautifulSoup
import spacy
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import pickle


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


# Charger le modèle spacy
try:
    nlp = spacy.load("en_core_web_sm")
    st.write("Modèle Spacy chargé avec succès.")
except OSError:
    st.write("Le modèle Spacy 'en_core_web_sm' n'est pas installé. Installation en cours...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    st.write("Modèle Spacy chargé avec succès après installation.")

# Définir les chemins pour charger le modèle, le tokenizer et le MultiLabelBinarizer
model_save_path = "distilbert_model.pth"  # Chemin du modèle DistilBERT sauvegardé
tokenizer_save_path = "distilbert_tokenizer"  # Chemin du tokenizer DistilBERT sauvegardé
mlb_load_path_drive = "mlb.pkl"  # Chemin du MultiLabelBinarizer sauvegardé dans Google Drive (à ajuster si nécessaire)

# Charger le modèle DistilBERT
num_labels = 201  #  nombre de labels correspondant a celui utilisé lors de l'entrainement du modele

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)  # Remplacez `num_labels=50` par le nombre réel de labels
model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
model.eval()  # Passer le modèle en mode évaluation

# Charger le tokenizer DistilBERT
tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_save_path)

# Charger le MultiLabelBinarizer depuis le fichier .pkl
with open(mlb_load_path_drive, 'rb') as f:
    mlb = pickle.load(f)
st.write("MultiLabelBinarizer chargé avec succès.")

# Définir les stopwords et autres éléments de prétraitement
nltk_stopwords = set(stopwords.words('english'))
custom_stopwords = set([
    'like', 'question', 'use', 'want', 'one', 'know', 'work', 'example', 'code', 'seem', 
    'using', 'instead', 'way', 'get', 'would', 'need', 'following', '1', '2', 'run', 
    'something', 'trying', 'tried', 'also', 'new', 'could', 'see', 'line', 'however', 
    'solution', '3', '4', '5', 'without', 'still', 'answer', 'say', 'another', 'help', 
    'anyone', 'best', 'looking', 'show', 'give', 'better', 'many', 'good', 'even', 
    'think', 'thing', 'look', 'problem', 'try', 'possible'
])
all_stopwords = nltk_stopwords.union(custom_stopwords)

# Fonction de nettoyage du texte HTML et code
def clean_html_code(text):
    # Supprimer les balises <code> et leur contenu
    text = re.sub(r'<code>.*?</code>', '', text, flags=re.DOTALL)
    # Supprimer les balises <p> en conservant le contenu
    text = re.sub(r'</?p>', '', text)
    text = re.sub(r'\n', ' ', text)
    # Utiliser BeautifulSoup pour nettoyer les balises HTML
    soup = BeautifulSoup(text, "html.parser")
    cleaned_text = soup.get_text()
    return cleaned_text

# Fonction de normalisation du texte
def normalize_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)  # Supprimer la ponctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Supprimer les espaces multiples
    return text

# Prétraitement complet du texte
def preprocess_text(text):
    # Nettoyer le texte
    text = clean_html_code(text)
    # Normaliser le texte
    text = normalize_text(text)
    # Lemmatisation et suppression des stopwords
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.pos_ == 'NOUN' and token.text not in all_stopwords]
    return ' '.join(tokens)

# Définir l'interface de l'application Streamlit
st.title("Classification multi-labels des questions StackOverflow avec DistilBERT")

# Champ de texte pour saisir la question
question_input = st.text_area("Entrez une question StackOverflow au format brut (comme le champ 'Body') :")

# Bouton pour lancer la prédiction
if st.button("Prédire les tags"):
    if question_input.strip() != "":
        # Prétraiter la question avec le même prétraitement que celui utilisé pour ProcessedBody_assembled
        processed_question = preprocess_text(question_input)
        st.write(f"Question prétraitée : {processed_question}")

        # Tokenisation de la question prétraitée
        inputs = tokenizer(processed_question, return_tensors="pt", truncation=True, padding=True, max_length=128)

        # Prédire les tags avec DistilBERT
        with torch.no_grad():
            logits = model(**inputs).logits
            predicted_labels = torch.sigmoid(logits).numpy() > 0.5  # Seuil à 0.5

        # Convertir les prédictions en tags réels avec le MultiLabelBinarizer chargé
        predicted_tags = mlb.inverse_transform(np.array([predicted_labels[0]]))

        # Afficher les tags prédits
        st.write(f"Tags prédits pour la question : {predicted_tags}")
    else:
        st.warning("Veuillez entrer une question valide.")
