import streamlit as st
import json
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("💬 Conversational Learning Agent")
st.caption("Basé sur le dataset QReCC — Projet Deep Learning")

@st.cache_resource
def charger_donnees():
    with open("qrecc_train.json", "r", encoding="utf-8", errors="ignore") as f:
        train_data = json.load(f)
    return train_data

@st.cache_resource
def construire_index(train_data):
    documents = []
    for ex in train_data:
        if ex.get("Answer") and len(ex["Answer"].strip()) > 10:
            documents.append({
                "question": ex.get("Rewrite", ex["Question"]),
                "reponse": ex["Answer"]
            })
    corpus = [f"{d['question']} {d['reponse']}" for d in documents]
    vectorizer = TfidfVectorizer(max_features=15000, stop_words="english")
    index = vectorizer.fit_transform(corpus)
    return documents, vectorizer, index

train_data = charger_donnees()
documents, vectorizer, index = construire_index(train_data)

def trouver_entites(texte):
    mots_courants = {"The","A","An","In","On","At","Is","Was","Are","Were","He","She","They","It"}
    pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b"
    entites = re.findall(pattern, texte)
    return [e for e in entites if e not in mots_courants]

def trouver_sujet(contexte):
    if not contexte:
        return ""
    tous = []
    for i, texte in enumerate(contexte):
        tous.extend(trouver_entites(texte) * (i + 1))
    if not tous:
        return ""
    return Counter(tous).most_common(1)[0][0]

def réécrire(question, contexte):
    pronoms = [r"\bhe\b", r"\bshe\b", r"\bthey\b", r"\bhim\b",
               r"\bher\b", r"\bhis\b", r"\btheir\b", r"\bit\b"]
    est_ambigue = any(re.search(p, question.lower()) for p in pronoms)
    if not est_ambigue or not contexte:
        return question, False
    sujet = trouver_sujet(contexte)
    if not sujet:
        return question, False
    remplacements = {
        r"\bhe\b": sujet, r"\bshe\b": sujet, r"\bthey\b": sujet,
        r"\bhim\b": sujet, r"\bher\b": sujet, r"\bthem\b": sujet,
        r"\bhis\b": sujet + "'s", r"\btheir\b": sujet + "'s",
    }
    for pattern, remplacement in remplacements.items():
        nouvelle = re.sub(pattern, remplacement, question, flags=re.IGNORECASE)
        if nouvelle != question:
            return nouvelle, True
    return question, False

def chercher(question):
    vecteur = vectorizer.transform([question])
    scores = cosine_similarity(vecteur, index)[0]
    meilleurs = scores.argsort()[::-1][:5]
    return [{"reponse": documents[i]["reponse"], "score": float(scores[i])} for i in meilleurs]

def confiance(resultats):
    score = resultats[0]["score"]
    if score > 0.15:
        return "élevée", "🟢"
    elif score > 0.05:
        return "moyenne", "🟡"
    else:
        return "faible", "🔴"

if "messages" not in st.session_state:
    st.session_state.messages = []
if "contexte" not in st.session_state:
    st.session_state.contexte = []

if st.button("🔄 Nouvelle conversation"):
    st.session_state.messages = []
    st.session_state.contexte = []
    st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "info" in msg:
            st.caption(msg["info"])

question = st.chat_input("Pose ta question ici...")

if question:
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    reecrite, modifiee = réécrire(question, st.session_state.contexte)
    resultats = chercher(reecrite)
    niveau, emoji = confiance(resultats)

    if niveau == "faible":
        reponse = "Je ne suis pas sûr de la réponse. Peux-tu reformuler ta question ?"
    else:
        reponse = resultats[0]["reponse"]

    info = f"{emoji} Confiance : {niveau} (score={resultats[0]['score']:.3f})"
    if modifiee:
        info = f"✏️ Réécrite : *{reecrite}* | " + info

    with st.chat_message("assistant"):
        st.write(reponse)
        st.caption(info)

    st.session_state.messages.append({
        "role": "assistant",
        "content": reponse,
        "info": info
    })

    st.session_state.contexte.append(question)
    st.session_state.contexte.append(reponse)
