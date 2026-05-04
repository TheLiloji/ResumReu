"""Minimal Streamlit UI — upload audio, monitor processing, run Q&A.

The UI talks to the FastAPI backend via HTTP — it does NOT import
infrastructure layers directly. This keeps a clean delivery boundary.
"""

from __future__ import annotations

import os
import time

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="ResumReu", page_icon="🎙️", layout="wide")
st.title("ResumReu — Comptes rendus de réunion")

tab_upload, tab_meetings, tab_qa = st.tabs(["📤 Upload", "📋 Réunions", "💬 Q&A"])


with tab_upload:
    st.subheader("Charger un nouvel enregistrement")
    title = st.text_input("Titre de la réunion")
    audio = st.file_uploader(
        "Fichier audio", type=["mp3", "wav", "m4a", "flac", "ogg", "opus"]
    )
    if st.button("Lancer le traitement", disabled=not (title and audio)):
        files = {"audio": (audio.name, audio.getvalue(), audio.type)}
        resp = requests.post(
            f"{API_URL}/meetings", data={"title": title}, files=files, timeout=120
        )
        if resp.ok:
            data = resp.json()
            st.success(f"Réunion créée : {data['meeting_id']}")
            st.json(data)
        else:
            st.error(f"Erreur {resp.status_code}: {resp.text}")


with tab_meetings:
    st.subheader("Réunions récentes")
    if st.button("Rafraîchir"):
        st.rerun()
    resp = requests.get(f"{API_URL}/meetings", timeout=10)
    if resp.ok:
        meetings = resp.json()
        if not meetings:
            st.info("Aucune réunion pour le moment.")
        for m in meetings:
            with st.expander(f"{m['title']} — {m['status']} ({m['created_at']})"):
                detail = requests.get(
                    f"{API_URL}/meetings/{m['id']}", timeout=10
                ).json()
                st.write(f"**Statut :** {detail['status']}")
                if detail.get("summary"):
                    st.markdown("**Résumé :**")
                    st.write(detail["summary"])
                if detail.get("error_message"):
                    st.error(detail["error_message"])
                if detail["status"] == "completed":
                    st.markdown(
                        f"[📄 Télécharger le compte rendu]"
                        f"({API_URL}/meetings/{m['id']}/document)"
                    )
    else:
        st.error("API indisponible.")


with tab_qa:
    st.subheader("Poser une question")
    question = st.text_area(
        "Question", placeholder="Ex. Quelles décisions ont été prises sur le projet Y2 ?"
    )
    top_k = st.slider("Nombre de passages", 3, 12, 6)
    if st.button("Envoyer", disabled=not question):
        with st.spinner("Recherche en cours..."):
            resp = requests.post(
                f"{API_URL}/query",
                json={"question": question, "top_k": top_k},
                timeout=180,
            )
        if resp.ok:
            data = resp.json()
            st.markdown("### Réponse")
            st.write(data["answer"])
            with st.expander(f"Sources ({len(data['sources'])})"):
                for src in data["sources"]:
                    st.markdown(f"**[#{src['rank']}]** score={src['score']:.3f}")
                    st.caption(str(src["metadata"]))
                    st.write(src["text"])
                    st.divider()
        else:
            st.error(f"Erreur {resp.status_code}: {resp.text}")
