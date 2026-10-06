# app.py (Code Complet et Mis à Jour pour Vitesse, Précision et Correction de la Comparaison)

import streamlit as st
import os
import shutil 
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- CONFIGURATION GLOBALE ---
DB_PATH = "faiss_index"
DOCS_PATH = "documents"
# MODÈLE RECOMMANDÉ pour la VITESSE : Assurez-vous d'avoir téléchargé ce modèle via Ollama.
MODEL_NAME = "mistral:7b-instruct-v0.2-q4_0" 
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" 
SUPPORTED_FORMATS = ["pdf", "docx", "txt", "xlsx"]

# --- TEMPLATE DE PROMPT EN FRANÇAIS (Stricte pour la Précision) ---
FRENCH_QA_TEMPLATE = """
Tu es un assistant expert RAG serviable et courtois. Ton rôle est de répondre uniquement en français.

Règles de Conversation:
1. Si la question est une simple salutation (ex: bonjour, merci), réponds de manière amicale.
2. Si la question nécessite une recherche RAG: **Ton unique source d'information est le Contexte fourni**. En te basant strictement sur ces informations, réponds de manière complète et concise à la Question de l'utilisateur.
3. **IMPÉRATIF DE PRÉCISION : TU NE DEVEZ PAS inventer ou halluciner des informations qui ne figurent pas dans le Contexte.**
4. Si les informations fournies dans le Contexte ne contiennent pas la réponse, indique clairement et poliment : "Je suis désolé, les documents disponibles ne contiennent pas l'information demandée."

---
Contexte: {context}
---
Question: {question}
"""

FRENCH_QA_PROMPT = PromptTemplate(
    template=FRENCH_QA_TEMPLATE, input_variables=["context", "question"]
)

# --- FONCTIONS UTILITAIRES ---

@st.cache_resource
def load_faiss_index():
    """Charge l'index FAISS une seule fois, avec le modèle d'embeddings rapide."""
    try:
        embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        # Nécessaire pour charger les indexes créés localement
        return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Erreur de chargement de l'index FAISS. Assurez-vous d'avoir exécuté ingest.py. Détail: {e}")
        return None

def get_loader(file_path, file_name):
    """Détermine le loader approprié pour le type de fichier."""
    if file_name.endswith(".pdf"):
        return PyPDFLoader(file_path)
    elif file_name.endswith((".txt", ".md")):
        return TextLoader(file_path)
    elif file_name.endswith(".docx"):
        return Docx2txtLoader(file_path)
    elif file_name.endswith((".xls", ".xlsx")):
        return UnstructuredExcelLoader(file_path)
    return None

def setup_qa_chain(vector_store):
    """Configure la chaîne de Q&A CLASSIQUE avec le prompt en français."""
    llm = Ollama(model=MODEL_NAME) 
    
    # k=3 pour une meilleure vitesse
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": FRENCH_QA_PROMPT} 
    )
    return qa_chain

def upload_and_reindex_documents(vector_store):
    """Permet l'upload et l'indexation dynamique de documents."""
    st.sidebar.subheader("📥 Ajouter des Documents à la Base")
    uploaded_file = st.sidebar.file_uploader(
        "Sélectionnez un document pour indexer (PDF, DOCX, TXT, XLSX).",
        type=SUPPORTED_FORMATS
    )
    
    if uploaded_file is not None and st.sidebar.button("Indexer et Mettre à Jour la Base"):
        
        if uploaded_file.type.split('/')[-1] not in SUPPORTED_FORMATS and uploaded_file.name.split('.')[-1] not in SUPPORTED_FORMATS:
            st.sidebar.error("Format de fichier non supporté.")
            return

        file_path = os.path.join(DOCS_PATH, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner(f"Indexation de {uploaded_file.name}..."):
            
            loader = get_loader(file_path, uploaded_file.name)
            if not loader:
                st.sidebar.error(f"Erreur: Le loader pour {uploaded_file.name} n'a pas pu être initialisé.")
                return

            new_docs = loader.load()
            # Paramètres de découpage optimisés
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=250)
            new_chunks = text_splitter.split_documents(new_docs)

            embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME) 
            vector_store.add_documents(new_chunks)
            vector_store.save_local(DB_PATH) 
            
            st.sidebar.success(f"Document '{uploaded_file.name}' indexé et base mise à jour ! ({len(new_chunks)} chunks ajoutés)")
            st.rerun() 
            
# --- PAGES DE L'APPLICATION ---

def qa_classique_page(vector_store):
    """Implémentation de la Fonctionnalité A: Q&A Classique."""
    st.title("🤖 1. Q&A Classique (RAG)")
    
    qa_chain = setup_qa_chain(vector_store)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Votre question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Recherche et génération de la réponse..."):
                
                result = qa_chain.invoke({"query": prompt})
                response = result['result']
                
                st.markdown(response)
                
                sources = set(doc.metadata.get("source", "Inconnu") for doc in result.get("source_documents", []))
                is_rag_answer = (
                    "ne contient pas" not in response.lower() and 
                    "bonjour" not in prompt.lower() and 
                    "salut" not in prompt.lower()
                )
                if sources and is_rag_answer:
                    st.info(f"Sources utilisées: {', '.join(sources)}")
        
        st.session_state.messages.append({"role": "assistant", "content": response})


def document_comparison_page(vector_store):
    """Implémentation de la Fonctionnalité B: Comparaison de Documents (CORRIGÉE)."""
    st.title("📑 2. Comparaison de Documents")
    st.markdown("Sélectionnez ou **uploadez** un document de référence pour le comparer avec les extraits les plus similaires dans la base.")

    doc_files = [f for f in os.listdir(DOCS_PATH) if f.endswith(tuple(f".{fmt}" for fmt in SUPPORTED_FORMATS))]
    
    st.subheader("A. Choisir un document de la base")
    selected_doc = st.selectbox("Document de référence :", ["-- Sélectionner --"] + doc_files)
    
    st.subheader("B. Uploader un document temporaire")
    temp_uploaded_file = st.file_uploader(
        "Uploader un document (non indexé, pour comparaison seulement)",
        type=SUPPORTED_FORMATS,
        key="temp_upload_comparison"
    )
    
    file_to_compare = None
    if temp_uploaded_file:
        # Sauvegarder temporairement l'upload
        temp_dir = os.path.join(DOCS_PATH, "temp_upload")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, temp_uploaded_file.name)
        
        with open(temp_path, "wb") as f:
            f.write(temp_uploaded_file.getbuffer())
        file_to_compare = temp_path
        st.info(f"Document temporaire prêt pour la comparaison : {temp_uploaded_file.name}")
        
    elif selected_doc != "-- Sélectionner --":
        file_to_compare = os.path.join(DOCS_PATH, selected_doc)

    if st.button("Lancer la comparaison") and file_to_compare:
        
        with st.spinner("Recherche des extraits similaires et génération du rapport..."):
            
            file_name = os.path.basename(file_to_compare)
            
            # --- ÉTAPE CLÉ DE CORRECTION ---
            # Clé d'exclusion : le nom de fichier du document de référence
            exclusion_key = file_name
            if selected_doc != "-- Sélectionner --":
                # Si c'est un doc de la base, utilisons son nom exact
                exclusion_key = selected_doc 

            # Charger le document de référence
            loader = get_loader(file_to_compare, file_name)
            if not loader:
                 st.error(f"Impossible de charger le document {file_name}.")
                 return
                 
            ref_docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=250)
            ref_chunks = text_splitter.split_documents(ref_docs)
            
            unique_comparable_chunks = []
            embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
            
            # Chercher les chunks les plus proches
            for chunk in ref_chunks:
                # k=5 pour une comparaison riche
                similar_chunks = vector_store.similarity_search(chunk.page_content, k=5) 
                
                for doc in similar_chunks:
                    
                    # FILTRAGE AMÉLIORÉ : Exclure le document de référence
                    # Vérifier si la clé d'exclusion (nom de fichier) est contenue dans le chemin source du chunk.
                    chunk_source = os.path.basename(doc.metadata.get("source", ""))
                    if exclusion_key not in chunk_source:
                         if doc not in unique_comparable_chunks:
                            unique_comparable_chunks.append(doc)
            
            # ---------------------------------
            
            if not unique_comparable_chunks:
                st.info(f"Aucun extrait similaire trouvé dans d'autres documents de la base que '{exclusion_key}'.")
                return
            
            # Limiter à 4 chunks pour la comparaison (Vitesse)
            chunks_to_compare = unique_comparable_chunks[:4] 
            
            comparison_context = f"Informations et extraits de documents pour la comparaison :\n"
            for i, doc in enumerate(chunks_to_compare):
                # Utiliser os.path.basename pour n'afficher que le nom du fichier (plus propre)
                comparison_context += f"\n\n--- Doc Similaire {i+1} (Source: {os.path.basename(doc.metadata.get('source'))}): {doc.page_content[:500]}..."

            
            # Préparation du Prompt de Comparaison
            llm = Ollama(model=MODEL_NAME)
            comparison_prompt = f"""
            En te basant **STRICTEMENT** sur le contexte fourni, génère un rapport de comparaison structuré entre le document de référence ('{file_name}') et les autres extraits de la base de données.
            
            Le rapport doit impérativement contenir deux sections :
            1. **Points de Similarité :** Utilise des listes à puces pour lister les points communs.
            2. **Divergences/Différences :** Utilise une liste à puces ou un tableau pour lister les différences claires.
            Ta réponse doit être uniquement en français.

            ---
            Contexte : {comparison_context}
            ---
            Rapport de comparaison :
            """
            
            # Génération du Rapport
            comparison_report = llm.invoke(comparison_prompt)
            st.subheader("Rapport de Comparaison Généré")
            st.markdown(comparison_report)

            # Nettoyage du fichier temporaire si c'était un upload
            if temp_uploaded_file:
                if os.path.exists(os.path.dirname(file_to_compare)):
                    shutil.rmtree(os.path.dirname(file_to_compare))
                st.session_state.pop("temp_upload_comparison", None)


def intelligent_qa_page(vector_store):
    """Implémentation de la Fonctionnalité C: Q&A Intelligent (Recherche Hybride)."""
    st.title("🧠 3. Q&A Intelligent par Information Clé")
    st.markdown("Recherche hybride (texte exact + contexte sémantique) pour trouver les documents liés à une donnée sensible. La synthèse sera en français.")

    search_query = st.text_input("Information clé (ex: référence, numéro, nom) :", "")

    if st.button("Rechercher la donnée"):
        if search_query:
            with st.spinner("Recherche hybride et synthèse..."):
                
                # 1. Recherche Vectorielle (Contexte Sémantique) - k=10 pour ne rien manquer
                retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})
                vector_results = retriever.get_relevant_documents(search_query)

                # 2. Recherche Exacte (Filtrage simple sur le contenu)
                exact_results = [doc for doc in vector_results if search_query.lower() in doc.page_content.lower()]

                all_results = exact_results + [doc for doc in vector_results if doc not in exact_results]
                
                if not all_results:
                    st.info("Aucun document lié trouvé.")
                    return
                
                # Limiter à 4 chunks pour la synthèse (Vitesse)
                llm = Ollama(model=MODEL_NAME)
                chunks_for_synthesis = all_results[:4] 
                
                synthesis_context = f"Informations et extraits de documents liés à l'information clé '{search_query}' :"
                for i, doc in enumerate(chunks_for_synthesis): 
                    synthesis_context += f"\n\n--- Extrait {i+1} (Source: {doc.metadata.get('source', 'Inconnu')}): {doc.page_content}..."

                # Prompt forcé pour un format synthétique.
                synthesis_prompt = f"""
                En te basant **UNIQUEMENT** sur le contexte suivant, crée une synthèse. 
                
                La synthèse doit impérativement contenir deux parties :
                1. **Documents Pertinents :** Une liste des noms de documents (Source) où l'information clé a été trouvée.
                2. **Résumé des Informations :** Un résumé des points importants et des détails qui entourent l'information clé ('{search_query}') dans ces documents.
                Ta réponse doit être en français et ne doit rien inventer.

                ---
                Contexte: {synthesis_context}
                ---
                Synthèse :
                """

                synthesis_report = llm.invoke(synthesis_prompt)
                st.subheader("Rapport de Synthèse pour l'Information Clé")
                st.markdown(synthesis_report)

# --- APPLICATION PRINCIPALE (NAVIGATION) ---
def main_app():
    st.set_page_config(page_title="Chatbot RAG Expert", layout="wide")
    st.sidebar.title("Navigation")
    
    vector_store = load_faiss_index()

    if vector_store:
        upload_and_reindex_documents(vector_store)
        
        st.sidebar.markdown("---")
        page = st.sidebar.radio("Sélectionnez une fonctionnalité :", ["Q&A Classique", "Comparaison de Documents", "Q&A Intelligent"])

        if page == "Q&A Classique":
            qa_classique_page(vector_store)
        elif page == "Comparaison de Documents":
            document_comparison_page(vector_store)
        elif page == "Q&A Intelligent":
            intelligent_qa_page(vector_store)

if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
         st.error(f"Le dossier d'index FAISS '{DB_PATH}' n'existe pas. Veuillez lancer `python ingest.py` d'abord pour créer la base de données!")
    else:
        main_app()