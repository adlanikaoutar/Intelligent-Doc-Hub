# ingest.py (Code Complet et Mis à Jour pour Multi-format et Vitesse/Précision)
#il prend les documents sources (PDF, Word, TXT, Excel…),
#les analyse, les découpe, puis les convertit en représentations numériques (embeddings) afin de les stocker dans une base vectorielle FAISS.

import os
#pour charger différents formats de documents.
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredExcelLoader
#pour convertir les textes en vecteurs numériques (embeddings).
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


# --- CONFIGURATION ---
DOCS_PATH = "documents"
DB_PATH = "faiss_index"

# Utilisation d'un modèle d'embeddings optimisé pour la vitesse sur CPU
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" 

def get_loader(file_path, file_name):
    """Détermine le loader approprié pour le type de fichier."""
    if file_name.endswith(".pdf"):
        print(f"Chargement de {file_name} (PDF)...")
        return PyPDFLoader(file_path)
    elif file_name.endswith((".txt", ".md")):
        print(f"Chargement de {file_name} (TXT/MD)...")
        return TextLoader(file_path)
    elif file_name.endswith(".docx"):
        print(f"Chargement de {file_name} (DOCX)...")
        return Docx2txtLoader(file_path)
    elif file_name.endswith((".xls", ".xlsx")):
        print(f"Chargement de {file_name} (Excel)...")
        # NOTE : UnstructuredExcelLoader nécessite l'installation préalable de 'openpyxl'
        return UnstructuredExcelLoader(file_path)
    return None

def ingest_documents():
    print("--- Démarrage de l'indexation des documents ---")
    documents = []
    #Étape 1 : Chargement des documents
    # 1. Chargement des documents depuis le dossier
    for file_name in os.listdir(DOCS_PATH):
        file_path = os.path.join(DOCS_PATH, file_name)
        
        if file_name.startswith('.'): # Ignorer les fichiers cachés
            continue

        loader = get_loader(file_path, file_name)
        
        if loader:
            try:
                documents.extend(loader.load())
            except Exception as e:
                print(f"Erreur lors du chargement de {file_name}: {e}")
        else:
            print(f"ATTENTION : Format non pris en charge ({file_name}). Ignoré.")
            
    if not documents:
        print(f"Aucun document supporté trouvé dans le dossier '{DOCS_PATH}'. Veuillez y placer des PDF, DOCX, TXT ou XLSX.")
        return
        
    # 2. Découpage en chunks (Optimisé pour la précision)
    print(f"Découpage de {len(documents)} pages/fichiers en chunks...")
    # MODIFICATION : Taille des chunks réduite et Overlap augmenté pour plus de contexte local
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,  # Réduit de 1000 à 750
        chunk_overlap=250 # Augmenté de 200 à 250
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Nombre total de chunks créés : {len(chunks)}")
    
    # 3. Génération des embeddings et stockage (RAPIDE)
    print("Génération des embeddings et création de la base de données vectorielle (utilisant SentenceTransformer)...")
    # L'installation de 'sentence-transformers' est nécessaire ici
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # 4. Sauvegarde
    if not os.path.exists(DB_PATH):
        os.makedirs(DB_PATH)
        
    vector_store.save_local(DB_PATH)
    print(f"--- Indexation terminée avec succès ! Index sauvegardé dans {DB_PATH} ---")

if __name__ == "__main__":
    if not os.path.exists(DOCS_PATH):
        os.makedirs(DOCS_PATH)
    ingest_documents()