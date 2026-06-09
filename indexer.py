import os
import requests
from io import BytesIO
from dotenv import load_dotenv
from pinecone import Pinecone
from llama_index.readers.llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore

load_dotenv()

# Tes 4 sources officielles bien ciblées
SOURCES_FISCALES = {
    "CGI_2024": "https://www.impots.cm/sites/default/files/documents/CGI%202024%20version%20francaise.pdf",
    "LFI_2025": "https://impots.cm/sites/default/files/documents/loi_n_2024_013_du_23_12_2024-web.pdf",
    "LFI_2026": "https://rag-fisc-lfi2026-169136975521-eu-west-3-an.s3.eu-west-3.amazonaws.com/lfi-2026.pdf",
    "Circulaire_LFI_2026": "https://impots.cm/sites/default/files/publications/circulaire%20lf%202026%20VF%20%281%29-compress%C3%A9.pdf"
}

def executer_indexation_globale():
    # Initialisation de Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    pinecone_index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Configuration de LlamaParse pour la conversion intelligente en Markdown
    parser = LlamaParse(api_key=os.getenv("LLAMA_CLOUD_API_KEY"), result_type="markdown", language="fr")
    tous_les_documents = []
    
    for nom, url in SOURCES_FISCALES.items():
        print(f"🌐 Ingestion de : {nom}...")
        try:
            reponse = requests.get(url, timeout=45)
            if reponse.status_code == 200:
                fichier_en_memoire = BytesIO(reponse.content)
                
                # Extraction textuelle sémantique
                extraits = parser.load_data(file_data=fichier_en_memoire, extra_info={"file_name": f"{nom}.pdf"})
                
                # Injection des métadonnées pour la traçabilité des sources
                for doc in extraits:
                    doc.metadata["source_url"] = url
                    doc.metadata["source_nom"] = nom.replace("_", " ")
                    
                tous_les_documents.extend(extraits)
                print(f"✅ {nom} converti et préparé.")
            else:
                print(f"❌ Impossible de télécharger {nom} (Code {reponse.status_code})")
        except Exception as e:
            print(f"⚠️ Erreur sur {nom} : {str(e)}")
            
    if tous_les_documents:
        print(f"🚀 Envoi de {len(tous_les_documents)} blocs de texte vers le cloud Pinecone...")
        VectorStoreIndex.from_documents(tous_les_documents, storage_context=storage_context)
        print("🎯 Ingestion et vectorisation terminées avec succès !")

if __name__ == "__main__":
    executer_indexation_globale()