# import os
# from dotenv import load_dotenv
# from pinecone import Pinecone
# from pdf2image import convert_from_path
# import pytesseract

# from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
# from llama_index.vector_stores.pinecone import PineconeVectorStore
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# load_dotenv()

# # =====================================================================
# # CONFIGURATION TESSERACT WINDOWS
# # Indique ici le chemin exact où Tesseract a été installé sur ton PC :
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# # =====================================================================

# def executer_ocr_local_lfi2026():
#     # 1. Configuration de l'embedding gratuit (BGE-Small)
#     print("⏳ Initialisation du modèle d'embedding gratuit (BGE-Small)...")
#     Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

#     # 2. Initialisation de Pinecone
#     pinecone_api_key = os.getenv("PINECONE_API_KEY")
#     pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
    
#     pc = Pinecone(api_key=pinecone_api_key)
#     pinecone_index = pc.Index(pinecone_index_name)
#     vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
#     storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
#     # 3. Vérification du fichier
#     chemin_pdf = "lfi-2026.pdf"
#     if not os.path.exists(chemin_pdf):
#         print(f"❌ Erreur : Le fichier {chemin_pdf} est introuvable.")
#         return

#     print(f"📸 [MODE OCR LOCAL] Conversion du PDF scanné {chemin_pdf} en images...")
#     tous_les_documents = []
    
#     try:
#         # Convertit le PDF en liste d'images (une par page)
#         # Note : Si le PDF est très long, tu peux tester sur les 5 premières pages en ajoutant : last_page=5
#         images = convert_from_path(chemin_pdf, dpi=200)
#         print(f"✅ {len(images)} pages converties en images. Début de l'OCR en français...")
        
#         for i, img in enumerate(images):
#             # Extraction du texte avec le dictionnaire français ('fra')
#             texte_page = pytesseract.image_to_string(img, lang="fra")
#             texte_propre = texte_page.strip()
            
#             if texte_propre:
#                 doc = Document(
#                     text=texte_propre,
#                     metadata={
#                         "source_url": "Fichier Local LFI 2026 (Scanné)",
#                         "source_nom": "LFI 2026",
#                         "page_numero": i + 1
#                     }
#                 )
#                 tous_les_documents.append(doc)
#                 print(f"📝 Page {i + 1}/{len(images)} extraite par OCR.")
#             else:
#                 print(f"⚠️ Page {i + 1}/{len(images)} : aucun texte détecté.")
                
#     except Exception as e:
#         print(f"❌ Échec de la phase d'OCR locale : {str(e)}")
#         print("💡 Astuce : Vérifie que le chemin vers tesseract.exe (ligne 14) est correct pour ta machine.")
#         return

#     # 4. Envoi final vers Pinecone
#     if tous_les_documents:
#         print(f"\n🚀 Envoi de {len(tous_les_documents)} pages vectorisées vers Pinecone...")
#         VectorStoreIndex.from_documents(
#             tous_les_documents, 
#             storage_context=storage_context,
#             show_progress=True
#         )
#         print("🎯 Succès total ! La LFI 2026 scannée a été lue, vectorisée et ajoutée à ton Pinecone !")
#     else:
#         print("❌ Aucun texte n'a pu être lu par l'OCR.")

# if __name__ == "__main__":
#     executer_ocr_local_lfi2026()






import os
import requests
from dotenv import load_dotenv
from pinecone import Pinecone
from llama_index.readers.llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

# Liste des sources officielles
SOURCES_FISCALES = {
    "CGI_2024": "https://www.impots.cm/sites/default/files/documents/CGI%202024%20version%20francaise.pdf",
    "LFI_2025": "https://impots.cm/sites/default/files/documents/loi_n_2024_013_du_23_12_2024-web.pdf",
    "LFI_2026": "https://rag-fisc-lfi2026-169136975521-eu-west-3-an.s3.eu-west-3.amazonaws.com/lfi-2026.pdf",
    "Circulaire_LFI_2026": "https://impots.cm/sites/default/files/publications/circulaire%20lf%202026%20VF%20%281%29-compress%C3%A9.pdf"
}

def executer_indexation_globale():
    # 1. Configuration de l'embedding gratuit
    print("⏳ Initialisation du modèle d'embedding gratuit (BGE-Small)...")
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5") 

    # 2. Initialisation de Pinecone
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
    
    if not pinecone_api_key or not pinecone_index_name:
        print("❌ Erreur  : PINECONE_API_KEY ou PINECONE_INDEX_NAME manquant dans le .env")
        return

    pc = Pinecone(api_key=pinecone_api_key)
    pinecone_index = pc.Index(pinecone_index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # 3. Configuration de LlamaParse
    llama_cloud_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not llama_cloud_key:
        print("❌ Erreur : LLAMA_CLOUD_API_KEY manquante.")
        return

    parser = LlamaParse(api_key=llama_cloud_key, result_type="markdown", language="fr")
    tous_les_documents = []
    
    for nom, url in SOURCES_FISCALES.items():
        print(f"\n🌐 Ingestion sémantique de : {nom}...")
        
        # STRATÉGIE : Si c'est la LFI_2026 et que le fichier est déjà là, on ne télécharge pas
        if nom == "LFI_2026" and os.path.exists("lfi-2026.pdf"):
            print("📦 [MODE LOCAL] Fichier lfi-2026.pdf détecté sur le disque. Passage direct à l'analyse sémantique.")
            nom_fichier_cible = "lfi-2026.pdf"
            est_fichier_temporaire = False
        else:
            nom_fichier_cible = f"temp_{nom}.pdf"
            est_fichier_temporaire = True
            try:
                # Augmentation du timeout à 120 secondes pour plus de sécurité
                reponse = requests.get(url, timeout=120)
                if reponse.status_code == 200:
                    with open(nom_fichier_cible, "wb") as f:
                        f.write(reponse.content)
                    print(f"📥 Téléchargement réussi pour {nom}.")
                else:
                    print(f"❌ Impossible de télécharger {nom} (Code HTTP {reponse.status_code})")
                    continue
            except Exception as e:
                print(f"⚠️ Erreur réseau lors du téléchargement de {nom} : {str(e)}")
                continue

        # 4. Envoi du fichier à LlamaParse pour l'analyse
        try:
            print(f"⏳ Extraction sémantique par LlamaParse Cloud pour {nom} (cela peut prendre du temps)...")
            extraits = parser.load_data(file_path=nom_fichier_cible)
            
            # Injection des métadonnées pour la traçabilité
            for doc in extraits:
                doc.metadata["source_url"] = url
                doc.metadata["source_nom"] = nom.replace("_", " ")
                
            tous_les_documents.extend(extraits)
            print(f"✅ {nom} converti et préparé avec succès.")
            
        except Exception as e:
            print(f"❌ Échec de l'analyse sémantique pour {nom} : {str(e)}")
            
        finally:
            # On ne supprime le fichier que si c'était un fichier temporaire
            if est_fichier_temporaire and os.path.exists(nom_fichier_cible):
                os.remove(nom_fichier_cible)
            
    # 5. Injection finale dans Pinecone
    if tous_les_documents:
        print(f"\n🚀 Envoi de {len(tous_les_documents)} blocs de texte vers Pinecone...")
        VectorStoreIndex.from_documents(
            tous_les_documents, 
            storage_context=storage_context,
            show_progress=True
        )
        print("🎯 L'indexation globale et gratuite est terminée avec succès !")
    else:
        print("❌ Aucun document n'a pu être préparé pour l'envoi vers Pinecone.")

if __name__ == "__main__":
    executer_indexation_globale()