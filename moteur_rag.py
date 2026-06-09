import os
from dotenv import load_dotenv
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore

load_dotenv()

PROMPT_SYSTEME = (
    "Tu es un expert fiscaliste agréé au Cameroun (ConsulFiscal Pro). Tu rédiges des réponses claires, "
    "structurées et purement basées sur les documents fournis.\n\n"
    "RÈGLE DE PRIORITÉ JURIDIQUE : L'ordre de préséance est : Circulaire LFI 2026 > LFI 2026 > LFI 2025 > CGI 2024.\n"
    "Si le CGI 2024 stipule une règle qui a été réformée par une loi de finances ultérieure, tu dois appliquer "
    "uniquement la règle la plus récente (2026) et mettre poliment en garde le comptable contre l'ancienne règle.\n\n"
    "Reste professionnel, concis et cite systématiquement les numéros d'articles si présents dans le texte."
)

def interroger_le_moteur(question: str) -> dict:
    """Interroge la base Pinecone et retourne la réponse IA ainsi que le dictionnaire des sources uniques."""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    pinecone_index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
    
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    
    # On récupère les 3 meilleurs morceaux de texte pour croiser les lois
    query_engine = index.as_query_engine(similarity_top_k=3, system_prompt=PROMPT_SYSTEME)
    response_object = query_engine.query(question)
    
    # Collecte des métadonnées injectées lors de l'indexation
    liens_sources = {}
    for node in response_object.source_nodes:
        meta = node.node.metadata
        url = meta.get("source_url")
        nom = meta.get("source_nom")
        if url and nom:
            liens_sources[nom] = url  # Élimine automatiquement les doublons d'URL
            
    return {
        "reponse": str(response_object),
        "sources": liens_sources
    }