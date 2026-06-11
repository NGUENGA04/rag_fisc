import os
from dotenv import load_dotenv
from pinecone import Pinecone
from llama_index.core import Settings, VectorStoreIndex, PromptTemplate
from llama_index.llms.groq import Groq
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1. Chargement des variables d'environnement
load_dotenv()

# Définition du template de prompt officiel pour LlamaIndex
SISTEM_PROMPT_TEMPLATE = (
    "Tu es ConsuFiscal, un assistant virtuel expert fiscaliste spécialisé dans le Code Général des Impôts (CGI) du Cameroun.\n"
    "Voici les extraits de lois officiels fournis pour t'aider :\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Instructions impératives :\n"
    "1. Réponds de manière professionnelle, claire, polie et bien structurée (utilise le format Markdown WhatsApp : *gras*, _italique_).\n"
    "2. Base-toi EXCLUSIVEMENT sur les extraits de lois fournis dans le contexte ci-dessus.\n"
    "3. Si le contexte ne contient pas la réponse ou si tu as un doute, dis poliment : "
    "'Désolé, je ne trouve pas de disposition précise dans les textes fiscaux actuels pour répondre à votre demande.'\n"
    "N'invente JAMAIS d'articles ou de taux.\n"
    "4. Réponds toujours en français.\n\n"
    "Question de l'utilisateur : {query_str}\n"
    "Réponse de l'expert ConsuFiscal :"
)

class MoteurRAG:
    def __init__(self):
        """Initialise l'architecture RAG."""
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
        
        if not all([self.groq_api_key, self.pinecone_api_key, self.pinecone_index_name]):
            raise ValueError(
                "❌ Erreur Critique : Variables d'environnement manquantes !"
            )

        print("⏳ Initialisation du Moteur RAG...")

        # 2. Configuration du LLM
        Settings.llm = Groq(model="llama-3.1-8b-instant", api_key=self.groq_api_key)

        # 3. Configuration des Embeddings (BGE-Small) - Exécution locale dans le conteneur
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

        # 4. Connexion à Pinecone
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.pinecone_index = self.pc.Index(self.pinecone_index_name)
        self.vector_store = PineconeVectorStore(pinecone_index=self.pinecone_index)
        
        # 5. Chargement de l'index
        self.index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store)
        
        # 6. Configuration du prompt mis à jour
        text_qa_template = PromptTemplate(SISTEM_PROMPT_TEMPLATE)
        
        # 7. Création du moteur de requêtage avec intégration du Prompt (top_k calé à 5 pour Groq TPM)
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=5,
            text_qa_template=text_qa_template
        )
        print("✅ Moteur RAG prêt à recevoir des requêtes !")

    def _enrichir_question(self, question: str) -> str:
        """Nettoie la ponctuation et applique l'expansion d'acronymes bilingues de manière robuste."""
        acronymes = {
            # Acronymes Français
            "is": "impôt sur les sociétés",
            "tva": "taxe sur la valeur ajoutée",
            "irpp": "impôt sur le revenu des personnes physiques",
            "cac": "centimes additionnels communaux",
            "cgi": "code général des impôts",
            "lfi": "loi de finances",
            
            # Acronymes Anglais (Traduits pour correspondre sémantiquement à la base FR)
            "cit": "impôt sur les sociétés corporate income tax",
            "vat": "taxe sur la valeur ajoutée value added tax"
        }
        
        # Nettoyage agressif des apostrophes et ponctuations pour isoler les acronymes (ex: "l'is" -> "l is")
        question_nettoyee = question.lower()
        for char in ["'", "’", "?", "!", ".", ",", ";", ":", "(", ")"]:
            question_nettoyee = question_nettoyee.replace(char, " ")
            
        mots_question = question_nettoyee.split()
        enrichissements = []
        
        for acronyme, texte_complet in acronymes.items():
            if acronyme in mots_question:
                enrichissements.append(texte_complet)
                
        if enrichissements:
            extension = " , ".join(enrichissements)
            question_enrichie = f"{question.strip()} ({extension})"
            print(f"🔍 Requête enrichie transmise à Pinecone : {question_enrichie}")
            return question_enrichie
            
        return question

    def _formater_reponse_avec_sources(self, response) -> str:
        """Méthode utilitaire partagée pour extraire les sources et structurer la réponse."""
        texte_ia = str(response)
        sources_utilisees = set()
        
        if hasattr(response, "source_nodes"):
            for node in response.source_nodes:
                nom_doc = node.node.metadata.get("source_nom", "Source inconnue")
                num_page = node.node.metadata.get("page_numero", "")
                if num_page:
                    sources_utilisees.add(f"{nom_doc} (Page {num_page})")
                else:
                    sources_utilisees.add(nom_doc)
        
        reponse_finale = texte_ia
        if sources_utilisees:
            reponse_finale += "\n\n📋 *Sources consultées :*\n"
            for src in sources_utilisees:
                reponse_finale += f"- _{src}_\n"
        
        return reponse_finale

    async def generer_reponse_async(self, question: str) -> str:
        """Interroge l'index de manière ASYNCHRONE pour FastAPI avec expansion des acronymes."""
        try:
            # Application du filtre d'acronymes
            question = self._enrichir_question(question)

            # .aquery() est la méthode native asynchrone de LlamaIndex
            response = await self.query_engine.aquery(question)
            return self._formater_reponse_avec_sources(response)
            
        except Exception as e:
            print(f"❌ Erreur lors du traitement asynchrone de la requête RAG : {str(e)}")
            return (
                "Désolé, je rencontre actuellement une difficulté technique pour analyser "
                "le Code Général des Impôts. Veuillez réessayer dans quelques instants."
            )

    def generer_reponse(self, question: str) -> str:
        """Interroge l'index de manière synchrone (Enrichi aussi pour le fallback synchrone)."""
        try:
            # Application du filtre d'acronymes
            question = self._enrichir_question(question)

            response = self.query_engine.query(question)
            return self._formater_reponse_avec_sources(response)
            
        except Exception as e:
            print(f"❌ Erreur lors du traitement synchrone de la requête RAG : {str(e)}")
            return (
                "Désolé, je rencontre actuellement une difficulté technique pour analyser "
                "le Code Général des Impôts. Veuillez réessayer dans quelques instants."
            )

# Singleton
moteur_rag = MoteurRAG()