import os
from dotenv import load_dotenv
from pinecone import Pinecone
from llama_index.core import Settings, VectorStoreIndex, PromptTemplate
from llama_index.llms.groq import Groq
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1. Chargement des variables d'environnement
load_dotenv()

# Définition du template de prompt corrigé et optimisé
SISTEM_PROMPT_TEMPLATE = (
    "Tu es ConsuFiscal, un assistant virtuel expert fiscaliste spécialisé dans le Code Général des Impôts (CGI) du Cameroun.\n"
    "Voici les extraits de lois officiels fournis pour t'aider :\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Instructions impératives de sécurité :\n"
    "1. Réponds de manière professionnelle, claire, polie et bien structurée (format Markdown WhatsApp : *gras*, _italique_).\n"
    "2. Base-toi EXCLUSIVEMENT sur les faits explicitement écrits dans le contexte fourni. Ne fais/n'invente JAMAIS de déductions, d'interprétations ou de suppositions.\n"
    "3. Si un texte mentionne une valeur monétaire (ex: 10 000 FCFA), ne transforme JAMAIS cette valeur en pourcentage ou en taux.\n"
    "4. Si le chiffre exact du taux (général, réduit ou spécifique) demandé n'est pas écrit noir sur blanc dans le contexte fourni, dis STRICTEMENT et poliment :\n"
    "'Désolé, ce taux de TVA n'est pas mentionné de manière chiffrée dans les extraits de lois fournis. Je ne trouve pas de disposition précise dans les textes fiscaux actuels pour répondre à votre demande.'\n"
    "5. Ne suggère jamais de taux hypothétique..\n\n"
    "Question de l'utilisateur : {query_str}\n"
    "Réponse de l'expert ConsuFiscal :"
)

class MoteurRAG:
    def __init__(self):
        """Initialise l'architecture RAG."""
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
        self.memoire_conversations = {}
        self.max_tours_memoire = 4
        self.ready = False
        self.initialization_error = None
        
        if not all([self.groq_api_key, self.pinecone_api_key, self.pinecone_index_name]):
            self.initialization_error = (
                "❌ Variables d'environnement manquantes pour le moteur RAG. "
                "La mémoire reste disponible, mais le moteur ne peut pas répondre tant que la configuration est incomplète."
            )
            print(self.initialization_error)
            return

        print("⏳ Initialisation du Moteur RAG...")

        # 2. Configuration du LLM
        self.llm = Groq(model="llama-3.1-8b-instant", api_key=self.groq_api_key, temperature=0.0)
        Settings.llm = self.llm

        # 3. Configuration des Embeddings (BGE-Small) - Exécution locale
        self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        Settings.embed_model = self.embed_model

        # 4. Connexion à Pinecone
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.pinecone_index = self.pc.Index(self.pinecone_index_name)
        self.vector_store = PineconeVectorStore(pinecone_index=self.pinecone_index)
        
        # 5. Chargement de l'index
        self.index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store)
        
        # 6. Configuration du prompt mis à jour
        self.text_qa_template = PromptTemplate(SISTEM_PROMPT_TEMPLATE)
        
        # 7. Création du moteur de requêtage (top_k optimisé)
        # Note : similarity_top_k=4 réduit la charge de tokens tout en gardant le contexte nécessaire
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=4,
            text_qa_template=self.text_qa_template
        )
        self.ready = True
        print("✅ Moteur RAG prêt à recevoir des requêtes !")

    def _enrichir_question(self, question: str) -> str:
        """
        Nettoyage et expansion conceptuelle stricte pour optimiser la recherche vectorielle.
        """
        acronymes = {
            "is": "is impôt sociétés",
            "tva": "tva taxe valeur ajoutée",
            "irpp": "irpp impôt revenu personnes physiques",
            "cac": "cac centimes additionnels communaux",
            "cgi": "cgi code général impôts",
            "lfi": "lfi loi finances",
            "cit": "cit impôt sociétés",
            "vat": "vat taxe valeur ajoutée"
        }
        
        # 1. Nettoyage de la ponctuation (avec gestion propre des espaces blancs excédentaires)
        question_nettoyee = question.lower()
        for char in ["'", "’", "?", "!", ".", ",", ";", ":", "(", ")", "-", "_"]:
            question_nettoyee = question_nettoyee.replace(char, " ")
            
        mots_question = question_nettoyee.split()
        acronyme_trouve = False
        
        # 2. Remplacement strict
        mots_finaux = []
        for mot in mots_question:
            if mot in acronymes:
                mots_finaux.append(acronymes[mot])
                acronyme_trouve = True
            else:
                mots_finaux.append(mot)
                
        # 3. Reconstruction de la question
        if acronyme_trouve:
            question_enrichie = " ".join(mots_finaux).strip()
            print(f"🎯 Requête conceptuelle transmise à Pinecone : {question_enrichie}")
            return question_enrichie
            
        return question

    def _construire_contexte_memoire(self, conversation_id: str | None) -> str:
        """Construit un contexte de conversation très léger à partir de l'historique récent."""
        if not conversation_id:
            return ""

        historique = self.memoire_conversations.get(conversation_id, [])
        if not historique:
            return ""

        messages = []
        for echange in historique[-self.max_tours_memoire:]:
            messages.append(f"Utilisateur : {echange['question']}\nAssistant : {echange['reponse']}")

        return "Historique de conversation récent :\n" + "\n\n".join(messages)

    def ajouter_echange(self, conversation_id: str, question: str, reponse: str) -> None:
        """Ajoute un échange à la mémoire locale de la conversation."""
        if not conversation_id:
            return

        historique = self.memoire_conversations.setdefault(conversation_id, [])
        historique.append({"question": question, "reponse": reponse})

        if len(historique) > self.max_tours_memoire:
            historique.pop(0)

    def _preparer_question(self, question: str, conversation_id: str | None) -> str:
        """Enrichit la question avec un contexte très court issu de la mémoire locale."""
        question_traitee = self._enrichir_question(question)
        contexte_memoire = self._construire_contexte_memoire(conversation_id)

        if contexte_memoire:
            return f"{contexte_memoire}\n\nNouvelle question utilisateur : {question_traitee}"

        return question_traitee

    def _formater_reponse_avec_sources(self, response) -> str:
        """Méthode utilitaire pour extraire les sources uniques et structurer la réponse."""
        texte_ia = str(response).strip()
        sources_utilisees = set()
        
        if hasattr(response, "source_nodes"):
            for node in response.source_nodes:
                metadata = node.node.metadata
                nom_doc = metadata.get("source_nom", "Source inconnue")
                num_page = metadata.get("page_numero", "")
                
                if num_page:
                    sources_utilisees.add(f"{nom_doc} (Page {num_page})")
                else:
                    sources_utilisees.add(nom_doc)
        
        if sources_utilisees:
            # Tri des sources pour un affichage propre et stable
            sources_triees = sorted(list(sources_utilisees))
            liste_sources = "\n".join([f"- _{src}_" for src in sources_triees])
            return f"{texte_ia}\n\n📋 *Sources consultées :*\n{liste_sources}"
        
        return texte_ia

    async def generer_reponse_async(self, question: str, conversation_id: str | None = None) -> str:
        """Interroge l'index de manière ASYNCHRONE pour FastAPI, avec mémoire locale de conversation."""
        if not self.ready:
            return self.initialization_error or "Le moteur RAG n'est pas encore prêt."

        try:
            conversation_id = conversation_id or "default"
            question_traitee = self._preparer_question(question, conversation_id)
            response = await self.query_engine.aquery(question_traitee)
            reponse_formatee = self._formater_reponse_avec_sources(response)
            self.ajouter_echange(conversation_id, question, reponse_formatee)
            return reponse_formatee
            
        except Exception as e:
            print(f"❌ Erreur lors du traitement asynchrone de la requête RAG : {str(e)}")
            return (
                "Désolé, je rencontre actuellement une difficulté technique pour analyser "
                "le Code Général des Impôts. Veuillez réessayer dans quelques instants."
            )

    def generer_reponse(self, question: str, conversation_id: str | None = None) -> str:
        """Interroge l'index de manière synchrone (Fallback / Tests), avec mémoire locale de conversation."""
        if not self.ready:
            return self.initialization_error or "Le moteur RAG n'est pas encore prêt."

        try:
            conversation_id = conversation_id or "default"
            question_traitee = self._preparer_question(question, conversation_id)
            response = self.query_engine.query(question_traitee)
            reponse_formatee = self._formater_reponse_avec_sources(response)
            self.ajouter_echange(conversation_id, question, reponse_formatee)
            return reponse_formatee
            
        except Exception as e:
            print(f"❌ Erreur lors du traitement synchrone de la requête RAG : {str(e)}")
            return (
                "Désolé, je rencontre actuellement une difficulté technique pour analyser "
                "le Code Général des Impôts. Veuillez réessayer dans quelques instants."
            )

# Initialisation du Singleton
moteur_rag = MoteurRAG()