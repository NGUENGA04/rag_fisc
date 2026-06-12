import os
import logging
import httpx
from fastapi import FastAPI, Form, Response, BackgroundTasks
from dotenv import load_dotenv

# Importation du moteur RAG configuré
from moteur_rag import moteur_rag

load_dotenv()

# Configuration des logs pour suivre l'activité dans le terminal
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Récupération des variables Green-API depuis le fichier .env
ID_INSTANCE = os.getenv("GREEN_API_ID_INSTANCE")
API_TOKEN = os.getenv("GREEN_API_TOKEN")

# Ton numéro fixe pour centraliser tes tests (si besoin)
MON_NUMERO_WHATSAPP = "237692001642"

app = FastAPI(
    title="ConsuFiscal API",
    description="Backend RAG pour l'analyse du Code des Impôts via Green-API",
    version="1.0.0"
)

async def envoyer_reponse_green_api(chat_id: str, message_texte: str):
    """
    Fonction asynchrone qui effectue la requête HTTP POST vers Green-API
    pour envoyer la réponse du RAG sur WhatsApp.
    """
    if not ID_INSTANCE or not API_TOKEN:
        logger.error("❌ Variables d'environnement Green-API manquantes dans le .env.")
        return

    url = f"https://api.green-api.com/waInstance{ID_INSTANCE}/sendMessage/{API_TOKEN}"
    
    payload = {
        "chatId": chat_id,
        "message": message_texte
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, timeout=15.0)
            if response.status_code == 200:
                logger.info(f"🚀 Réponse fiscale envoyée avec succès à {chat_id}")
            else:
                logger.error(f"❌ Échec de l'envoi Green-API ({response.status_code}) : {response.text}")
        except Exception as e:
            logger.error(f"❌ Erreur réseau lors de la connexion à Green-API : {str(e)}")


async def traiter_et_repondre_rag(message_utilisateur: str, chat_id: str):
    """
    Tâche exécutée en arrière-plan : interroge le RAG et appelle Green-API.
    """
    if message_utilisateur:
        try:
            logger.info(f"🧠 Interrogation du moteur RAG pour : '{message_utilisateur}'")
            reponse_fiscale = await moteur_rag.generer_reponse_async(message_utilisateur)
        except Exception as e:
            reponse_fiscale = "⚠️ Une erreur technique est survenue lors de l'analyse fiscale."
            logger.error(f"❌ Erreur lors du traitement de la requête RAG : {str(e)}")
    else:
        reponse_fiscale = "Désolé, je n'ai pas pu lire le contenu de votre message."

    # Formatage final du message pour WhatsApp
    entete = "🤖 *ConsulFiscal Pro*\n\n"
    message_final = f"{entete}{reponse_fiscale}"
    
    # Envoi du message via Green-API
    await envoyer_reponse_green_api(chat_id, message_final)


@app.get("/health")
async def health_check(q: str = "Bonjour"):
    try:
        reponse = await moteur_rag.generer_reponse_async(q)
        return {
            "status": "healthy",
            "moteur": "Llama 3.1 via Groq",
            "test_question": q,
            "test_reponse": reponse
        }
    except Exception as e:
        return {"status": "error", "details": str(e)}


@app.post("/webhook/whatsapp")
async def whatsapp_webhook(background_tasks: BackgroundTasks, body: dict = None):
    """
    Webhook universel pour Green-API.
    Green-API envoie des données au format JSON (contrairement à Twilio qui envoyait du Form).
    """
    # On vérifie qu'il s'agit bien d'un événement de type 'message reçu'
    if not body or "messageData" not in body:
        return Response(status_code=200)

    try:
        # Extraction du texte et de l'identifiant de la discussion (chatId)
        message_utilisateur = body["messageData"]["textMessageData"]["textMessage"].strip()
        chat_id = body["senderData"]["chatId"]
        
        logger.info(f"📩 Message reçu de {chat_id} : '{message_utilisateur}'")
        
        # On lance le traitement RAG + Envoi WhatsApp en tâche de fond
        background_tasks.add_task(traiter_et_repondre_rag, message_utilisateur, chat_id)
        
    except KeyError:
        # On ignore les messages qui ne sont pas du texte brut (images, audios, etc.)
        pass

    # Green-API attend juste un statut HTTP 200 pour confirmer la réception du webhook
    return Response(status_code=200)