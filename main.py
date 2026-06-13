import os
import logging
import httpx
from fastapi import FastAPI, Request, Response, BackgroundTasks
from dotenv import load_dotenv

# Importation du moteur RAG configuré connecté à Pinecone
from moteur_rag import moteur_rag

load_dotenv()

# Configuration des logs pour le terminal de soutenance
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Récupération des variables d'environnement Meta Cloud API
META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN", "123456")

app = FastAPI(
    title="ConsulFiscal API",
    description="Backend RAG pour l'analyse du Code des Impôts via Meta WhatsApp Cloud API",
    version="2.0.0"
)

async def envoyer_reponse_whatsapp_meta(numero_destinataire: str, message_texte: str):
    """
    Effectue la requête HTTP POST officielle vers les serveurs de Meta 
    pour distribuer la réponse du RAG sur le WhatsApp de l'utilisateur.
    """
    if not PHONE_NUMBER_ID or not META_ACCESS_TOKEN:
        logger.error("❌ Variables d'environnement Meta manquantes dans le .env.")
        return

    url = f"https://graph.facebook.com/v17.0/{PHONE_NUMBER_ID}/messages"
    
    headers = {
        "Authorization": f"Bearer {META_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "messaging_product": "whatsapp",
        "to": numero_destinataire,
        "type": "text",
        "text": {"body": message_texte}
    }
    
    logger.info(f"📡 Envoi vers Meta... Destination: {numero_destinataire}")
    
    # trust_env=False ignore les proxys d'hébergements tiers (ex: Hugging Face) pour éviter les blocages
    async with httpx.AsyncClient(trust_env=False) as client:
        try:
            response = await client.post(url, json=payload, headers=headers, timeout=15.0)
            if response.status_code in [200, 201]:
                logger.info(f"🚀 Réponse fiscale envoyée avec succès (Status: {response.status_code})")
            else:
                logger.error(f"❌ Échec Meta ({response.status_code}) : {response.text}")
        except Exception as e:
            logger.error(f"❌ Erreur réseau lors de la connexion à l'API Meta : {repr(e)}")


async def traiter_et_repondre_rag(message_utilisateur: str, numero_expediteur: str):
    """
    Tâche exécutée en arrière-plan : interroge le RAG local et déclenche l'envoi Meta.
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
    message_final = f"🤖 *ConsulFiscal Pro*\n\n{reponse_fiscale}"
    
    # Envoi direct via l'infrastructure Meta
    await envoyer_reponse_whatsapp_meta(numero_expediteur, message_final)


@app.get("/health", tags=["Diagnostic"])
async def health_check(q: str = "Bonjour"):
    """Route de diagnostic pour valider l'intégrité opérationnelle du RAG."""
    try:
        reponse = await moteur_rag.generer_reponse_async(q)
        status_meta = "Configuré ✅" if (PHONE_NUMBER_ID and META_ACCESS_TOKEN) else "Incomplet ❌"
        return {
            "status": "healthy",
            "moteur": "Llama 3.1 via Groq & Pinecone",
            "meta_integration": status_meta,
            "test_question": q,
            "test_reponse": reponse
        }
    except Exception as e:
        return {"status": "error", "details": str(e)}


@app.get("/webhook/whatsapp", tags=["Webhook"])
async def verifier_webhook(request: Request):
    """
    Vérification GET obligatoire requise par Meta (Verification Request).
    Sert de poignée de main initiale pour valider ton URL ngrok.
    """
    params = request.query_params
    hub_mode = params.get("hub.mode")
    hub_challenge = params.get("hub.challenge")
    hub_verify_token = params.get("hub.verify_token")
    
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        logger.info("✅ Poignée de main Meta validée avec succès !")
        return Response(content=str(hub_challenge), media_type="text/plain")
    
    logger.error("❌ Échec de la poignée de main GET Meta.")
    return Response(content="Échec de la vérification", status_code=403)


@app.post("/webhook/whatsapp", tags=["Webhook"])
async def recevoir_message_whatsapp(request: Request, background_tasks: BackgroundTasks):
    """
    Réception POST des événements WhatsApp.
    Filtre les notifications systèmes pour n'extraire que les messages textuels.
    """
    try:
        body = await request.json()
        
        # Structure de parsing sécurisée du dictionnaire imbriqué de Meta
        entry = body.get("entry", [{}])[0]
        changes = entry.get("changes", [{}])[0]
        value = changes.get("value", {})
        
        # Traitement exclusif s'il s'agit d'un message entrant (évite les statuts read/delivered)
        if "messages" in value:
            message_data = value["messages"][0]
            message_utilisateur = message_data.get("text", {}).get("body", "").strip()
            numero_expediteur = message_data.get("from")
            
            if message_utilisateur and numero_expediteur:
                logger.info(f"📩 Message intercepté de {numero_expediteur} : '{message_utilisateur}'")
                
                # Allocation de la tâche asynchrone pour libérer le webhook immédiatement
                background_tasks.add_task(traiter_et_repondre_rag, message_utilisateur, numero_expediteur)
                logger.info("⏳ Tâche de fond ajoutée au pipeline RAG.")
                
    except Exception as e:
        logger.error(f"❌ Erreur lors du traitement du webhook POST Meta : {str(e)}")

    # Meta exige impérativement un code HTTP 200 immédiat
    return Response(status_code=200)