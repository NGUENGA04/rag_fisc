import os
import httpx
from fastapi import FastAPI, Request, Response, BackgroundTasks, Query
from dotenv import load_dotenv

load_dotenv()

# Variables Meta Cloud API
META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN")  # Un mot de passe arbitraire que tu choisis (ex: "MonSecret123")

app = FastAPI(title="ConsuFiscal Meta API")

async def envoyer_message_meta(numero_destinataire: str, texte: str):
    """Envoie un message WhatsApp via l'API officielle de Meta."""
    url = f"https://graph.facebook.com/v17.0/{PHONE_NUMBER_ID}/messages"
    
    headers = {
        "Authorization": f"Bearer {META_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "messaging_product": "whatsapp",
        "to": numero_destinataire,
        "type": "text",
        "text": {"body": texte}
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, headers=headers, timeout=10.0)
            if response.status_code != 200:
                print(f"❌ Erreur Meta API: {response.text}")
        except Exception as e:
            print(f"❌ Erreur réseau Meta: {str(e)}")

# 1. ÉTAPE DE VÉRIFICATION DU WEBHOOK (Exigée par Meta lors de la configuration)
@app.get("/webhook/whatsapp")
async def verifier_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_challenge: str = Query(None, alias="hub.challenge"),
    hub_verify_token: str = Query(None, alias="hub.verify_token")
):
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        return Response(content=hub_challenge, media_type="text/plain")
    return Response(content="Échec de la vérification", status_code=403)

# 2. RÉCEPTION DES MESSAGES DE META
@app.post("/webhook/whatsapp")
async def recevoir_message_meta(request: Request, background_tasks: BackgroundTasks):
    body = await request.json()
    
    try:
        # Structure d'extraction du JSON très spécifique à Meta
        entry = body["entry"][0]
        changes = entry["changes"][0]
        value = changes["value"]
        message = value["messages"][0]
        
        message_utilisateur = message["text"]["body"].strip()
        numero_expediteur = message["from"]  # Format international complet (ex: 237692001642)

        # Ici tu lances ton traitement RAG habituel en tâche de fond
        # ex: background_tasks.add_task(traiter_rag_et_repondre, message_utilisateur, numero_expediteur)
        print(f"📩 Message reçu de {numero_expediteur} : {message_utilisateur}")
        
    except (KeyError, IndexError):
        pass

    return Response(status_code=200)