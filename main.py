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
async def verifier_webhook(request: Request):
    # On récupère directement les paramètres bruts de l'URL
    params = request.query_params
    
    hub_mode = params.get("hub.mode")
    hub_challenge = params.get("hub.challenge")
    hub_verify_token = params.get("hub.verify_token")
    
    # Sécurité : On affiche dans tes logs Hugging Face ce que ton code reçoit VRAIMENT
    print(f"🔍 Mode reçu: {hub_mode} | Token reçu: {hub_verify_token} | Challenge: {hub_challenge}")
    
    # REMPLACE "123456" par ton vrai token secret si tu l'as changé dans l'interface de Meta
    # Ici, d'après tes logs, Meta envoie "123456"
    VERIFY_TOKEN_ATTENDU = "123456" 
    
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN_ATTENDU:
        print("✅ Jeton validé ! Envoi du challenge à Meta...")
        return Response(content=str(hub_challenge), media_type="text/plain")
    
    print("❌ Échec de la validation : les jetons ne correspondent pas.")
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