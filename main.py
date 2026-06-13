import os
import httpx
from fastapi import FastAPI, Request, Response, BackgroundTasks
from dotenv import load_dotenv

load_dotenv()

# Configuration des secrets Meta
META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN", "123456")

app = FastAPI(title="ConsulFiscal Pro - Meta API", version="1.1.0")

# Vérification de sécurité au démarrage dans les logs
print("⚙️ [INITIALISATION] Vérification des variables d'environnement :")
print(f"   - PHONE_NUMBER_ID : {'✅ Configuré' if PHONE_NUMBER_ID else '❌ MANQUANT (Vérifie tes Secrets HF)'}")
print(f"   - META_ACCESS_TOKEN : {'✅ Configuré' if META_ACCESS_TOKEN else '❌ MANQUANT (Vérifie tes Secrets HF)'}")
print(f"   - META_VERIFY_TOKEN : {VERIFY_TOKEN}")

# --- 1. FONCTION D'ENVOI OPTIMISÉE ---
async def envoyer_message_whatsapp_meta(numero_destinataire: str, texte: str):
    if not PHONE_NUMBER_ID or not META_ACCESS_TOKEN:
        print("❌ Envoi annulé : PHONE_NUMBER_ID ou META_ACCESS_TOKEN est vide.")
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
        "text": {"body": texte}
    }
    
    print(f"📡 Tentative d'envoi à Meta (Timeout étendu à 30s)... URL: {url}")
    
    # Configuration d'un client HTTP limits et timeout renforcés
    limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
    async with httpx.AsyncClient(limits=limits, trust_env=True) as client:
        try:
            response = await client.post(
                url, 
                json=payload, 
                headers=headers, 
                timeout=30.0  # On passe à 30 secondes pour laisser le temps au proxy HF
            )
            print(f"📊 RETOUR API META (Status: {response.status_code}) : {response.text}")
        except httpx.ConnectTimeout:
            print("❌ Erreur : Temps de connexion dépassé (ConnectTimeout). Hugging Face bloque la sortie vers Meta.")
        except Exception as e:
            print(f"❌ Autre erreur réseau : {repr(e)}")


# --- 3. ROUTE DE VALIDATION GET (Handshake) ---
@app.get("/webhook/whatsapp")
async def verifier_webhook(request: Request):
    params = request.query_params
    hub_mode = params.get("hub.mode")
    hub_challenge = params.get("hub.challenge")
    hub_verify_token = params.get("hub.verify_token")
    
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        print("✅ Poignée de main Meta réussie !")
        return Response(content=str(hub_challenge), media_type="text/plain")
    
    print("❌ Échec de la poignée de main GET : Tokens non alignés.")
    return Response(content="Échec de la vérification", status_code=403)


# --- 4. ROUTE DE RÉCEPTION POST (Webhook) ---
@app.post("/webhook/whatsapp")
async def recevoir_message_whatsapp(request: Request, background_tasks: BackgroundTasks):
    try:
        body = await request.json()
        
        # Parcours sécurisé du JSON hautement imbriqué de Meta
        entry = body.get("entry", [{}])[0]
        changes = entry.get("changes", [{}])[0]
        value = changes.get("value", {})
        
        if "messages" in value:
            message_data = value["messages"][0]
            message_utilisateur = message_data.get("text", {}).get("body", "").strip()
            numero_expediteur = message_data.get("from")
            
            print(f"📩 Message intercepté de {numero_expediteur} : '{message_utilisateur}'")
            
            if message_utilisateur and numero_expediteur:
                # Ajout immédiat de la tâche en arrière-plan pour libérer Meta rapidement
                background_tasks.add_task(executer_pipeline_rag, message_utilisateur, numero_expediteur)
                print("⏳ Tâche de fond ajoutée au pipeline RAG avec succès.")
        else:
            # Ignore les événements secondaires (ex: accusés de réception "read", "delivered")
            pass
            
    except Exception as e:
        print(f"❌ Erreur critique lors de l'interception du webhook : {str(e)}")

    # Toujours renvoyer un 200 OK à Meta pour éviter qu'il ne sature ton webhook
    return Response(status_code=200)