import os
import httpx
from fastapi import FastAPI, Request, Response, BackgroundTasks
from dotenv import load_dotenv

load_dotenv()

# Configuration des secrets Meta (récupérés depuis tes variables Hugging Face)
META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN", "123456")

app = FastAPI(title="ConsulFiscal Pro - Meta API")

# --- 1. FONCTION D'ENVOI VERS META ---
async def envoyer_message_whatsapp_meta(numero_destinataire: str, texte: str):
    if not PHONE_NUMBER_ID or not META_ACCESS_TOKEN:
        print("❌ Envoi annulé : PHONE_NUMBER_ID ou META_ACCESS_TOKEN vide.")
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
    
    print(f"📡 Envoi vers Meta... Destination: {numero_destinataire}")
    
    # trust_env=False force httpx à ignorer les proxys d'Hugging Face qui peuvent bloquer
    async with httpx.AsyncClient(trust_env=False) as client:
        try:
            response = await client.post(url, json=payload, headers=headers, timeout=30.0)
            print(f"📊 RETOUR API META (Status: {response.status_code}) : {response.text}")
        except Exception as e:
            # ICI LE REPR(E) VA ENFIN TOUT T'AFFICHER NOIR SUR BLANC
            print(f"❌ CAUSE EXACTE DE L'ÉCHEC RÉSEAU : {repr(e)}")

# --- 2. FONCTION DU PIPELINE RAG (Définie avant son utilisation) ---
async def executer_pipeline_rag(message_utilisateur: str, numero_expediteur: str):
    """Exécute la recherche fiscale et envoie la réponse."""
    print(f"🧠 Lancement du RAG pour le message : '{message_utilisateur}'")
    
    try:
        # --- ICI : METS TON CODE APPEL RAG EXISTANT ---
        # Exemple de simulation en attendant d'exécuter ton vrai script :
        # reponse_fiscale = "Le taux standard de la TVA au Cameroun est de 19,25% (17,5% principal + 10% CAC)."
        
        # Si tu as importé ton vrai moteur RAG (ex: query_engine), décommente et ajuste la ligne ci-dessous :
        # response = query_engine.query(message_utilisateur)
        # reponse_fiscale = str(response)
        
        reponse_fiscale = f"Merci pour votre question : '{message_utilisateur}'. Le moteur RAG est en cours de traitement."
        
    except Exception as e:
        reponse_fiscale = "⚠️ Une erreur technique est survenue dans le traitement fiscal."
        print(f"❌ Erreur moteur RAG : {str(e)}")

    # Formatage de la réponse pour WhatsApp
    message_final = f"🤖 *ConsulFiscal Pro*\n\n{reponse_fiscale}"
    
    # Envoi de la réponse sur le téléphone
    await envoyer_message_whatsapp_meta(numero_expediteur, message_final)


# --- 3. ROUTE DE VALIDATION GET (La poignée de main) ---
@app.get("/webhook/whatsapp")
async def verifier_webhook(request: Request):
    params = request.query_params
    hub_mode = params.get("hub.mode")
    hub_challenge = params.get("hub.challenge")
    hub_verify_token = params.get("hub.verify_token")
    
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        return Response(content=str(hub_challenge), media_type="text/plain")
    return Response(content="Échec de la vérification", status_code=403)


# --- 4. ROUTE DE RÉCEPTION POST ---
@app.post("/webhook/whatsapp")
async def recevoir_message_whatsapp(request: Request, background_tasks: BackgroundTasks):
    try:
        body = await request.json()
        
        # Extraction sécurisée des données du dictionnaire Meta
        entry = body.get("entry", [{}])[0]
        changes = entry.get("changes", [{}])[0]
        value = changes.get("value", {})
        
        if "messages" in value:
            message_data = value["messages"][0]
            message_utilisateur = message_data.get("text", {}).get("body", "").strip()
            numero_expediteur = message_data.get("from")
            
            print(f"📩 Message intercepté de {numero_expediteur} : '{message_utilisateur}'")
            
            if message_utilisateur and numero_expediteur:
                # Ajout de la tâche de fond (Python connaît maintenant la fonction !)
                background_tasks.add_task(executer_pipeline_rag, message_utilisateur, numero_expediteur)
                print("⏳ Tâche de fond ajoutée au pipeline RAG.")
                
        else:
            # Ignore les statuts "sent", "delivered", "read" envoyés par Meta
            pass
            
    except Exception as e:
        print(f"❌ Erreur lors du traitement du webhook : {str(e)}")

    return Response(status_code=200)