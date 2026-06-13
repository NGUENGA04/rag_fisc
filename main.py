import os
import httpx
from fastapi import FastAPI, Request, Response, BackgroundTasks
from dotenv import load_dotenv

# Chargement des variables d'environnement locales (.env)
load_dotenv()

META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN", "123456")

app = FastAPI(
    title="ConsulFiscal Pro - API", 
    version="1.2.0",
    description="Système RAG d'analyse fiscale connecté à l'API Cloud WhatsApp de Meta"
)

# --- 1. ROUTE DE HEALTH CHECK ---
@app.get("/health", tags=["Diagnostic"])
async def health_check():
    """Vérifie l'état de l'API et la configuration des clés Meta."""
    status_meta = "Configuré ✅" if (PHONE_NUMBER_ID and META_ACCESS_TOKEN) else "Incomplet ❌"
    return {
        "status": "healthy",
        "app": "ConsulFiscal Pro",
        "infrastructure": "Local via Ngrok",
        "meta_integration": status_meta
    }


# --- 2. FONCTION D'ENVOI VERS META ---
async def envoyer_message_whatsapp_meta(numero_destinataire: str, texte: str):
    """Envoie la réponse finale calculée sur le WhatsApp de l'utilisateur."""
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
    
    async with httpx.AsyncClient(trust_env=False) as client:
        try:
            response = await client.post(url, json=payload, headers=headers, timeout=15.0)
            print(f"📊 RETOUR API META (Status: {response.status_code}) : {response.text}")
        except Exception as e:
            print(f"❌ CAUSE EXACTE DE L'ÉCHEC RÉSEAU : {repr(e)}")


# --- 3. PIPELINE RAG (À RECONNECTER À TON MOTEUR) ---
async def executer_pipeline_rag(message_utilisateur: str, numero_expediteur: str):
    """Gère l'interrogation de la base documentaire fiscale et déclenche l'envoi."""
    print(f"🧠 Lancement du RAG pour le message : '{message_utilisateur}'")
    
    try:
        # =========================================================================
        # 🔌 EMPLACEMENT DE TA CONNEXION RAG (LlamaIndex / LangChain)
        # =========================================================================
        # Exemple de reconnexon à ton vrai moteur de recherche de documents :
        #
        # from mon_moteur_rag import query_engine
        # response = query_engine.query(message_utilisateur)
        # reponse_fiscale = str(response)
        # =========================================================================
        
        # Simulation actuelle de traitement intelligent
        if "tva" in message_utilisateur.lower():
            reponse_fiscale = "Le taux standard de la TVA au Cameroun est de 19,25% (17,5% en principal + 10% de Centimes Additionnels Communaux)."
        else:
            reponse_fiscale = f"J'ai bien reçu votre requête concernant : '{message_utilisateur}'. Le moteur d'analyse examine le Code Général des Impôts."

    except Exception as e:
        reponse_fiscale = "⚠️ Une erreur technique est survenue lors de la génération de la réponse fiscale."
        print(f"❌ Erreur interne moteur RAG : {str(e)}")

    # Formatage de l'émetteur pour WhatsApp (Style soigné pour le jury)
    message_final = f"🤖 *ConsulFiscal Pro*\n\n{reponse_fiscale}"
    
    # Envoi effectif
    await envoyer_message_whatsapp_meta(numero_expediteur, message_final)


# --- 4. WEBHOOK : VALIDATION GET (Poignée de main Meta) ---
@app.get("/webhook/whatsapp", tags=["Webhook"])
async def verifier_webhook(request: Request):
    params = request.query_params
    hub_mode = params.get("hub.mode")
    hub_challenge = params.get("hub.challenge")
    hub_verify_token = params.get("hub.verify_token")
    
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        print("✅ Poignée de main Meta validée avec succès !")
        return Response(content=str(hub_challenge), media_type="text/plain")
    
    print("❌ Échec de la poignée de main GET")
    return Response(content="Échec de la vérification", status_code=403)


# --- 5. WEBHOOK : RÉCEPTION POST (Interception des messages) ---
@app.post("/webhook/whatsapp", tags=["Webhook"])
async def recevoir_message_whatsapp(request: Request, background_tasks: BackgroundTasks):
    try:
        body = await request.json()
        
        entry = body.get("entry", [{}])[0]
        changes = entry.get("changes", [{}])[0]
        value = changes.get("value", {})
        
        # On extrait le message s'il existe (évite de traiter les statuts "read"/"delivered")
        if "messages" in value:
            message_data = value["messages"][0]
            message_utilisateur = message_data.get("text", {}).get("body", "").strip()
            numero_expediteur = message_data.get("from")
            
            # Sécurité anti-chaîne vide pour ne capturer que le texte de l'utilisateur
            if message_utilisateur and numero_expediteur:
                print(f"📩 Message intercepté de {numero_expediteur} : '{message_utilisateur}'")
                
                # Planification de la tâche de fond
                background_tasks.add_task(executer_pipeline_rag, message_utilisateur, numero_expediteur)
                print("⏳ Tâche de fond ajoutée au pipeline RAG.")
                
    except Exception as e:
        print(f"❌ Erreur lors du traitement du webhook POST : {str(e)}")

    # On renvoie toujours le 200 OK requis par Meta
    return Response(status_code=200)