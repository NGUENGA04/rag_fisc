import os
import nest_asyncio
from fastapi import FastAPI, Form, Response
from fastapi.responses import Response
from dotenv import load_dotenv

# Appliquer nest_asyncio au tout début pour autoriser les boucles imbriquées (LlamaIndex dans FastAPI)
nest_asyncio.apply()

# Importation du moteur RAG configuré
from moteur_rag import moteur_rag

load_dotenv()

app = FastAPI(
    title="ConsuFiscal API",
    description="Backend RAG pour l'analyse du Code des Impôts via WhatsApp",
    version="1.0.0"
)

@app.get("/health")
def health_check():
    return {"status": "healthy", "moteur": "Llama 3 (70B) via Groq"}

@app.post("/webhook/whatsapp")
async def whatsapp_webhook(Body: str = Form(""), From: str = Form("")):
    # 1. Extraction directe et propre des données du formulaire Twilio grâce au typage Form
    message_utilisateur = Body.strip()
    numero_expediteur = From
    
    print(f"📩 Message reçu de {numero_expediteur} : '{message_utilisateur}'")
    
    # 2. Génération de la réponse via le moteur RAG
    if message_utilisateur:
        try:
            # nest_asyncio permet à cet appel de s'exécuter sans bloquer ou crasher
            reponse_fiscale = moteur_rag.generer_reponse(message_utilisateur)
        except Exception as e:
            reponse_fiscale = "⚠️ Une erreur technique est survenue lors de l'analyse fiscale."
            print(f"❌ Erreur lors du traitement de la requête RAG : {str(e)}")
    else:
        reponse_fiscale = "Désolé, je n'ai pas pu lire le contenu de votre message."

    # 3. Formatage de la réponse pour WhatsApp (Gras avec *)
    entete = "🤖 *ConsulFiscal Pro*\n\n"
    message_final = f"{entete}{reponse_fiscale}"

    # 4. Préparation du TwiML XML strict avec la balise <Body>
    twiml_response = (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<Response>'
        f'<Message><Body>{message_final}</Body></Message>'
        f'</Response>'
    )
    
    # 5. Renvoi avec le media_type text/xml attendu par Twilio
    return Response(content=twiml_response, media_type="text/xml")