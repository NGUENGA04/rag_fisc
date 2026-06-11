import os
from fastapi import FastAPI, Form, Response
from dotenv import load_dotenv

# Importation du moteur RAG configuré
from moteur_rag import moteur_rag

load_dotenv()

app = FastAPI(
    title="ConsuFiscal API",
    description="Backend RAG pour l'analyse du Code des Impôts via WhatsApp",
    version="1.0.0"
)

@app.get("/health")
async def health_check(q: str = "Bonjour"):
    try:
        # On teste le moteur RAG directement en asynchrone
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
async def whatsapp_webhook(Body: str = Form(""), From: str = Form("")):
    message_utilisateur = Body.strip()
    numero_expediteur = From
    
    print(f"📩 Message reçu de {numero_expediteur} : '{message_utilisateur}'")
    
    if message_utilisateur:
        try:
            # 🚀 APPEL ASYNCHRONE : Plus besoin de nest_asyncio !
            reponse_fiscale = await moteur_rag.generer_reponse_async(message_utilisateur)
        except Exception as e:
            reponse_fiscale = "⚠️ Une erreur technique est survenue lors de l'analyse fiscale."
            print(f"❌ Erreur lors du traitement de la requête RAG : {str(e)}")
    else:
        reponse_fiscale = "Désolé, je n'ai pas pu lire le contenu de votre message."

    entete = "🤖 *ConsulFiscal Pro*\n\n"
    message_final = f"{entete}{reponse_fiscale}"

    twiml_response = (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<Response>'
        f'<Message><Body>{message_final}</Body></Message>'
        f'</Response>'
    )
    
    return Response(content=twiml_response, media_type="text/xml")