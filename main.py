from fastapi import FastAPI, Form
from fastapi.responses import Response
from moteur_rag import interroger_le_moteur

app = FastAPI(title="ConsulFiscal Backend WhatsApp API")

@app.post("/webhook/whatsapp")
async def webhook_whatsapp(From: str = Form(...), Body: str = Form(...)):
    question_comptable = Body
    print(f"📩 Message reçu de {From} : {question_comptable}")
    
    # 1. Exécution de la recherche sémantique RAG
    resultat = interroger_le_moteur(question_comptable)
    texte_ia = resultat["reponse"]
    dictionnaire_sources = resultat["sources"]
    
    # 2. Construction du message final au format WhatsApp (Gras avec *, italique avec _)
    message_whatsapp = f"🤖 *ConsulFiscal Pro*\n\n{texte_ia}\n\n"
    
    # 3. Concatenation propre des sources en bas du bloc de texte
    if dictionnaire_sources:
        message_whatsapp += "📋 *Sources officielles consultées :*\n"
        for index, (nom_doc, url_doc) in enumerate(dictionnaire_sources.items(), 1):
            message_whatsapp += f"{index}. 📄 _{nom_doc}_ : {url_doc}\n"
            
    # 4. Génération de la réponse XML pour Twilio
    twiml_response = (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<Response>'
        f'<Message><Body>{message_whatsapp}</Body></Message>'
        f'</Response>'
    )
    
    return Response(content=twiml_response, media_type="text/xml")