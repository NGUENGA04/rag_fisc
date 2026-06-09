# 1. Utiliser une image Python officielle et légère
FROM python:3.10-slim

# 2. Définir le dossier de travail dans le serveur
WORKDIR /code

# 3. Copier le fichier des dépendances et les installer
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 4. Copier tout le reste du code du projet dans le serveur
COPY . .

# 5. Lancer l'API FastAPI sur le port 7860 (le port imposé par Hugging Face)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]