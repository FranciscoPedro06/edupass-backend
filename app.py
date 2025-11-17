from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from deepface import DeepFace
import numpy as np
import uuid
import os
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = FastAPI(title="EduPass - Reconhecimento Facial")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

facial_db = {}

@app.get("/")
async def health():
    return {
        "status": "online",
        "usuarios_cadastrados": len(facial_db)
    }


@app.post("/cadastrar")
async def cadastrar(
    nome: str = Form(...),
    file: UploadFile = File(...)
):
    if not file.content_type.startswith("image/"):
        return {"success": False, "error": "Envie uma imagem válida."}

    temp_path = f"temp_{uuid.uuid4()}.jpg"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        # Modelo MAIS LEVE para Render
        embedding_obj = DeepFace.represent(
            img_path=temp_path,
            model_name="Facenet512",
            detector_backend="ssd",
            enforce_detection=True
        )[0]

        embedding = embedding_obj["embedding"]

        user_id = str(uuid.uuid4())

        facial_db[user_id] = {
            "id": user_id,
            "nome": nome,
            "embedding": embedding,
            "embedding_size": len(embedding),
            "data_cadastro": datetime.now().isoformat()
        }

        os.remove(temp_path)

        return {
            "success": True,
            "mensagem": f"{nome} cadastrado com sucesso!",
            "id": user_id
        }

    except Exception as e:
        os.remove(temp_path)
        return {"success": False, "error": str(e)}



@app.post("/reconhecer")
async def reconhecer(file: UploadFile = File(...)):

    if not facial_db:
        return {
            "success": False,
            "error": "Nenhum usuário cadastrado ainda."
        }

    temp_path = f"temp_{uuid.uuid4()}.jpg"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        embedding_obj = DeepFace.represent(
            img_path=temp_path,
            model_name="Facenet512",
            detector_backend="ssd",
            enforce_detection=True
        )[0]

        embedding_atual = np.array(embedding_obj["embedding"])

    except Exception as e:
        os.remove(temp_path)
        return {"success": False, "error": "Não foi detectado rosto."}

    menor_dist = float("inf")
    usuario_final = None

    for _, user in facial_db.items():
        emb = np.array(user["embedding"])

        dot = np.dot(embedding_atual, emb)
        normA = np.linalg.norm(embedding_atual)
        normB = np.linalg.norm(emb)
        dist = 1 - (dot / (normA * normB))

        if dist < menor_dist:
            menor_dist = dist
            usuario_final = user

    os.remove(temp_path)

    THRESHOLD = 0.35 

    if menor_dist < THRESHOLD:
        confianca = (1 - menor_dist) * 100
        return {
            "success": True,
            "mensagem": "Reconhecido",
            "usuario": usuario_final,
            "distancia": round(menor_dist, 4),
            "confianca": round(confianca, 2)
        }

    return {
        "success": False,
        "mensagem": "Não reconhecido",
        "distancia": round(menor_dist, 4)
    }


@app.get("/usuarios")
async def usuarios():
    lista = [
        {
            "id": u["id"],
            "nome": u["nome"],
            "data_cadastro": u["data_cadastro"]
        }
        for u in facial_db.values()
    ]

    return {"usuarios": lista}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )
