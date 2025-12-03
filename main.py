from fastapi import FastAPI, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# ======================================
# Import MiniMind
# ======================================
from minimind import MiniMindModel

# Charger une version du modèle MiniMind
# (mets ici le bon nom selon ton installation)
model = MiniMindModel.load_pretrained("gpt-41-mini")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================
# 1️⃣ MODE CHAT — MiniMind répond à un texte
# ==========================================================
@app.post("/api/chat")
async def chat_endpoint(prompt: str = Form(...)):
    """
    MiniMind répond à une question (mode Chat)
    """
    # Appel au modèle MiniMind
    answer = model.generate_text(prompt)

    return {"answer": answer}



# ==========================================================
# 2️⃣ MODE CLASSIFICATION — MiniMind classe un texte
# ==========================================================
@app.post("/api/classify")
async def classify_endpoint(text: str = Form(...)):
    """
    MiniMind analyse un texte et retourne un label
    """
    result = model.classify_text(text)

    return {
        "label": result["label"],
        "confidence": result["confidence"],
        "reason": result["explanation"],
    }



# ==========================================================
# 3️⃣ MODE VISION — MiniMind analyse une IMAGE
# ==========================================================
@app.post("/api/analyze-image")
async def analyze_image_endpoint(file: UploadFile = File(...)):
    """
    MiniMind décrit ce que montre une image
    """
    image_bytes = await file.read()

    # Analyse avec le modèle MiniMind
    description = model.analyze_image(image_bytes)

    return {"description": description}
