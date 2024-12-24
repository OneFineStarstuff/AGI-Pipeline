# === Imports ===
import os
import asyncio
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from PIL import Image
from fastapi import FastAPI, UploadFile, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
import whisper
from ultralytics import YOLO
import pyttsx3
from loguru import logger
import io
import nest_asyncio
import uvicorn

# === Logging Setup ===
logger.add("pipeline_{time}.log", rotation="1 MB", level="DEBUG", enqueue=True, backtrace=True, diagnose=True)

# === Environment Variables and Authentication ===
SECURE_TOKEN = os.getenv("SECURE_TOKEN", "your_actual_secure_token")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def authenticate_user(token: str = Depends(oauth2_scheme)):
    if token != SECURE_TOKEN:
        logger.warning("Authentication failed.")
        raise HTTPException(status_code=401, detail="Invalid token")

# === Request and Response Models ===
class TextRequest(BaseModel):
    text: str

class TextResponse(BaseModel):
    response: str

# === NLP Module ===
class NLPModule:
    def __init__(self):
        model_name = "google/flan-t5-small"
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            logger.info("NLP model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load NLP model: {e}")
            raise RuntimeError("Failed to load NLP model.")

    def generate_text(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")
        logger.debug(f"Generating text for prompt: {prompt}")
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(inputs["input_ids"], max_length=100)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Generated response: {response}")
            return response
        except Exception as e:
            logger.error(f"Error in text generation: {e}")
            raise RuntimeError("Text generation failed.")

# === CV Module with Object Detection ===
class CVModule:
    def __init__(self):
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = YOLO('yolov5su.pt').to(self.device)
            logger.info("CV model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load CV model: {e}")
            raise RuntimeError("Failed to load CV model.")

    def detect_objects(self, image: Image.Image) -> str:
        logger.debug("Detecting objects in the image.")
        try:
            results = self.model(image)
            return results.pandas().xyxy[0].to_json()
        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            raise ValueError("Object detection error.")

# === Speech Processor ===
class SpeechProcessor:
    def __init__(self):
        try:
            import whisper  # Import inside the class to ensure correct package
            self.whisper_model = whisper.load_model("base")
            self.tts = pyttsx3.init()
            logger.info("Speech processor initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize speech processor: {e}")
            raise RuntimeError("Failed to initialize speech processor.")

    def speech_to_text(self, audio_file: UploadFile) -> str:
        logger.debug("Processing speech-to-text.")
        try:
            with audio_file.file as audio_data:
                result = self.whisper_model.transcribe(audio_data)
            return result['text']
        except Exception as e:
            logger.error(f"Speech-to-text failed: {e}")
            raise ValueError("Speech-to-text error.")

    def text_to_speech(self, text: str) -> None:
        if not text.strip():
            raise ValueError("Text cannot be empty.")
        logger.debug("Processing text-to-speech.")
        try:
            self.tts.say(text)
            self.tts.runAndWait()
        except Exception as e:
            logger.error(f"Text-to-speech failed: {e}")
            raise RuntimeError("Text-to-speech error.")

    def __del__(self):
        self.tts.stop()

# === Enhanced AGI Pipeline ===
class EnhancedAGIPipeline:
    def __init__(self):
        self.nlp = NLPModule()
        self.cv = CVModule()
        self.speech_processor = SpeechProcessor()

    async def process_nlp(self, text: str) -> str:
        return self.nlp.generate_text(text)

    async def process_cv(self, image: Image.Image) -> str:
        return await asyncio.to_thread(self.cv.detect_objects, image)

    async def process_speech_to_text(self, audio_file: UploadFile) -> str:
        return await asyncio.to_thread(self.speech_processor.speech_to_text, audio_file)

    async def process_text_to_speech(self, text: str) -> None:
        await asyncio.to_thread(self.speech_processor.text_to_speech, text)

# === FastAPI Application ===
app = FastAPI()
pipeline = EnhancedAGIPipeline()

@app.post("/process-nlp/", response_model=TextResponse, dependencies=[Depends(authenticate_user)])
async def process_nlp(request: TextRequest):
    try:
        response = await pipeline.process_nlp(request.text)
        logger.info("NLP processed successfully.")
        return {"response": response}
    except Exception as e:
        logger.error(f"NLP processing failed: {e}")
        raise HTTPException(status_code=500, detail="NLP processing error.")

@app.post("/process-cv-detection/", dependencies=[Depends(authenticate_user)])
async def process_cv_detection(file: UploadFile):
    try:
        image = Image.open(io.BytesIO(await file.read()))
        response = await pipeline.process_cv(image)
        logger.info("Object detection processed successfully.")
        return {"detections": response}
    except Exception as e:
        logger.error(f"Object detection failed: {e}")
        raise HTTPException(status_code=500, detail="Object detection error.")

@app.post("/speech-to-text/", response_model=TextResponse, dependencies=[Depends(authenticate_user)])
async def speech_to_text(file: UploadFile):
    try:
        response = await pipeline.process_speech_to_text(file)
        logger.info("Speech-to-text processed successfully.")
        return {"response": response}
    except Exception as e:
        logger.error(f"Speech-to-text failed: {e}")
        raise HTTPException(status_code=500, detail="Speech-to-text error.")

@app.post("/text-to-speech/", dependencies=[Depends(authenticate_user)])
async def text_to_speech(request: TextRequest):
    try:
        await pipeline.process_text_to_speech(request.text)
        logger.info("Text-to-speech processed successfully.")
        return {"response": "Speech synthesis complete."}
    except Exception as e:
        logger.error(f"Text-to-speech failed: {e}")
        raise HTTPException(status_code=500, detail="Text-to-speech error.")

# === Run the Application with HTTPS ===
if __name__ == "__main__":
    nest_asyncio.apply()
    config = uvicorn.Config(app, host="0.0.0.0", port=8000)
    server = uvicorn.Server(config)
    asyncio.run(server.serve())
