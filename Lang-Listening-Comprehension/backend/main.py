from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from audio_generator import AudioGenerator
from get_transcript import get_video_transcript
from question_generator import QuestionGenerator
from vector_store import QuestionVectorStore
import uvicorn

app = FastAPI()

class TranscriptRequest(BaseModel):
    video_url: str

class QuestionRequest(BaseModel):
    text: str
    num_questions: int = 5

@app.get("/")
async def root():
    return {"message": "Language Listening Comprehension API"}

@app.post("/transcript")
async def get_transcript(request: TranscriptRequest):
    try:
        transcript = get_video_transcript(request.video_url)
        return {"transcript": transcript}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/generate-questions")
async def generate_questions(request: QuestionRequest):
    try:
        generator = QuestionGenerator()
        questions = generator.generate_questions(request.text, request.num_questions)
        return {"questions": questions}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/generate-audio")
async def generate_audio(question_data: dict):
    try:
        generator = AudioGenerator()
        audio_file = generator.generate_audio(question_data)
        return {"audio_file": audio_file}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
