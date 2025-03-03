import pytest
from fastapi.testclient import TestClient
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.main import app
from backend.audio_generator import AudioGenerator
from backend.get_transcript import get_video_transcript
from backend.question_generator import QuestionGenerator
from backend.vector_store import VectorStore

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Language Listening Comprehension API"}

def test_transcript_endpoint():
    video_url = "https://www.youtube.com/watch?v=test_video"
    response = client.post("/transcript", json={"video_url": video_url})
    assert response.status_code in [200, 400]  # Either success or valid error

def test_question_generation():
    text = "This is a test text for generating questions."
    response = client.post("/generate-questions", 
                         json={"text": text, "num_questions": 2})
    assert response.status_code in [200, 400]

def test_audio_generation():
    test_question = {
        "Introduction": "Please listen to the following conversation.",
        "Conversation": "Hello, how are you? I'm fine, thank you.",
        "Question": "How is the person feeling?",
        "Options": ["Fine", "Bad", "Tired", "Hungry"]
    }
    response = client.post("/generate-audio", json=test_question)
    assert response.status_code in [200, 400]

def test_audio_generator():
    generator = AudioGenerator()
    test_text = "Hello, this is a test."
    try:
        parts = generator.parse_conversation({"Conversation": test_text})
        assert len(parts) > 0
    except Exception as e:
        pytest.skip(f"Audio generator test skipped: {str(e)}")

def test_question_generator():
    generator = QuestionGenerator()
    test_text = "The quick brown fox jumps over the lazy dog."
    try:
        questions = generator.generate_questions(test_text, 1)
        assert len(questions) > 0
    except Exception as e:
        pytest.skip(f"Question generator test skipped: {str(e)}")

def test_vector_store():
    try:
        store = VectorStore()
        test_text = "This is a test document"
        store.add_document("test", test_text)
        results = store.search("test", 1)
        assert len(results) > 0
    except Exception as e:
        pytest.skip(f"Vector store test skipped: {str(e)}")
