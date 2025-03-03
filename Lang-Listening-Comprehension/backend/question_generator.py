import json
from typing import Dict, List, Optional
from vector_store import QuestionVectorStore

class QuestionGenerator:
    def __init__(self):
        """Initialize the question generator"""
        self.vector_store = QuestionVectorStore()

    def generate_similar_question(self, section_num: int, topic: str) -> Dict:
        """Generate a sample question based on the topic"""
        if section_num == 2:
            return {
                "Introduction": "You are at a restaurant with your friend.",
                "Conversation": "A: What would you like to order?\nB: I'm thinking about getting pasta.\nA: That's a good choice! The pasta here is delicious.",
                "Question": "What does B want to order?",
                "Options": [
                    "ピザを食べる (Pizza)",
                    "パスタを食べる (Pasta)",
                    "ハンバーガーを食べる (Hamburger)",
                    "サラダを食べる (Salad)"
                ]
            }
        else:  # section 3
            return {
                "Situation": "You hear this announcement at a train station.",
                "Question": "The next train to Tokyo will depart from which platform?",
                "Options": [
                    "Platform 1",
                    "Platform 2",
                    "Platform 3",
                    "Platform 4"
                ]
            }

    def get_feedback(self, question: Dict, selected_answer: int) -> Dict:
        """Generate feedback for the selected answer"""
        # For demonstration, always use option 2 as correct
        correct_answer = 2
        is_correct = selected_answer == correct_answer
        
        return {
            "correct": is_correct,
            "explanation": "The correct answer is option 2. Listen carefully to the conversation/announcement again.",
            "correct_answer": correct_answer
        }
