from typing import Dict, List, Optional
import json
import os

class QuestionVectorStore:
    def __init__(self, persist_directory: str = "backend/data/vectorstore"):
        """Initialize a simple store for JLPT listening questions"""
        self.persist_directory = persist_directory
        self.questions = {
            "section2": [],
            "section3": []
        }
        
    def add_questions(self, section_num: int, questions: List[Dict], video_id: str):
        """Add questions to the store"""
        if section_num not in [2, 3]:
            raise ValueError("Only sections 2 and 3 are currently supported")
        
        section_key = f"section{section_num}"
        for idx, question in enumerate(questions):
            question_id = f"{video_id}_{section_num}_{idx}"
            self.questions[section_key].append({
                "id": question_id,
                "question": question,
                "video_id": video_id
            })

    def search_similar_questions(self, section_num: int, query: str, n_results: int = 5) -> List[Dict]:
        """Return some sample questions (no actual similarity search for now)"""
        if section_num not in [2, 3]:
            raise ValueError("Only sections 2 and 3 are currently supported")
        
        section_key = f"section{section_num}"
        questions = self.questions[section_key]
        return [q["question"] for q in questions[:n_results]]

    def get_question_by_id(self, section_num: int, question_id: str) -> Optional[Dict]:
        """Get a specific question by ID"""
        if section_num not in [2, 3]:
            raise ValueError("Only sections 2 and 3 are currently supported")
        
        section_key = f"section{section_num}"
        for q in self.questions[section_key]:
            if q["id"] == question_id:
                return q["question"]
        return None

    def parse_questions_from_file(self, filename: str) -> List[Dict]:
        """Parse questions from a structured text file"""
        questions = []
        current_question = {}
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                if line.startswith('<question>'):
                    current_question = {}
                elif line.startswith('Introduction:'):
                    i += 1
                    if i < len(lines):
                        current_question['Introduction'] = lines[i].strip()
                elif line.startswith('Conversation:'):
                    i += 1
                    if i < len(lines):
                        current_question['Conversation'] = lines[i].strip()
                elif line.startswith('Situation:'):
                    i += 1
                    if i < len(lines):
                        current_question['Situation'] = lines[i].strip()
                elif line.startswith('Question:'):
                    i += 1
                    if i < len(lines):
                        current_question['Question'] = lines[i].strip()
                elif line.startswith('Options:'):
                    options = []
                    for _ in range(4):
                        i += 1
                        if i < len(lines):
                            option = lines[i].strip()
                            if option.startswith('1.') or option.startswith('2.') or option.startswith('3.') or option.startswith('4.'):
                                options.append(option[2:].strip())
                    current_question['Options'] = options
                elif line.startswith('</question>'):
                    if current_question:
                        questions.append(current_question)
                        current_question = {}
                i += 1
            return questions
        except Exception as e:
            print(f"Error parsing questions from {filename}: {str(e)}")
            return []

    def index_questions_file(self, filename: str, section_num: int):
        """Index all questions from a file into the store"""
        video_id = os.path.basename(filename).split('_section')[0]
        questions = self.parse_questions_from_file(filename)
        self.add_questions(section_num, questions, video_id)

if __name__ == "__main__":
    # Example usage
    store = QuestionVectorStore()
    
    # Index questions from files
    question_files = [
        ("backend/data/questions/sY7L5cfCWno_section2.txt", 2),
        ("backend/data/questions/sY7L5cfCWno_section3.txt", 3)
    ]
    
    for filename, section_num in question_files:
        if os.path.exists(filename):
            store.index_questions_file(filename, section_num)
    
    # Search for similar questions
    similar = store.search_similar_questions(2, "誕生日について質問", n_results=1)
