import streamlit as st
import sys
import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

# Add the backend directory to Python path
backend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend')
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from question_generator import QuestionGenerator
from audio_generator import AudioGenerator, TTSServiceError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="JLPT Listening Practice",
    page_icon="🎧",
    layout="wide"
)

class DataManager:
    def __init__(self):
        self.base_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.questions_file = self.base_dir / "backend/data/stored_questions.json"
        self.questions_file.parent.mkdir(parents=True, exist_ok=True)

    def load_stored_questions(self) -> Dict:
        """Load previously stored questions from JSON file"""
        try:
            if self.questions_file.exists():
                with open(self.questions_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading questions: {str(e)}")
            return {}

    def save_question(self, question: Dict, practice_type: str, topic: str, audio_file: Optional[str] = None) -> Optional[str]:
        """Save a generated question to JSON file"""
        try:
            # Load existing questions
            stored_questions = self.load_stored_questions()
            
            # Create a unique ID for the question
            question_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Add metadata
            question_data = {
                "question": question,
                "practice_type": practice_type,
                "topic": topic,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "audio_file": audio_file
            }
            
            # Add to stored questions
            stored_questions[question_id] = question_data
            
            # Save back to file
            with open(self.questions_file, 'w', encoding='utf-8') as f:
                json.dump(stored_questions, f, ensure_ascii=False, indent=2)
            
            return question_id
        except Exception as e:
            logger.error(f"Error saving question: {str(e)}")
            return None

def render_interactive_stage():
    """Render the interactive learning stage"""
    try:
        # Initialize session state
        if 'data_manager' not in st.session_state:
            st.session_state.data_manager = DataManager()
        if 'question_generator' not in st.session_state:
            st.session_state.question_generator = QuestionGenerator()
        if 'audio_generator' not in st.session_state:
            try:
                st.session_state.audio_generator = AudioGenerator()
            except TTSServiceError as e:
                st.error("Audio generation is currently unavailable. Please check your configuration.")
                logger.error(f"TTS service error: {str(e)}")
        if 'current_question' not in st.session_state:
            st.session_state.current_question = None
        if 'feedback' not in st.session_state:
            st.session_state.feedback = None
        if 'current_practice_type' not in st.session_state:
            st.session_state.current_practice_type = None
        if 'current_topic' not in st.session_state:
            st.session_state.current_topic = None
        if 'current_audio' not in st.session_state:
            st.session_state.current_audio = None
        if 'error_message' not in st.session_state:
            st.session_state.error_message = None
            
        # Load stored questions for sidebar
        stored_questions = st.session_state.data_manager.load_stored_questions()
        
        # Create sidebar
        with st.sidebar:
            st.header("Saved Questions")
            if stored_questions:
                for qid, qdata in stored_questions.items():
                    # Create a button for each question
                    button_label = f"{qdata['practice_type']} - {qdata['topic']}\n{qdata['created_at']}"
                    if st.button(button_label, key=qid):
                        st.session_state.current_question = qdata['question']
                        st.session_state.current_practice_type = qdata['practice_type']
                        st.session_state.current_topic = qdata['topic']
                        st.session_state.current_audio = qdata.get('audio_file')
                        st.session_state.feedback = None
                        st.session_state.error_message = None
                        st.rerun()
            else:
                st.info("No saved questions yet. Generate some questions to see them here!")
        
        # Show any error message
        if st.session_state.error_message:
            st.error(st.session_state.error_message)
            if st.button("Clear Error"):
                st.session_state.error_message = None
                st.rerun()
        
        # Practice type selection
        practice_type = st.selectbox(
            "Select Practice Type",
            ["Dialogue Practice", "Phrase Matching"]
        )
        
        # Topic selection
        topics = {
            "Dialogue Practice": ["Daily Conversation", "Shopping", "Restaurant", "Travel", "School/Work"],
            "Phrase Matching": ["Announcements", "Instructions", "Weather Reports", "News Updates"]
        }
        
        topic = st.selectbox(
            "Select Topic",
            topics[practice_type]
        )
        
        # Generate new question button
        if st.button("Generate New Question"):
            try:
                with st.spinner("Generating question..."):
                    section_num = 2 if practice_type == "Dialogue Practice" else 3
                    new_question = st.session_state.question_generator.generate_similar_question(
                        section_num, topic
                    )
                    
                    if new_question:
                        st.session_state.current_question = new_question
                        st.session_state.current_practice_type = practice_type
                        st.session_state.current_topic = topic
                        st.session_state.feedback = None
                        st.session_state.error_message = None
                        
                        # Save the generated question
                        question_id = st.session_state.data_manager.save_question(
                            new_question, practice_type, topic
                        )
                        if not question_id:
                            st.warning("Question generated but couldn't be saved")
                            
                        st.session_state.current_audio = None
                        st.rerun()
                    else:
                        st.session_state.error_message = "Failed to generate question. Please try again."
                        st.rerun()
            except Exception as e:
                logger.error(f"Error generating question: {str(e)}")
                st.session_state.error_message = "An error occurred while generating the question"
                st.rerun()
        
        if st.session_state.current_question:
            st.subheader("Practice Scenario")
            
            # Display question components
            if practice_type == "Dialogue Practice":
                st.write("**Introduction:**")
                st.write(st.session_state.current_question['Introduction'])
                st.write("**Conversation:**")
                st.write(st.session_state.current_question['Conversation'])
            else:
                st.write("**Situation:**")
                st.write(st.session_state.current_question['Situation'])
            
            st.write("**Question:**")
            st.write(st.session_state.current_question['Question'])
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Display options with proper error handling
                try:
                    options = st.session_state.current_question['Options']
                    
                    if st.session_state.feedback:
                        correct = st.session_state.feedback.get('correct', False)
                        correct_answer = st.session_state.feedback.get('correct_answer', 1) - 1
                        selected_index = st.session_state.selected_answer - 1 if hasattr(st.session_state, 'selected_answer') else -1
                        
                        st.write("\n**Your Answer:**")
                        for i, option in enumerate(options):
                            if i == correct_answer and i == selected_index:
                                st.success(f"{i+1}. {option} ✓ (Correct!)")
                            elif i == correct_answer:
                                st.success(f"{i+1}. {option} ✓ (This was the correct answer)")
                            elif i == selected_index:
                                st.error(f"{i+1}. {option} ✗ (Your answer)")
                            else:
                                st.write(f"{i+1}. {option}")
                        
                        # Show explanation
                        st.write("\n**Explanation:**")
                        explanation = st.session_state.feedback.get('explanation', 'No feedback available')
                        if correct:
                            st.success(explanation)
                        else:
                            st.error(explanation)
                        
                        # Add button to try new question
                        if st.button("Try Another Question"):
                            st.session_state.feedback = None
                            st.rerun()
                    else:
                        # Display options as radio buttons when no feedback yet
                        selected = st.radio(
                            "Choose your answer:",
                            options,
                            index=None,
                            format_func=lambda x: f"{options.index(x) + 1}. {x}"
                        )
                        
                        # Submit answer button
                        if selected and st.button("Submit Answer"):
                            selected_index = options.index(selected) + 1
                            st.session_state.selected_answer = selected_index
                            st.session_state.feedback = st.session_state.question_generator.get_feedback(
                                st.session_state.current_question,
                                selected_index
                            )
                            st.rerun()
                except Exception as e:
                    logger.error(f"Error displaying options: {str(e)}")
                    st.error("Error displaying question options")
            
            with col2:
                # Audio controls with proper error handling
                try:
                    if not st.session_state.current_audio and hasattr(st.session_state, 'audio_generator'):
                        if st.button("Generate Audio"):
                            with st.spinner("Generating audio..."):
                                audio_file = st.session_state.audio_generator.generate_audio(
                                    st.session_state.current_question
                                )
                                if audio_file:
                                    st.session_state.current_audio = audio_file
                                    # Update stored question with audio file
                                    st.session_state.data_manager.save_question(
                                        st.session_state.current_question,
                                        st.session_state.current_practice_type,
                                        st.session_state.current_topic,
                                        audio_file
                                    )
                                    st.rerun()
                                else:
                                    st.error("Failed to generate audio. Please try again.")
                    
                    if st.session_state.current_audio:
                        st.audio(st.session_state.current_audio)
                        
                        # Add refresh button for audio
                        if st.button("Regenerate Audio"):
                            st.session_state.current_audio = None
                            st.rerun()
                except Exception as e:
                    logger.error(f"Error with audio controls: {str(e)}")
                    st.error("Error with audio playback")
                
    except Exception as e:
        logger.error(f"Error in interactive stage: {str(e)}")
        st.error("An unexpected error occurred. Please refresh the page and try again.")

def main():
    render_interactive_stage()

if __name__ == "__main__":
    main()
