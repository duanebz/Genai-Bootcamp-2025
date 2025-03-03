import streamlit as st
import requests
import time
from PIL import Image
import io
import random
import json
import base64

# Initialize session state
if 'page_state' not in st.session_state:
    st.session_state.page_state = "setup"
if 'vocabulary' not in st.session_state:
    st.session_state.vocabulary = None
if 'loading' not in st.session_state:
    st.session_state.loading = False
if 'current_word' not in st.session_state:
    st.session_state.current_word = None
if 'current_sentence' not in st.session_state:
    st.session_state.current_sentence = None
if 'review_data' not in st.session_state:
    st.session_state.review_data = None

# Set page configuration
st.set_page_config(
    page_title="Japanese Practice App",
    page_icon="🇯🇵",
    layout="centered"
)

def load_vocabulary():
    # Get group ID from query string
    try:
        group_id = st.query_params.get('id')
        if not group_id:
            raise ValueError("No group ID")
    except:
        # Auto-redirect to group 1
        new_url = "http://localhost:8501/?id=1"
        html = f'<meta http-equiv="refresh" content="0; url={new_url}">'
        st.markdown(html, unsafe_allow_html=True)
        st.info(f"Redirecting to {new_url}")
        return {'error': 'Redirecting...'}
    
    st.session_state.loading = True
    try:
        # Fetch vocabulary from the API endpoint
        api_url = f'http://localhost:5000/api/groups/{group_id}/raw'
        response = requests.get(api_url)
        response.raise_for_status()
        vocabulary = response.json()
        
        # Check if we got an empty list
        if not vocabulary:
            return {'error': f'No vocabulary found for group {group_id}'}
        
        # Store vocabulary in session state
        st.session_state.vocabulary = vocabulary
        return {'message': f'Successfully loaded {len(vocabulary)} words'}
    except requests.exceptions.ConnectionError:
        return {'error': 'Could not connect to the backend server. Make sure it is running on localhost:5000'}
    except requests.exceptions.RequestException as e:
        return {'error': f'Failed to fetch vocabulary: {str(e)}'}
    finally:
        st.session_state.loading = False

# Function to generate a sentence using the LLM
def generate_sentence(word):
    # In a real implementation, this would call your LLM API
    # For now, we'll simulate the response
    prompt = f"""
    Generate a sentcne using the following {word}
    The grammar should be JLPIN5
    You ca use the following vocabulary to construct a simple sentecne:
    - simple objects eg. book, car, rame, sushi
    - simple verbs eg. to drink, toeat, to meet
    - simple times eg. tomorrow, today, yesterday
    
    Return the English sentence only.
    """
    
    # Simulate different sentences based on the word
    english_sentences = {
        "飲む": "I will drink water tomorrow.",
        "食べる": "She ate sushi yesterday.",
        "本": "This book is interesting.",
        "車": "My car is blue.",
        "明日": "I will go to school tomorrow."
    }
    
    # Return a default sentence if the word is not in our examples
    return english_sentences.get(word, f"Please use the word '{word}' in a simple sentence.")

# Function to grade the submission
def grade_submission(image, english_sentence):
    # In a real implementation, this would call your grading API
    # For now, we'll simulate the response
    
    # Simulate processing delay
    time.sleep(1)
    
    # Convert PIL Image to bytes for storage
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # Simulate different grades based on the English sentence
    grades = {
        "I will drink water tomorrow.": {
            "transcription": "明日水を飲みます。",  # MangoOCR transcription
            "translation": "I will drink water tomorrow.",  # LLM translation of image
            "transcription_translation": "I will drink water tomorrow.",  # LLM translation of transcription
            "grade": "A",  # 5 Rank score
            "feedback": "Excellent! Your sentence structure is correct and the translation matches perfectly. Your handwriting is clear and legible.",
            "image": img_byte_arr
        },
        "She ate sushi yesterday.": {
            "transcription": "彼女は昨日寿司を食べました。",
            "translation": "She ate sushi yesterday.",
            "transcription_translation": "She ate sushi yesterday.",
            "grade": "A",
            "feedback": "Great job! Your sentence is grammatically correct and conveys the meaning accurately. Your particle usage is perfect.",
            "image": img_byte_arr
        }
    }
    
    # Return a default grade if the sentence is not in our examples
    return grades.get(english_sentence, {
        "transcription": "手書きの文章",  # Placeholder for MangoOCR
        "translation": "Handwritten sentence",  # LLM translation of image
        "transcription_translation": "Handwritten sentence",  # LLM translation of transcription
        "grade": "B",  # 5 Rank score
        "feedback": "Good attempt! Make sure to check your verb tense and particle usage. Try to write more clearly for better OCR recognition.",
        "image": img_byte_arr
    })

# Function to handle state transitions
def move_to_practice_state():
    if st.session_state.vocabulary:
        # Select a random word from the vocabulary
        random_word = random.choice(st.session_state.vocabulary)
        japanese_word = random_word.get("kanji", "飲む")  # Use kanji key instead of japanese
        
        # Generate a sentence
        st.session_state.current_sentence = generate_sentence(japanese_word)
        st.session_state.page_state = "practice"
    else:
        st.error("No vocabulary loaded. Please reload the page.")

def move_to_review_state():
    st.session_state.page_state = "review"

def move_to_next_question():
    move_to_practice_state()

# Main application logic based on page state
def main():
    st.title("Japanese Practice App")
    
    # Load vocabulary from query string group ID
    load_vocabulary_result = load_vocabulary()
    
    # Show loading spinner while fetching vocabulary
    if st.session_state.loading:
        with st.spinner('Loading vocabulary...'):
            st.empty()
    
    if 'error' in load_vocabulary_result:
        st.error(load_vocabulary_result['error'])
    elif 'message' in load_vocabulary_result:
        st.success(load_vocabulary_result['message'])
    
    # Display content based on the current page state
    if st.session_state.page_state == "setup":
        st.write("Welcome to the Japanese Practice App!")
        st.write("Click the button below to generate a sentence for practice.")
        
        if st.button("Generate Sentence"):
            if st.session_state.vocabulary:  # Only move to practice if we have vocabulary
                move_to_practice_state()
            else:
                st.error("No vocabulary loaded. Please add ?id=1 to the URL and reload the page.")
                st.code("http://localhost:8501/?id=1", language="text")
    
    elif st.session_state.page_state == "practice":
        st.subheader("Translate this sentence to Japanese:")
        st.write(st.session_state.current_sentence)
        
        uploaded_file = st.file_uploader("Upload your handwritten answer (image):", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Your uploaded answer", use_container_width=True)
            
            if st.button("Submit for Review"):
                st.session_state.review_data = grade_submission(image, st.session_state.current_sentence)
                move_to_review_state()
        else:
            if st.button("Submit for Review"):
                st.error("Please upload an image of your answer before submitting.")
    
    elif st.session_state.page_state == "review":
        st.subheader("Review your submission:")
        st.write(f"English sentence: {st.session_state.current_sentence}")
        
        if st.session_state.review_data:
            # Display the uploaded image
            st.write("Your submission:")
            img = Image.open(io.BytesIO(st.session_state.review_data["image"]))
            st.image(img, use_container_width=True)
            
            # Display grading results
            st.write("Grading results:")
            st.write(f"Transcription: {st.session_state.review_data['transcription']}")
            st.write(f"Translation: {st.session_state.review_data['translation']}")
            st.write(f"Transcription Translation: {st.session_state.review_data['transcription_translation']}")
            st.write(f"Grade: {st.session_state.review_data['grade']}")
            st.write(f"Feedback: {st.session_state.review_data['feedback']}")
        
        if st.button("Next Question"):
            move_to_practice_state()

if __name__ == "__main__":
    main()