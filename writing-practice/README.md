# Japanese Writing Practice App

This Streamlit app helps users practice writing Japanese by providing English sentences to translate.

## Setup Instructions

1. First, set up and start the backend server:
   ```bash
   # In WSL terminal:
   cd /mnt/c/Users/duane/Documents/Coding/GenAIBootcamp/Week1/free-genai-bootcamp-2025-main/free-genai-bootcamp-2025/lang-portal/backend-flask
   pip install -r requirements.txt
   flask run
   ```

2. Then, in a new terminal, install the frontend dependencies and start the app:
   ```bash
   # In WSL terminal:
   cd /mnt/c/Users/duane/Documents/Coding/GenAIBootcamp/Week1/free-genai-bootcamp-2025-main/free-genai-bootcamp-2025/writing-practice
   pip install -r requirements.txt
   streamlit run app.py
   ```

3. Access the app with a group ID:
   ```
   http://localhost:8501/?id=1
   ```

## Features

- Loads vocabulary from specified group ID
- Generates practice sentences using loaded vocabulary
- Allows image upload of handwritten answers
- Provides grading and feedback
- Supports multiple practice sessions

## Troubleshooting

1. If you see "No group ID provided":
   - Make sure to add `?id=1` to the URL
   - The error message will guide you to add the ID parameter

2. If you see "Could not connect to backend":
   - Ensure the Flask backend is running on port 5000
   - Check that there are no CORS issues
   - Verify that the backend URL is correct in the frontend code

3. If you see "No vocabulary found":
   - Verify that the database is initialized
   - Check that the group ID exists in the database
   - Try using a different group ID (e.g., try both 1 and 2)

4. Image Upload Issues:
   - Make sure the image is in JPG, JPEG, or PNG format
   - Check that the file size is reasonable
   - Try a different image if one fails

5. Database Issues:
   - If the database isn't initialized, run:
     ```bash
     cd /mnt/c/Users/duane/Documents/Coding/GenAIBootcamp/Week1/free-genai-bootcamp-2025-main/free-genai-bootcamp-2025/lang-portal/backend-flask
     python migrate.py
     ```

## Development

The app consists of two main components:

1. Backend (Flask):
   - Provides vocabulary data through REST API
   - Manages SQLite database
   - Handles CORS for local development
   - Endpoints:
     - `/api/groups` - List all groups
     - `/api/groups/:id` - Get group details
     - `/api/groups/:id/words` - Get paginated words
     - `/api/groups/:id/raw` - Get all words without pagination

2. Frontend (Streamlit):
   - User interface for practice
   - Image upload and review
   - State management:
     - Setup: Initial state, shows "Generate Sentence" button
     - Practice: Shows English sentence and image upload
     - Review: Shows grading and feedback
   - Error handling for:
     - Missing group ID
     - Backend connection issues
     - Empty vocabulary
     - Image upload errors

## Code Organization

```
writing-practice/
├── app.py           # Main Streamlit application
├── requirements.txt # Frontend dependencies
└── README.md       # This documentation

lang-portal/backend-flask/
├── app.py          # Flask application entry point
├── lib/
│   └── db.py      # Database management
├── routes/
│   └── groups.py  # Group-related endpoints
├── sql/
│   └── setup/     # Database schema
└── seed/          # Sample data
