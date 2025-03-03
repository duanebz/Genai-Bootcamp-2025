import sqlite3
import os
from app import create_app

def run_migrations():
    # Create Flask app to use its database configuration
    app = create_app()
    
    try:
        # Initialize the database using the app's db instance
        app.db.init(app)
        print("Database initialized successfully")
        
    except Exception as e:
        print(f"Error initializing database: {str(e)}")

if __name__ == '__main__':
    run_migrations()
