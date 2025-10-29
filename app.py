import os
import pandas as pd
import numpy as np
import json
import ast
import faiss
from sentence_transformers import SentenceTransformer
from googletrans import Translator, LANGUAGES
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# --- 1. Initialize App ---
app = Flask(__name__)
CORS(app)  # Enable CORS

# --- 2. Load Models and Data (at startup) ---
# This code runs only ONCE when the server starts.
print("Loading models and data... This may take a moment.")
try:
    model = SentenceTransformer('all-mpnet-base-v2')
    translator = Translator()
    index = faiss.read_index("index.faiss")
    df = pd.read_csv("prompts.csv")
    print("Models and data loaded successfully.")
except Exception as e:
    print(f"CRITICAL Error loading models or data: {e}")
    # In a real app, you might want to exit if models fail to load
    # For now, we'll print the error and continue, but /retrieve will fail
    model = None
    translator = None
    index = None
    df = None

# --- 3. Backend Logic (Your retrieve function) ---
def retrieve_top_n(user_query, top_n, model, index, data):
    # Check if models loaded correctly
    if not all([model, index, data is not None, translator]):
        print("Error: Models or data not loaded. Cannot process query.")
        return []
        
    if not user_query:
        return []

    # Detect and translate language
    translated_query = user_query
    try:
        detected_language = translator.detect(user_query).lang
        if detected_language != 'en':
            translation = translator.translate(user_query, src=detected_language, dest='en')
            translated_query = translation.text
            print(f"Translated Query: {translated_query} (from {LANGUAGES.get(detected_language, detected_language)})")
    except Exception as e:
        print(f"Translation error: {e}. Using original query: {user_query}")

    # FAISS search
    try:
        query_embedding = model.encode(translated_query).reshape(1, -1).astype('float32')
        distances, indices = index.search(query_embedding, top_n)
    except Exception as e:
        print(f"Error during FAISS search: {e}")
        return []

    results = []
    for i, idx in enumerate(indices[0]):
        try:
            # Use .iloc[idx] to get the row by its integer position
            result_row = data.iloc[idx]
            prompt_content = result_row['prompt']

            # Safely parse the 'prompt' string
            prompt_dict = None
            if isinstance(prompt_content, str):
                try:
                    # ast.literal_eval is safer than eval
                    prompt_dict = ast.literal_eval(prompt_content)
                except (ValueError, SyntaxError):
                    # Fallback to json.loads if literal_eval fails
                    try:
                        prompt_dict = json.loads(prompt_content)
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse prompt string: {prompt_content}")
                        continue # Skip this result if unparseable
            
            if not isinstance(prompt_dict, dict):
                print(f"Warning: Parsed content is not a dictionary. Skipping.")
                continue

            distance = float(distances[0][i]) # Ensure distance is a standard float
            result_metadata = {
                'Title': prompt_dict.get('Title', 'No Title'),
                'Author': prompt_dict.get('Author', 'No Author'),
                "Labels": prompt_dict.get('Labels', 'No Labels'),
                "Read Level": prompt_dict.get('Read Level', 'No Read level'),
                'Hyperlink': prompt_dict.get('Hyperlink', 'No Hyperlink'),
                'Match_Percentage': f"{distance:.2f}"
            }
            results.append(result_metadata)
        except Exception as e:
            print(f"Error processing result at index {idx}: {e}")
            
    return results

# --- 4. API Endpoint (Backend) ---
@app.route('/retrieve', methods=['POST'])
def retrieve():
    data = request.get_json()
    query = data.get('query', '')
    print(f"Received query: {query}")
    
    # Pass the globally loaded models to the function
    results = retrieve_top_n(query, 5, model, index, df)
    
    return jsonify(results)

# --- 5. Page Endpoint (Frontend) ---
@app.route('/')
def home():
    # This will find and serve your 'index.html' from the 'templates' folder.
    return render_template('index.html')

# --- 6. Run the App ---
if __name__ == '__main__':
    # Get port from environment variable (for Render) or default to 5000 (for local)
    port = int(os.environ.get('PORT', 5000))
    # Run on 0.0.0.0 to be accessible on the network
    app.run(host='0.0.0.0', port=port)