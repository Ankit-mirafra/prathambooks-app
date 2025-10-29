import os
import pandas as pd
import numpy as np
import json
import ast
import faiss
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# --- 1. Initialize App ---
app = Flask(__name__)
CORS(app)  # Enable CORS

# --- 2. Load Models and Data (at startup) ---
print("Loading models and data... This may take a moment.")
try:
    model = SentenceTransformer('all-mpnet-base-v2')
    translator = GoogleTranslator(source='auto', target='en')
    index = faiss.read_index("index.faiss")
    df = pd.read_csv("prompts.csv")
    print("Models and data loaded successfully.")
except Exception as e:
    print(f"CRITICAL Error loading models or data: {e}")
    model = None
    translator = None
    index = None
    df = None

# --- 3. Backend Logic (Your retrieve function) ---
def retrieve_top_n(user_query, top_n, model, index, data):
    if not all([model, index, data is not None, translator]):
        print("Error: Models or data not loaded. Cannot process query.")
        return []
        
    if not user_query:
        return []

    # Detect and translate language
    translated_query = user_query
    try:
        translated_query = translator.translate(user_query)
        if translated_query and (translated_query.lower() != user_query.lower()):
            print(f"Translated Query: {translated_query}")
    except Exception as e:
        print(f"Translation error: {e}. Using original query: {user_query}")

    # FAISS search
    try:
        # --- IMPROVEMENT ---
        # Encode the query
        query_embedding = model.encode(translated_query).reshape(1, -1).astype('float32')
        # Normalize the query embedding for Cosine Similarity
        faiss.normalize_L2(query_embedding)
        
        # Distances will now be Cosine Similarity scores (higher is better)
        distances, indices = index.search(query_embedding, top_n)
        
    except Exception as e:
        print(f"Error during FAISS search: {e}")
        return []

    results = []
    for i, idx in enumerate(indices[0]):
        try:
            result_row = data.iloc[idx]
            prompt_content = result_row['prompt']
            prompt_dict = None
            
            if isinstance(prompt_content, str):
                try:
                    prompt_dict = ast.literal_eval(prompt_content)
                except (ValueError, SyntaxError):
                    try:
                        prompt_dict = json.loads(prompt_content)
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse prompt string: {prompt_content}")
                        continue
            
            if not isinstance(prompt_dict, dict):
                print(f"Warning: Parsed content is not a dictionary. Skipping.")
                continue

            # --- IMPROVEMENT ---
            # 'distance' is now a Cosine Similarity score (e.g., 0.85)
            cosine_score = float(distances[0][i])
            
            # Convert the 0.0-1.0 score to a 0-100 percentage
            # We also clamp it to ensure it's never < 0 or > 100
            match_percentage = max(0, min(100, cosine_score * 100))
            
            result_metadata = {
                'Title': prompt_dict.get('Title', 'No Title'),
                'Author': prompt_dict.get('Author', 'No Author'),
                "Labels": prompt_dict.get('Labels', 'No Labels'),
                "Read Level": prompt_dict.get('Read Level', 'No Read level'),
                'Hyperlink': prompt_dict.get('Hyperlink', 'No Hyperlink'),
                # We now display the true percentage!
                'Match_Percentage': f"{match_percentage:.2f}%"
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
    results = retrieve_top_n(query, 5, model, index, df)
    return jsonify(results)

# --- 5. Page Endpoint (Frontend) ---
@app.route('/')
def home():
    return render_template('index.html')

# --- 6. Run the App ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port)