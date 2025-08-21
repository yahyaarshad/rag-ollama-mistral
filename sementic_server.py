from flask import Flask, request, jsonify
from DocumentIndexer import DocumentIndexer
import os
import requests

app = Flask(__name__)

indexer = DocumentIndexer()

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'txt'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# -------------------------------
# 5. Generate Answer with Mistral via Ollama
# -------------------------------
def generate_answer(user_context, query):
    # Retrieve documents via FAISS
    retrieved_context = indexer.semantic_search(query)

    # Combine retrieved context + optional user context
    final_prompt = f"""
Use the following context to answer:

Context:
{retrieved_context}
{user_context}

Prompt:
{query}
"""

    # Call Ollama's Mistral model
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral",
            "prompt": final_prompt,
            "stream": False
        }
    )

    if response.status_code != 200:
        print("Failed to call Ollama:", response.text)
        return "Error generating answer."

    return jsonify({"context": query, "response": response.json().get("response")})


@app.route('/', methods=['GET'])
def health_check():
    # Check if the POST request has the file part
    return jsonify({'message': 'Semantic server is up'}), 200


@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the POST request has the file part
    if len(request.files) == 0:
        return jsonify({'error': 'No file part in the request'}), 400

    for key, file_storage in request.files.items():
        if allowed_file(file_storage.filename):
            content = file_storage.read().decode('utf-8')
            indexer.index_new_document(content)
        else:
            print(f"Skipping file {file_storage.filename}: invalid file type")

    return jsonify({'message': 'Indexed successfully'}), 200


@app.route('/query', methods=['POST'])
def query():
    return generate_answer('', request.json['query'])


if __name__ == '__main__':
    app.run(port=5000, debug=True)
