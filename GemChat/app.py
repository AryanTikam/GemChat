# app.py
import os
import json
import csv
import zipfile
import tempfile
from io import StringIO
import pdfplumber
import pandas as pd
from flask import Flask, render_template, request, jsonify, session
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "your-secret-key")

# Initialize Gemini model
model = genai.GenerativeModel('gemini-2.0-flash')

def extract_text_from_pdf(file):
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        text = f"Error extracting text from PDF: {str(e)}"
    return text

def extract_from_zip(file):
    result = []
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            
            for root, _, files in os.walk(temp_dir):
                for filename in files:
                    file_path = os.path.join(root, filename)
                    file_ext = os.path.splitext(filename)[1].lower()
                    
                    content = ""
                    if file_ext == '.pdf':
                        with open(file_path, 'rb') as f:
                            content = extract_text_from_pdf(f)
                    elif file_ext == '.csv':
                        with open(file_path, 'r', encoding='utf-8') as f:
                            df = pd.read_csv(f)
                            content = df.to_string()
                    elif file_ext == '.json':
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = json.dumps(json.load(f), indent=2)
                    elif file_ext in ['.txt', '.py', '.js', '.html', '.css']:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                    
                    if content:
                        result.append({
                            "filename": filename,
                            "content": content[:5000] + ("..." if len(content) > 5000 else "")
                        })
    
    return result

def parse_file(file):
    filename = file.filename
    file_ext = os.path.splitext(filename)[1].lower()
    
    try:
        if file_ext == '.pdf':
            return extract_text_from_pdf(file)
        elif file_ext == '.zip':
            return extract_from_zip(file)
        elif file_ext == '.csv':
            df = pd.read_csv(file)
            return df.to_string()
        elif file_ext == '.json':
            content = json.load(file)
            return json.dumps(content, indent=2)
        elif file_ext in ['.txt', '.py', '.js', '.html', '.css']:
            content = file.read().decode('utf-8', errors='ignore')
            return content
        else:
            return f"Unsupported file type: {file_ext}"
    except Exception as e:
        return f"Error processing {filename}: {str(e)}"

def get_gemini_response(prompt, file_content=None):
    try:
        if file_content:
            prompt = f"I've uploaded the following file content:\n\n{file_content}\n\nMy question is: {prompt}"
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error communicating with Gemini API: {str(e)}"

@app.route('/')
def index():
    if 'chat_history' not in session:
        session['chat_history'] = []
    return render_template('index.html', chat_history=session['chat_history'])

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form.get('message', '')
    file = request.files.get('file')
    
    file_content = None
    file_info = None
    
    if file and file.filename:
        file_content = parse_file(file)
        file_info = {
            "filename": file.filename,
            "type": os.path.splitext(file.filename)[1][1:].upper()
        }
    
    # Get response from Gemini
    gemini_response = get_gemini_response(user_message, file_content)
    
    # Update chat history
    new_interaction = {
        "user_message": user_message,
        "bot_response": gemini_response,
        "file": file_info
    }
    
    if 'chat_history' not in session:
        session['chat_history'] = []
    
    session['chat_history'].append(new_interaction)
    session.modified = True
    
    return jsonify({
        "response": gemini_response,
        "file_processed": file_info is not None
    })

@app.route('/clear', methods=['POST'])
def clear_history():
    session['chat_history'] = []
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(debug=True)