# app.py
import os
import json
import csv
import zipfile
import tempfile
from io import StringIO, BytesIO
import pdfplumber
import pandas as pd
from flask import Flask, render_template, request, jsonify, session
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image
import pytesseract
import speech_recognition as sr
from pydub import AudioSegment
import cv2
import numpy as np
import moviepy as mp
import concurrent.futures
from datetime import datetime
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.schema import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Configure the API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "your-secret-key")

# Initialize Gemini model
direct_model = genai.GenerativeModel('gemini-2.0-flash')

# Initialize LangChain components
def get_langchain_model():
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash", 
                                google_api_key=os.getenv("GEMINI_API_KEY"),
                                temperature=0.7)

def get_conversation_chain(history=None):
    """Initialize or retrieve conversation chain with memory"""
    # Initialize memory with existing history if provided
    if history:
        memory = ConversationBufferMemory(chat_memory=history)
    else:
        memory = ConversationBufferMemory()
    
    prompt_template = """
    You are an AI assistant helping users with their files and questions.
    Your goal is to be helpful, informative, and provide accurate answers based on the conversation history and any uploaded files.
    
    Current conversation:
    {history}
    Human: {input}
    AI Assistant:
    """
    
    PROMPT = PromptTemplate(
        input_variables=["history", "input"], 
        template=prompt_template
    )
    
    chain = ConversationChain(
        llm=get_langchain_model(),
        memory=memory,
        prompt=PROMPT,
        verbose=True
    )
    
    return chain

def classify_query(llm, query):
    """Use the LLM to classify the user's query."""
    classification_prompt = """
    You are an AI assistant. Classify the following query into one of the following categories:
    - "date_query" if the query is related to dates (e.g., asking for today's date or calculating date differences).
    - "file_query" if the query is related to uploaded files.
    - "general_query" for any other type of question.

    Query: {query}
    Classification:
    """
    prompt = classification_prompt.format(query=query)
    response = llm.predict(prompt)  # Use predict instead of generate_content
    return response.strip().lower()

def handle_date_query(query):
    """Provide the LLM with the current date and time to handle date-related queries."""
    # Get the current date and time
    now = datetime.now()
    current_date = now.strftime('%A, %d %B %Y')
    current_time = now.strftime('%H:%M:%S')

    # Provide the LLM with the query, current date/time, and conversation history
    date_prompt = f"""
    You are an AI assistant. The user has asked a date-related question. Use the following information to answer:
    - Current date: {current_date}
    - Current time: {current_time}

    User's query: {query}
    """
    return date_prompt.strip()

def extract_text_from_pdf(file):
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        text = f"Error extracting text from PDF: {str(e)}"
    return text

def process_image(file):
    """Extract text from images using OCR and perform basic analysis"""
    try:
        image = Image.open(file)
        
        # Create a summary of image properties
        image_info = {
            "Format": image.format,
            "Size": image.size,
            "Mode": image.mode
        }
        
        # Extract text using OCR
        text = pytesseract.image_to_string(image)
        
        result = f"Image Analysis:\n"
        result += f"Properties: {json.dumps(image_info, indent=2)}\n\n"
        
        if text.strip():
            result += f"Extracted Text:\n{text}\n"
        else:
            result += "No text detected in the image.\n"
            
        # Save a compressed version if it's a large image
        if os.path.getsize(file.name) > 1000000:  # If larger than 1MB
            output = BytesIO()
            image.save(output, format=image.format, optimize=True, quality=85)
            compression_ratio = os.path.getsize(file.name) / output.tell()
            result += f"\nImage compressed by {compression_ratio:.2f}x"
        
        return result
    except Exception as e:
        return f"Error processing image: {str(e)}"

def transcribe_audio(file):
    """Transcribe audio files using speech recognition"""
    try:
        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_path = temp_file.name
            
        # Convert to wav if needed using pydub
        sound = AudioSegment.from_file(file)
        sound.export(temp_path, format="wav")
        
        # Perform speech recognition
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
        
        os.unlink(temp_path)
        
        # Return the transcription
        return f"Audio Transcription:\n{text}"
    except Exception as e:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        return f"Error transcribing audio: {str(e)}"

def process_video(file):
    """Extract frames, info, and audio from video files"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_path = temp_file.name
            temp_file.write(file.read())
        
        # Get video info using OpenCV
        video = cv2.VideoCapture(temp_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Extract a few sample frames
        frames = []
        if frame_count > 0:
            sample_points = [int(frame_count * x) for x in [0.25, 0.5, 0.75]]
            for frame_idx in sample_points:
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                success, frame = video.read()
                if success:
                    # Convert to grayscale to save space and describe
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    avg_brightness = np.mean(gray)
                    frames.append(f"Frame at {frame_idx/fps:.2f}s - Avg brightness: {avg_brightness:.2f}")
        
        video.release()
        
        # Try to extract audio using moviepy
        audio_text = ""
        try:
            video_clip = mp.VideoFileClip(temp_path)
            if video_clip.audio:
                audio_path = temp_path + "_audio.wav"
                video_clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
                
                # Transcribe extracted audio
                recognizer = sr.Recognizer()
                with sr.AudioFile(audio_path) as source:
                    audio_data = recognizer.record(source)
                    transcription = recognizer.recognize_google(audio_data)
                    audio_text = f"Audio Transcription Sample:\n{transcription[:500]}...\n"
                
                os.unlink(audio_path)
            video_clip.close()
        except:
            audio_text = "No audio track found or unable to transcribe.\n"
        
        # Clean up
        os.unlink(temp_path)
        
        result = "Video Analysis:\n"
        result += f"Duration: {duration:.2f} seconds\n"
        result += f"Resolution: {width}x{height}\n"
        result += f"FPS: {fps}\n"
        result += f"Frame Count: {frame_count}\n\n"
        result += "Sample Frames Info:\n" + "\n".join(frames) + "\n\n"
        result += audio_text
        
        return result
    except Exception as e:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        return f"Error processing video: {str(e)}"

def _process_zip_member(zip_ref, member):
    """Process a single file from a zip archive with better content handling"""
    filename = os.path.basename(member.filename)
    ext = os.path.splitext(filename)[1].lower()
    
    # Skip directories, empty files, and files over 10MB
    if member.is_dir() or member.file_size == 0 or member.file_size > 10 * 1024 * 1024:
        if member.file_size > 10 * 1024 * 1024:
            return {'filename': filename, 'content': "File too large to process (over 10MB)."}
        return None
        
    try:
        # Read the file data into a BytesIO object
        raw = zip_ref.read(member)
        stream = BytesIO(raw)
        stream.name = filename  # Add filename attribute for compatibility
        
        # Set a reasonable size limit for text extraction
        MAX_TEXT_LENGTH = 10000
        
        # Process based on file type
        if ext == '.pdf':
            content = extract_text_from_pdf(stream)
        elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            content = "Image file detected - skipping detailed processing in ZIP context."
        elif ext in ['.mp3', '.wav', '.ogg', '.flac']:
            content = "Audio file detected - skipping transcription in ZIP context."
        elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
            content = "Video file detected - skipping analysis in ZIP context."
        elif ext == '.csv':
            # For CSV files, read rows but limit the output
            try:
                df = pd.read_csv(BytesIO(raw), encoding='utf-8', nrows=100)
                content = {
                    'text': df.head(5).to_string(),
                    'columns': df.columns.tolist(),
                    'rows': len(df),
                    'sample': "First 5 rows shown above"
                }
                content = json.dumps(content, indent=2)
            except:
                content = "CSV file could not be parsed."
        elif ext == '.json':
            try:
                obj = json.loads(raw.decode('utf-8', errors='ignore'))
                # Limit JSON representation to avoid very large outputs
                content = json.dumps(obj, indent=2)[:MAX_TEXT_LENGTH] + "..."
            except:
                content = "JSON file could not be parsed."
        elif ext in ['.txt', '.py', '.js', '.html', '.css']:
            content = raw.decode('utf-8', errors='ignore')[:MAX_TEXT_LENGTH]
            if len(content) == MAX_TEXT_LENGTH:
                content += "\n... (content truncated)"
        else:
            return {'filename': filename, 'content': f"Unsupported file type: {ext}"}

        return {'filename': filename, 'content': content}
    except Exception as e:
        return {'filename': filename, 'content': f"Error processing {filename}: {str(e)}"}

def extract_from_zip(file):
    """Extract and process contents from a zip file with better user query handling"""
    try:
        MAX_ZIP_SIZE = 50 * 1024 * 1024  # 50MB limit
        MAX_FILES_TO_PROCESS = 20  # Process at most 20 files from a ZIP
        
        zip_file_bytes = file.read()
        
        # Check file size
        if len(zip_file_bytes) > MAX_ZIP_SIZE:
            return "ZIP file is too large to process. Please use a ZIP file under 50MB."
        
        zip_bytesio = BytesIO(zip_file_bytes)
        result = []
        
        with zipfile.ZipFile(zip_bytesio) as zip_ref:
            # Get list of all files in the zip
            all_files = zip_ref.namelist()
            
            # Get a list of non-directory members, limit the number of files
            members = [m for m in zip_ref.infolist() if not m.is_dir()]
            if len(members) > MAX_FILES_TO_PROCESS:
                members = members[:MAX_FILES_TO_PROCESS]
                file_limit_notice = f"Note: Only processing the first {MAX_FILES_TO_PROCESS} files as the ZIP contains {len(all_files)} files."
            else:
                file_limit_notice = ""
            
            # Process each file with timeouts
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                future_to_member = {
                    executor.submit(_process_zip_member, zip_ref, m): m 
                    for m in members
                }
                
                # Add a timeout of 30 seconds per file
                for future in concurrent.futures.as_completed(future_to_member, timeout=30):
                    try:
                        entry = future.result()
                        if entry:
                            result.append(entry)
                    except concurrent.futures.TimeoutError:
                        member = future_to_member[future]
                        result.append({
                            'filename': member.filename,
                            'content': "Processing timed out for this file."
                        })
                    except Exception as e:
                        member = future_to_member[future]
                        result.append({
                            'filename': member.filename, 
                            'content': f"Error processing file: {str(e)}"
                        })
        
        # Format the results
        if result:
            output = {
                "zip_summary": f"ZIP archive with {len(all_files)} files{' - ' + file_limit_notice if file_limit_notice else ''}",
                "files": {}
            }
            
            for entry in result:
                output["files"][entry['filename']] = entry['content']
                
            return json.dumps(output, indent=2)
        else:
            return "ZIP file processed, but no readable content was found."
    except zipfile.BadZipFile:
        return "The uploaded file is not a valid ZIP archive."
    except Exception as e:
        return f"Error processing ZIP file: {str(e)}"

def parse_file(file):
    filename = file.filename
    file_ext = os.path.splitext(filename)[1].lower()
    
    try:
        if file_ext == '.pdf':
            return extract_text_from_pdf(file)
        elif file_ext == '.zip':
            # Reset file pointer to beginning
            file.seek(0)
            return extract_from_zip(file)
        elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            return process_image(file)
        elif file_ext in ['.mp3', '.wav', '.ogg', '.flac']:
            return transcribe_audio(file)
        elif file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
            return process_video(file)
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
    """Legacy direct model generation - without memory"""
    try:
        if file_content:
            prompt = f"I've uploaded the following file content:\n\n{file_content}\n\nMy question is: {prompt}"
        
        response = direct_model.generate_content(prompt)
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
    
    # Process uploaded file if present
    if file and file.filename:
        file_content = parse_file(file)
        file_info = {
            "filename": file.filename,
            "type": os.path.splitext(file.filename)[1][1:].upper()
        }
    
    # Initialize the LLM
    llm = get_langchain_model()
    
    # Classify the query
    query_type = classify_query(llm, user_message)
    
    # Initialize or restore chat history
    if 'memory_messages' not in session:
        session['memory_messages'] = []
    
    chat_history = ChatMessageHistory()
    for msg in session['memory_messages']:
        if msg['type'] == 'human':
            chat_history.add_user_message(msg['content'])
        else:
            chat_history.add_ai_message(msg['content'])
    
    # Reuse the conversation chain with the existing memory
    conversation = get_conversation_chain(chat_history)
    
    if query_type == "date_query":
        # Handle date-related queries
        input_text = handle_date_query(user_message)
    elif query_type == "file_query" and file_content:
        # Handle file-related queries
        ext = os.path.splitext(file.filename)[1].lower()
        if ext == '.zip':
            input_text = f"I've uploaded a ZIP file: {file.filename}. Here's the extracted content: {file_content}. Based on the above extracted content from the ZIP file, please answer this specific question: {user_message}. If the question refers to specific files or data within the ZIP, please focus on those parts of the content."
        else:
            input_text = f"I've uploaded a file: {file.filename}. Here's the content or analysis:\n\n{file_content}\n\nMy question or request is: {user_message}"
    else:
        # General query
        input_text = user_message
    
    # Get response using LangChain
    gemini_response = conversation.predict(input=input_text)
    
    # Update session memory
    session['memory_messages'].append({"type": "human", "content": user_message})
    session['memory_messages'].append({"type": "ai", "content": gemini_response})
    session.modified = True
    
    # Update chat history for display
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
    session.pop('chat_history', None)
    session.pop('memory_messages', None)
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(debug=True)