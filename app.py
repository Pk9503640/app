# --- FINAL APP CODE FOR RENDER (READY TO DEPLOY) ---
import os
import json
import time
import base64
from io import BytesIO
import traceback

from flask import Flask, request, Response
from flask_cors import CORS
from PIL import Image

import google.generativeai as genai
from groq import Groq
import firebase_admin
from firebase_admin import credentials, firestore

# --- INITIALIZATION ---
app = Flask(__name__)
CORS(app)

# ✅ Root route for Render health check
@app.route("/")
def home():
    return {"status": "ok", "message": "Service is live 🚀"}

# Initialize Firebase from an environment variable for Render
try:
    if not firebase_admin._apps:
        firebase_creds_json_str = os.environ.get('FIREBASE_CREDENTIALS_JSON')
        if firebase_creds_json_str:
            firebase_creds_dict = json.loads(firebase_creds_json_str)
            # Fix for Render private key line breaks
            firebase_creds_dict['private_key'] = firebase_creds_dict['private_key'].replace('\\n', '\n')
            cred = credentials.Certificate(firebase_creds_dict)
            firebase_admin.initialize_app(cred)
            print("✅ Firebase connected successfully from environment variable.")
        else:
            print("❌ FIREBASE_CREDENTIALS_JSON environment variable not set.")
    db = firestore.client()
except Exception as e:
    print(f"❌ Firebase initialization failed. Error: {e}")
    db = None

# Initialize API Clients using Environment Variables
try:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    GOOGLE_API_KEY_1 = os.environ.get("GOOGLE_API_KEY_1")

    if not GROQ_API_KEY or not GOOGLE_API_KEY_1:
        print("❌ API key environment variables are missing.")
        groq_client = None
        genai_model = None
    else:
        groq_client = Groq(api_key=GROQ_API_KEY)
        genai.configure(api_key=GOOGLE_API_KEY_1)
        print("✅ Groq and Gemini API clients initialized.")
except Exception as e:
    print(f"❌ FATAL ERROR during API client initialization: {e}")
    groq_client = None
    genai.model = None

# --- HELPER FUNCTIONS AND ENDPOINT ---

MODEL_MAP = {
    "AutoIntelligent": "llama3-70b-8192",
    "GroqLlama3": "llama3-8b-8192",
    "Gemini1.5": "gemini-1.5-flash",
}
SYSTEM_PROMPT = "You are a helpful AI assistant. Provide clear and concise answers."

def save_message_to_db(chat_id, message_data):
    if not db: return
    try:
        db.collection('chats').document(chat_id).collection('messages').add(message_data)
    except Exception as e:
        print(f"DB_ERROR: Could not save message for chat '{chat_id}'. Error: {e}")

def get_history_from_db(chat_id):
    if not db: return []
    try:
        messages_ref = db.collection('chats').document(chat_id).collection('messages').order_by('timestamp').limit(20)
        messages = messages_ref.stream()
        return [{"role": msg.to_dict()['role'], "content": msg.to_dict()['content']} for msg in messages]
    except Exception as e:
        print(f"DB_ERROR: Could not get history for chat '{chat_id}'. Error: {e}")
        return []

def process_image(image_file):
    img = Image.open(image_file.stream)
    img.thumbnail((1024, 1024))
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def handle_gemini_request(messages, image_base64=None):
    if not genai:
        def error_stream(): yield "Error: Gemini API is not configured."
        return error_stream, lambda: "Error: Gemini API is not configured."

    try:
        gemini_history = [{'role': "model" if msg["role"] == "assistant" else "user", 'parts': [{'text': msg['content']}]} for msg in messages[1:]]
        current_prompt = gemini_history.pop() if gemini_history else {'parts': [{'text': messages[-1]['content']}]}
        prompt_parts = [current_prompt['parts'][0]['text']]
        if image_base64:
            prompt_parts.append({"inline_data": {"mime_type": "image/jpeg", "data": image_base64}})

        model = genai.GenerativeModel("gemini-1.5-flash")
        chat_session = model.start_chat(history=gemini_history)
        response_stream = chat_session.send_message(prompt_parts, stream=True)

        full_text_container = {"text": ""}
        def generate_stream():
            for chunk in response_stream:
                if chunk.text:
                    full_text_container["text"] += chunk.text
                    yield chunk.text

        get_full_response_func = lambda: full_text_container["text"]
        return generate_stream, get_full_response_func
    except Exception as e:
        print(f"GEMINI_ERROR: {e}")
        error_message = f"An error occurred with the Gemini API: {e}"
        def error_stream(): yield error_message
        return error_stream, lambda: error_message


@app.route('/solve', methods=['POST'])
def solve_problem():
    try:
        prompt_text = request.form.get('prompt', '')
        chat_id = request.form.get('chat_id', f"chat_{int(time.time())}")
        model_choice_key = request.form.get('model', 'AutoIntelligent')
        image_file = request.files.get('image')

        user_message_content = f"{prompt_text} [Image Attached]" if image_file else prompt_text
        save_message_to_db(chat_id, {"role": "user", "content": user_message_content, "timestamp": time.time()})

        history = get_history_from_db(chat_id)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

        response_streamer = None
        get_full_response_func = lambda: ""

        if image_file:
            streamer_gen, get_full_response_func = handle_gemini_request(messages, process_image(image_file))
            response_streamer = streamer_gen()
        else:
            actual_model_id = MODEL_MAP.get(model_choice_key)
            if not actual_model_id:
                return Response(json.dumps({"error": f"Invalid model: {model_choice_key}"}), status=400)

            if 'llama' in actual_model_id:
                if not groq_client: return Response(json.dumps({"error": "Groq client not initialized"}), status=500)
                try:
                    chat_completion = groq_client.chat.completions.create(model=actual_model_id, messages=messages, stream=True)
                    full_text_container = {"text": ""}
                    def generate_groq_stream():
                        for chunk in chat_completion:
                            content = chunk.choices[0].delta.content
                            if content:
                                full_text_container["text"] += content
                                yield content
                    response_streamer = generate_groq_stream()
                    get_full_response_func = lambda: full_text_container["text"]
                except Exception as e:
                    print(f"GROQ_API_ERROR: {e}")
                    error_message = f"The Groq API call failed. Error: {e}"
                    def error_stream(): yield error_message
                    response_streamer = error_stream()
                    get_full_response_func = lambda: error_message

            elif 'gemini' in actual_model_id:
                streamer_gen, get_full_response_func = handle_gemini_request(messages)
                response_streamer = streamer_gen()

        def response_wrapper():
            yield from response_streamer
            final_bot_text = get_full_response_func()
            bot_message = {"role": "assistant", "content": final_bot_text, "timestamp": time.time()}
            save_message_to_db(chat_id, bot_message)

        return Response(response_wrapper(), mimetype='text/plain')
    except Exception as e:
        print(f"❌ Unhandled Error in /solve: {e}")
        traceback.print_exc()
        return Response(json.dumps({"error": "An internal server error occurred."}), status=500)
