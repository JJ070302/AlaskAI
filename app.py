import tempfile
from dotenv import load_dotenv
from openai import OpenAI
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import fitz

load_dotenv()

app = Flask(__name__)
CORS(app)

pdf_context = ""

client = OpenAI(
	api_key = os.getenv("TOGETHER_API_KEY"),
  base_url = "https://api.together.xyz/v1"
  )

conversation_history = []

@app.route('/chat', methods=['POST'])
def chatbot():
    global pdf_context

    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"Error": "Empty message"}), 400

    messages = []
    if pdf_context:
        messages.append({
            "role": "system",
            "content": (
                f"You are an assistant helping the user analyze a PDF document they just uploaded. This document contains the following text:\n\n{pdf_context}"
                "you help users understand symytoms, provide feneral healthcare quiance, and recommend when to see a doctor"
                "Answer questions about it as if you are familiar with the entire document. Read pdf carefully"
                "Keep answers friendlt, accurate, and understandable for non-experts"
                "You are a helpful healthcare assistant AI for users in Alaska"
                "You are not a licensed physician, so you should advise users to consult a professional for serious or unclear symptoms."
                "Always answer less than 5 sentences"
                "Match your language with user's language"
            )
        })

    messages += conversation_history
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        messages=messages,
        temperature=0.7,
        max_tokens=512
    )

    chat_message = response.choices[0].message.content.strip()
    conversation_history.append({"role": "user", "content": user_message})
    conversation_history.append({"role": "assistant", "content": chat_message})

    return jsonify({"reply": chat_message})

@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    global pdf_context

    if 'pdf' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['pdf']

    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are allowed'}), 400

    temp_path = os.path.join(tempfile.gettempdir(), file.filename)
    file.save(temp_path)

    try:
        # Extract text from PDF
        with fitz.open(temp_path) as doc:
            pdf_text = ""
            for page in doc:
                pdf_text += page.get_text()

        pdf_context = pdf_text[:8000]
        print("Extracted PDF context:", pdf_context[:500])

        return jsonify({'message': 'PDF uploaded and parsed successfully'})
    except Exception as e:
        return jsonify({'error': f'Failed to extract PDF: {str(e)}'}), 500

if __name__ == "__main__":
		app.run(debug=True)