import os
import gdown
from io import BytesIO
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import pytesseract
import nltk
import spacy
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dotenv import load_dotenv
from groq import Groq

# Disable warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

# NLTK & SpaCy Init
nltk.download("punkt")
nlp = spacy.load("en_core_web_sm")

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY is not set in the environment variables.")
client = Groq(api_key=api_key)

# Flask App Init
app = Flask(__name__, static_folder="frontend/build", static_url_path="/")
CORS(app, origins=['http://localhost:5173'])


# === 🔽 Download & Load Models from Google Drive ===

# Model paths
MODEL_DIR = "./models"
SUMMARY_DIR = os.path.join(MODEL_DIR, "bart-edu-finetuned")
QG_DIR = os.path.join(MODEL_DIR, "t5-qg-model")

# Folder IDs from sharable Google Drive links
SUMMARY_FOLDER_ID = "1clRwpGrcz0sRNyTqFIdf-rm-zVRIWdU9"
QG_FOLDER_ID = "16j57bS12MSZ9FRdtpyI_4vMOEnpPKgxg"

# Download if not already present
def download_model_from_gdrive(folder_id, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"🔽 Downloading model to {output_dir}")
        gdown.download_folder(id=folder_id, output=output_dir, quiet=False, use_cookies=False)

download_model_from_gdrive(SUMMARY_FOLDER_ID, SUMMARY_DIR)
download_model_from_gdrive(QG_FOLDER_ID, QG_DIR)

# Load HuggingFace models
summarizer_tokenizer = AutoTokenizer.from_pretrained(SUMMARY_DIR)
summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARY_DIR)
qg_tokenizer = AutoTokenizer.from_pretrained(QG_DIR)
qg_model = AutoModelForSeq2SeqLM.from_pretrained(QG_DIR)

# === 🔄 App Memory ===
memory = {
    "summary": "",
    "chat_log": []
}

# === 🖼 Text Extraction ===
def extract_text_from_image(file_stream):
    tesseract_path = os.getenv("TESSERACT_PATH")
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    image = Image.open(file_stream)
    return pytesseract.image_to_string(image).strip()

def get_text_from_input(req):
    if 'file' in req.files:
        file = req.files['file']
        ext = file.filename.lower()
        if ext.endswith(".pdf"):
            raise ValueError("PDF not supported.")
        elif ext.endswith((".png", ".jpg", ".jpeg")):
            return extract_text_from_image(BytesIO(file.read()))
        else:
            raise ValueError("Unsupported file format.")
    return req.form.get("text", "").strip()

# === 🧠 Summarization ===
def summarize(text):
    input_tokens = summarizer_tokenizer.tokenize(text)
    token_count = len(input_tokens)

    if token_count <= 150:
        min_len, max_len = 30, 80
    elif token_count <= 300:
        min_len, max_len = 60, 150
    elif token_count <= 600:
        min_len, max_len = 100, 300
    else:
        min_len, max_len = 150, 500

    inputs = summarizer_tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    with torch.no_grad():
        summary_ids = summarizer_model.generate(
            inputs.input_ids,
            max_length=max_len,
            min_length=min_len,
            length_penalty=1.0,
            num_beams=9,
            early_stopping=True
        )
    return summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# === ❓ Answer Extraction ===
def extract_answers(text):
    doc = nlp(text)
    answers = set()

    for chunk in doc.noun_chunks:
        cleaned = chunk.text.strip()
        if len(cleaned.split()) >= 2 and cleaned.lower() not in {"this", "that", "which", "it"}:
            answers.add(cleaned)

    for ent in doc.ents:
        cleaned = ent.text.strip()
        if len(cleaned.split()) >= 2 and cleaned.lower() not in {"this", "that", "which", "it"}:
            answers.add(cleaned)

    return list(answers)

# === 🧩 Question Generation ===
def generate_questions(text, answers, max_questions=5):
    questions = []
    used = set()
    for answer in answers:
        context = text.replace(answer, f"<hl> {answer} <hl>", 1)
        input_text = f"generate question: {context}"
        input_ids = qg_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            output_ids = qg_model.generate(input_ids, max_length=64, num_beams=6, early_stopping=True)
        question = qg_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if question not in used:
            questions.append({"question": question.strip(), "answer": answer.strip()})
            used.add(question)
        if len(questions) >= max_questions:
            break
    return questions

# === 🤖 Groq Chat ===
def chat_with_gpt(chat_log):
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=chat_log,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

# === 📡 API Endpoints ===
@app.route("/summarize", methods=["POST"])
def api_summarize():
    try:
        text = request.json.get("text") if request.is_json else get_text_from_input(request)
        if not text:
            return jsonify({"error": "No input text provided."}), 400
        summary = summarize(text)
        memory["summary"] = summary
        memory["chat_log"] = [{"role": "system", "content": f"This conversation is based on the following report summary:\n\n{summary}"}]
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/generate-questions", methods=["POST"])
def api_generate_questions():
    try:
        text = request.json.get("text") if request.is_json else get_text_from_input(request)
        if not text:
            return jsonify({"error": "No input text provided."}), 400
        answers = extract_answers(text)
        questions = generate_questions(text, answers)
        return jsonify({"questions": questions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ask", methods=["POST"])
def ask_question():
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        if not question:
            return jsonify({"error": "No question provided."}), 400
        memory["chat_log"].append({"role": "user", "content": question})
        if len(memory["chat_log"]) > 6:
            memory["chat_log"] = [memory["chat_log"][0]] + memory["chat_log"][-5:]
        answer = chat_with_gpt(memory["chat_log"])
        memory["chat_log"].append({"role": "assistant", "content": answer})
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === 🌐 React Static Files (Optional) ===
@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")

@app.errorhandler(404)
def not_found(e):
    return send_from_directory(app.static_folder, "index.html")

# === 🚀 Run App ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=5001)
