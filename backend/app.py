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
import re

# Disable warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

# NLTK & SpaCy Init
nltk.download("punkt")
nltk.download("stopwords")
from nltk.corpus import stopwords
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
    "original_text": "",  # Store original text for better context
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

# === 📏 Text Analysis Functions ===
def analyze_text_complexity(text):
    """Analyze text to determine appropriate question count and complexity"""
    words = text.split()
    word_count = len(words)
    sentences = text.split('.')
    sentence_count = len([s for s in sentences if s.strip()])
    
    # Calculate reading complexity (simple version)
    avg_sentence_length = word_count / max(sentence_count, 1)
    
    # Determine question count based on text length and complexity
    if word_count < 50:
        max_questions = 2
        complexity = "simple"
    elif word_count < 150:
        max_questions = 3
        complexity = "basic"
    elif word_count < 300:
        max_questions = 5
        complexity = "moderate"
    elif word_count < 600:
        max_questions = 7
        complexity = "detailed"
    elif word_count < 1000:
        max_questions = 10
        complexity = "comprehensive"
    else:
        max_questions = 12
        complexity = "extensive"
    
    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_length,
        "max_questions": max_questions,
        "complexity": complexity
    }

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

# === ❓ Enhanced Answer Extraction ===
def extract_answers(text, complexity_level="moderate"):
    """Enhanced answer extraction based on text complexity"""
    doc = nlp(text)
    answers = []
    stop_words = set(stopwords.words('english'))
    
    # Priority scoring for different answer types
    answer_candidates = {}
    
    # Extract named entities (higher priority)
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE"]:
            cleaned = ent.text.strip()
            if len(cleaned.split()) >= 1 and cleaned.lower() not in stop_words:
                answer_candidates[cleaned] = answer_candidates.get(cleaned, 0) + 3
    
    # Extract noun phrases (medium priority)
    for chunk in doc.noun_chunks:
        cleaned = chunk.text.strip()
        # Filter out pronouns and common words
        if (len(cleaned.split()) >= 2 and 
            cleaned.lower() not in {"this", "that", "which", "it", "they", "these", "those"} and
            not any(word.lower() in stop_words for word in cleaned.split()[:2])):
            answer_candidates[cleaned] = answer_candidates.get(cleaned, 0) + 2
    
    # Extract key phrases using dependency parsing
    for token in doc:
        if (token.pos_ in ["NOUN", "PROPN"] and 
            token.dep_ in ["nsubj", "dobj", "pobj"] and
            len(token.text) > 3):
            # Get the full phrase including modifiers
            phrase = ""
            for child in token.children:
                if child.dep_ in ["amod", "compound", "det"]:
                    phrase = child.text + " " + phrase
            phrase += token.text
            phrase = phrase.strip()
            if len(phrase.split()) >= 1:
                answer_candidates[phrase] = answer_candidates.get(phrase, 0) + 1
    
    # Sort by priority and return top candidates
    sorted_answers = sorted(answer_candidates.items(), key=lambda x: x[1], reverse=True)
    
    # Adjust number of answers based on complexity
    if complexity_level == "simple":
        max_answers = 3
    elif complexity_level == "basic":
        max_answers = 5
    elif complexity_level == "moderate":
        max_answers = 8
    elif complexity_level == "detailed":
        max_answers = 12
    else:
        max_answers = 15
    
    return [answer[0] for answer in sorted_answers[:max_answers]]

# === 🧩 Enhanced Question Generation ===
def generate_questions(text, answers, max_questions=5, complexity_level="moderate"):
    """Generate questions with improved variety and quality"""
    questions = []
    used_questions = set()
    used_answers = set()
    
    # Question type templates for variety
    question_starters = [
        "What is",
        "Who is",
        "Where is",
        "When did",
        "How does",
        "Why is",
        "Which",
        "Describe"
    ]
    
    for i, answer in enumerate(answers):
        if len(questions) >= max_questions:
            break
            
        # Skip if answer is too short or already used
        if len(answer.strip()) < 2 or answer in used_answers:
            continue
            
        # Create context with highlighted answer
        context = text.replace(answer, f"<hl> {answer} <hl>", 1)
        
        # Try different question generation approaches
        generation_methods = [
            f"generate question: {context}",
            f"ask about: {answer} context: {context[:200]}",
            f"create question for answer '{answer}': {context[:150]}"
        ]
        
        best_question = None
        best_score = 0
        
        for method in generation_methods:
            try:
                input_ids = qg_tokenizer.encode(method, return_tensors="pt", max_length=512, truncation=True)
                with torch.no_grad():
                    output_ids = qg_model.generate(
                        input_ids, 
                        max_length=64, 
                        num_beams=6, 
                        early_stopping=True,
                        temperature=0.8,
                        do_sample=True,
                        top_p=0.9
                    )
                question = qg_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
                
                # Score question quality
                score = score_question_quality(question, answer, text)
                if score > best_score and question not in used_questions:
                    best_question = question
                    best_score = score
            except:
                continue
        
        if best_question and best_score > 0.3:  # Quality threshold
            # Improve answer quality
            improved_answer = improve_answer_quality(answer, text, best_question)
            
            questions.append({
                "question": best_question,
                "answer": improved_answer,
                "confidence": min(best_score, 1.0)
            })
            used_questions.add(best_question)
            used_answers.add(answer)
    
    # Sort by confidence score
    questions.sort(key=lambda x: x.get('confidence', 0), reverse=True)
    return questions

def score_question_quality(question, answer, context):
    """Score the quality of a generated question"""
    score = 0.5  # Base score
    
    # Check if question is complete
    if question.endswith('?'):
        score += 0.2
    
    # Check question length (not too short, not too long)
    word_count = len(question.split())
    if 5 <= word_count <= 15:
        score += 0.2
    elif word_count < 3:
        score -= 0.3
    
    # Check if question contains question words
    question_words = ['what', 'who', 'where', 'when', 'how', 'why', 'which', 'describe']
    if any(qw in question.lower() for qw in question_words):
        score += 0.1
    
    # Penalize repetitive or generic questions
    generic_phrases = ['this', 'that', 'it is', 'there is']
    if any(phrase in question.lower() for phrase in generic_phrases):
        score -= 0.2
    
    return score

def improve_answer_quality(answer, context, question):
    """Improve answer quality by adding context when appropriate"""
    # If answer is very short, try to expand it with context
    if len(answer.split()) <= 2:
        doc = nlp(context)
        
        # Find sentences containing the answer
        sentences_with_answer = []
        for sent in doc.sents:
            if answer.lower() in sent.text.lower():
                sentences_with_answer.append(sent.text.strip())
        
        if sentences_with_answer:
            # Get the most relevant sentence
            best_sentence = min(sentences_with_answer, key=len)
            if len(best_sentence.split()) <= 20:  # Don't make it too long
                return best_sentence
    
    return answer

# === 🤖 Enhanced Groq Chat ===
def chat_with_gpt(chat_log):
    """Enhanced chat with better context awareness"""
    # Add context about the document if available
    if memory.get("original_text"):
        system_context = f"""You are a helpful assistant discussing a document. 
        Here's the original text for reference: {memory['original_text'][:500]}...
        
        Please provide clear, accurate answers based on this content. If the question is not related to the document, you can still answer helpfully."""
        
        # Update system message if it exists
        if chat_log and chat_log[0]["role"] == "system":
            chat_log[0]["content"] = system_context
        else:
            chat_log.insert(0, {"role": "system", "content": system_context})
    
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=chat_log,
        temperature=0.7,
        max_tokens=1000
    )
    return response.choices[0].message.content.strip()

# === 📡 API Endpoints ===
@app.route("/summarize", methods=["POST"])
def api_summarize():
    try:
        text = request.json.get("text") if request.is_json else get_text_from_input(request)
        if not text:
            return jsonify({"error": "No input text provided."}), 400
        
        # Store original text for better chat context
        memory["original_text"] = text
        
        summary = summarize(text)
        memory["summary"] = summary
        
        # Enhanced system message with document info
        analysis = analyze_text_complexity(text)
        system_message = f"""You are discussing a document with the following summary: {summary}

Document stats: {analysis['word_count']} words, {analysis['complexity']} complexity level.
Please provide helpful, accurate responses based on this content."""
        
        memory["chat_log"] = [{"role": "system", "content": system_message}]
        
        return jsonify({
            "summary": summary,
            "analysis": analysis
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/generate-questions", methods=["POST"])
def api_generate_questions():
    try:
        text = request.json.get("text") if request.is_json else get_text_from_input(request)
        if not text:
            return jsonify({"error": "No input text provided."}), 400
        
        # Analyze text complexity
        analysis = analyze_text_complexity(text)
        max_questions = analysis["max_questions"]
        complexity = analysis["complexity"]
        
        # Extract answers with improved quality
        answers = extract_answers(text, complexity)
        
        # Generate questions dynamically based on text length
        questions = generate_questions(text, answers, max_questions, complexity)
        
        return jsonify({
            "questions": questions,
            "analysis": analysis,
            "total_generated": len(questions)
        })
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
        
        # Maintain reasonable context window
        if len(memory["chat_log"]) > 8:
            # Keep system message and last 6 exchanges
            memory["chat_log"] = [memory["chat_log"][0]] + memory["chat_log"][-6:]
        
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
    app.run(host="0.0.0.0", port=5001, debug=True)