import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv()

# --- Initialize Clients and Models ---
app = Flask(__name__)
CORS(app)

# 1. Groq Client for Cloud-Powered Summary
try:
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in .env file")
    groq_client = Groq(api_key=groq_api_key)
    print("Groq client initialized successfully.")
except Exception as e:
    print(f"Error initializing Groq client: {e}")
    groq_client = None

# 2. Local Lightweight Classifiers
print("Loading local classifiers...")
emotion_classifier = pipeline("text-classification", model="joeddav/distilbert-base-uncased-go-emotions-student", top_k=None)
sentiment_classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
print("All local models loaded.")


# --- Advanced Explanation and Summary Logic ---
EMOTION_KEYWORDS = {
    'disgust': ['disgusting', 'gross', 'nasty', 'sick', 'yuck', 'ew', 'sour', 'gagged', 'revolting'],
    'sadness': ['sad', 'crying', 'depressed', 'miserable', 'unhappy', 'heartbroken', 'grief'],
    'joy': ['happy', 'joy', 'pleased', 'smiling', 'yay', 'delighted', 'ecstatic'],
    'surprise': ['omg', 'surprise', 'surprised', 'wow', 'unexpected', 'unbelievable'],
    'fear': ['afraid', 'fear', 'scared', 'terrified', 'anxious', 'nervous', 'horror'],
    'anger': ['angry', 'annoyed', 'furious', 'hate', 'pissed', 'enraged', 'livid'],
    'love': ['love', 'adore', 'heart', '<3', 'charming', 'lovely'],
    'gratitude': ['grateful', 'thank you', 'thanks', 'appreciate'],
}
EMOJI_MAP = {
    'admiration': '🤩', 'amusement': '😂', 'anger': '😡', 'annoyance': '😒', 'approval': '👍', 'caring': '🤗',
    'confusion': '🤔', 'curiosity': '🧐', 'desire': '😍', 'disappointment': '😞', 'disapproval': '👎', 'disgust': '🤢',
    'embarrassment': '😳', 'excitement': '🎉', 'fear': '😨', 'gratitude': '🙏', 'grief': '😭', 'joy': '😄',
    'love': '❤️', 'nervousness': '😬', 'optimism': '😊', 'pride': '😌', 'realization': '💡', 'relief': '😮‍💨',
    'remorse': '😔', 'sadness': '😢', 'surprise': '😲', 'neutral': '😐'
}

def construct_smart_explanation(text, emotion):
    """Generates a simple, direct explanation for a detected emotion."""
    text_lower = text.lower()
    keywords_found = [kw for kw in EMOTION_KEYWORDS.get(emotion, []) if kw in text_lower]
    
    if keywords_found:
        # Directly reference the user's text and the keyword
        return f"You mentioned '{keywords_found[0]}', which shows some {emotion}."
    else:
        # Simple fallback
        return f"This emotion is detected based on your words."

def generate_summary_with_groq(text, sentiment, emotions):
    """Uses the Groq API to generate a simple, direct summary."""
    if not groq_client:
        return f"You seem to feel mostly {emotions[0]['label']} and the overall mood is {sentiment['label'].lower()}."

    analysis_context = f"Sentiment: {sentiment['label']}. Top emotions: {emotions[0]['label']}, {emotions[1]['label']}, {emotions[2]['label']}."
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an AI assistant that writes very simple, clear summaries of emotion analyses. Use easy English. Reference the user's text and explain why the emotions were detected."},
                {"role": "user", "content": f"Original Text: \"{text}\"\n\nAnalysis Context: {analysis_context}\n\nSummary:"}
            ],
            model="llama3-8b-8192",
            temperature=0.5,
            max_tokens=80,
        )
        summary = chat_completion.choices[0].message.content.strip()
        return summary
    except Exception as e:
        print(f"Groq API call failed: {e}")
        return f"You seem to feel mostly {emotions[0]['label']} and the overall mood is {sentiment['label'].lower()}."


@app.route('/analyze', methods=['POST'])
def analyze_emotions_final():
    data = request.get_json()
    user_text = data.get('text', '').strip()
    if not user_text:
        return jsonify({"error": "No text provided."}), 400

    try:
        # Step 1: Local classification
        emotion_results = emotion_classifier(user_text)[0]
        emotion_results.sort(key=lambda x: x['score'], reverse=True)
        top_3_emotions = emotion_results[:3]
        
        sentiment_result = sentiment_classifier(user_text)[0]
        sentiment_label = sentiment_result['label'].capitalize()
        sentiment_score = sentiment_result['score']
        if sentiment_label == 'Negative': sentiment_score *= -1

        sentiment_data = {"label": sentiment_label, "score": sentiment_score}
        emotions_data = [{"label": e['label'], "score": e['score']} for e in top_3_emotions]

        # Step 2: Cloud-powered summary generation
        summary = generate_summary_with_groq(user_text, sentiment_data, emotions_data)

        # Step 3: Construct the final JSON with smart explanations
        final_analysis = {
            "sentiment": sentiment_data,
            "summary": summary,
            "emotions": []
        }
        for emotion in top_3_emotions:
            final_analysis["emotions"].append({
                "label": emotion['label'],
                "score": emotion['score'],
                "explanation": construct_smart_explanation(user_text, emotion['label']),
                "emoji": EMOJI_MAP.get(emotion['label'], '😐')
            })

        return jsonify(final_analysis)
    except Exception as e:
        print(f"Error in final analysis endpoint: {e}")
        return jsonify({"error": "An internal error occurred during analysis."}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)