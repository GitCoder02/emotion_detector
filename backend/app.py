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
EMOJI_MAP = {
    'admiration': '🤩', 'amusement': '😂', 'anger': '😡', 'annoyance': '😒', 'approval': '👍', 'caring': '🤗',
    'confusion': '🤔', 'curiosity': '🧐', 'desire': '😍', 'disappointment': '😞', 'disapproval': '👎', 'disgust': '🤢',
    'embarrassment': '😳', 'excitement': '🎉', 'fear': '😨', 'gratitude': '🙏', 'grief': '😭', 'joy': '😄',
    'love': '❤️', 'nervousness': '😬', 'optimism': '😊', 'pride': '😌', 'realization': '💡', 'relief': '😮‍💨',
    'remorse': '😔', 'sadness': '😢', 'surprise': '😲', 'neutral': '😐'
}

def generate_summary_with_groq(text, sentiment, emotions):
    """Uses the Groq API to generate a simple, direct summary."""
    if not groq_client:
        return f"You seem to feel mostly {emotions[0]['label']} and the overall mood is {sentiment['label'].lower()}."

    # Create a string of the top emotions for the context
    emotion_list_str = ", ".join([e['label'] for e in emotions])
    analysis_context = f"Sentiment: {sentiment['label']}. Top emotions: {emotion_list_str}."
    
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

def generate_explanations_with_groq(text, emotions):
    """Uses the Groq API to generate explanations for each of the top emotions."""
    if not groq_client:
        return {e['label']: "This emotion is detected based on your words." for e in emotions}

    explanations = {}
    for emotion in emotions:
        try:
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an AI assistant that explains why a specific emotion was detected in a user's text. Use simple, clear language. Refer to the user's text in your explanation."},
                    {"role": "user", "content": f"Original Text: \"{text}\"\n\nEmotion: \"{emotion['label']}\"\n\nExplain in one sentence why this emotion was detected:"}
                ],
                model="llama3-8b-8192",
                temperature=0.5,
                max_tokens=40,
            )
            explanation = chat_completion.choices[0].message.content.strip()
            explanations[emotion['label']] = explanation
        except Exception as e:
            print(f"Groq API call for explanation failed: {e}")
            explanations[emotion['label']] = f"The overall tone of the text suggests {emotion['label']}."
            
    return explanations


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
        # --- CHANGE: Get top 4 emotions ---
        top_emotions_raw = emotion_results[:4]
        
        sentiment_result = sentiment_classifier(user_text)[0]
        sentiment_label = sentiment_result['label'].capitalize()
        sentiment_score = sentiment_result['score']
        if sentiment_label == 'Negative': sentiment_score *= -1

        sentiment_data = {"label": sentiment_label, "score": sentiment_score}
        
        # Use raw emotions data for generating explanations, as it's the "true" output
        emotions_data_for_llm = [{"label": e['label'], "score": e['score']} for e in top_emotions_raw]

        # Step 2: Cloud-powered summary and explanation generation
        summary = generate_summary_with_groq(user_text, sentiment_data, emotions_data_for_llm)
        explanations = generate_explanations_with_groq(user_text, emotions_data_for_llm)

        # --- NEW: Normalize the scores of the top 4 emotions for frontend display ---
        total_score_of_top_4 = sum(e['score'] for e in top_emotions_raw)
        
        final_emotions = []
        if total_score_of_top_4 > 0:
            for emotion in top_emotions_raw:
                normalized_score = emotion['score'] / total_score_of_top_4
                final_emotions.append({
                    "label": emotion['label'],
                    "score": normalized_score, # Use the new normalized score
                    "explanation": explanations.get(emotion['label'], "This emotion is detected based on your words."),
                    "emoji": EMOJI_MAP.get(emotion['label'], '😐')
                })
        else:
            # Fallback for cases where no emotions are detected
            final_emotions = [{"label": "neutral", "score": 1, "explanation": "No significant emotion was detected.", "emoji": "😐"}]


        # Step 3: Construct the final JSON
        final_analysis = {
            "sentiment": sentiment_data,
            "summary": summary,
            "emotions": final_emotions
        }

        return jsonify(final_analysis)
    except Exception as e:
        print(f"Error in final analysis endpoint: {e}")
        return jsonify({"error": "An internal error occurred during analysis."}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)