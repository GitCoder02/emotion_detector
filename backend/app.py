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

# 1. Groq Client for Cloud-Powered Analysis
try:
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in .env file")
    groq_client = Groq(api_key=groq_api_key)
    print("Groq client initialized successfully.")
except Exception as e:
    print(f"Error initializing Groq client: {e}")
    groq_client = None

# 2. Local Classifiers
print("Loading local classifiers...")
emotion_classifier = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
sentiment_classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
print("All local models loaded.")


# --- Emoji Map ---
EMOJI_MAP = {
    'admiration': '🤩', 'amusement': '😂', 'anger': '😡', 'annoyance': '😒', 'approval': '👍', 'caring': '🤗',
    'confusion': '🤔', 'curiosity': '🧐', 'desire': '😍', 'disappointment': '😞', 'disapproval': '👎', 'disgust': '🤢',
    'embarrassment': '😳', 'excitement': '🎉', 'fear': '😨', 'gratitude': '🙏', 'grief': '😭', 'joy': '😄',
    'love': '❤️', 'nervousness': '😬', 'optimism': '😊', 'pride': '😌', 'realization': '💡', 'relief': '😮‍💨',
    'remorse': '😔', 'sadness': '😢', 'surprise': '😲', 'neutral': '😐'
}

def get_refined_analysis_with_groq(text, emotions):
    """
    Uses the Groq API (Llama 3) to act as an expert reviewer.
    It refines the emotion list, provides a summary, and gives explanations.
    """
    if not groq_client:
        return {
            "summary": "Could not generate an AI summary.",
            "emotions": emotions
        }

    candidate_emotions = ", ".join([f"'{e['label']}'" for e in emotions])

    system_prompt = (
        "You are an expert emotion analysis AI. You will be given a user's text and a list of candidate emotions "
        "detected by a less advanced model. Your tasks are:\n"
        "1. From the candidate list, identify the 1 to 4 most accurate emotions for the text.\n"
        "2. For each accurate emotion you identify, provide a simple, one-sentence explanation referencing the text.\n"
        "3. Write an insightful, user-friendly summary (2-3 sentences) of the overall emotional tone. Describe the primary feeling and how any secondary emotions add complexity.\n"
        "4. Format your response as a JSON object with three keys: 'summary', 'emotions'. "
        "The 'emotions' key should be an array of objects, where each object has 'label' and 'explanation' keys. "
        "Only include the emotions you have identified as accurate."
    )
    
    user_prompt = f"Text: \"{text}\"\nCandidate Emotions: [{candidate_emotions}]"

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama3-8b-8192",
            temperature=0.3,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        
        response_content = chat_completion.choices[0].message.content
        refined_data = json.loads(response_content)
        return refined_data

    except Exception as e:
        print(f"Groq API call for refined analysis failed: {e}")
        return {
            "summary": "An error occurred while generating the AI summary.",
            "emotions": emotions
        }


@app.route('/analyze', methods=['POST'])
def analyze_emotions_final():
    data = request.get_json()
    user_text = data.get('text', '').strip()
    if not user_text:
        return jsonify({"error": "No text provided."}), 400

    try:
        # --- Stage 1: Candidate Generation (Local Model) ---
        emotion_results = emotion_classifier(user_text)[0]
        emotion_results.sort(key=lambda x: x['score'], reverse=True)
        
        candidate_emotions = emotion_results[:5]

        sentiment_result = sentiment_classifier(user_text)[0]
        sentiment_label = sentiment_result['label'].capitalize()
        sentiment_score = sentiment_result['score']
        if sentiment_label == 'Negative': sentiment_score *= -1
        sentiment_data = {"label": sentiment_label, "score": sentiment_score}

        # --- Stage 2: Expert Review & Final Decision (Llama 3) ---
        emotions_data_for_llm = [{"label": e['label'], "score": e['score']} for e in candidate_emotions]
        refined_analysis = get_refined_analysis_with_groq(user_text, emotions_data_for_llm)

        raw_scores_map = {e['label']: e['score'] for e in candidate_emotions}
        
        final_emotions_list = []
        for refined_emotion in refined_analysis.get("emotions", []):
            label = refined_emotion.get("label")
            if label in raw_scores_map:
                final_emotions_list.append({
                    "label": label,
                    "score": raw_scores_map[label],
                    "explanation": refined_emotion.get("explanation"),
                    "emoji": EMOJI_MAP.get(label, '😐')
                })
        
        total_score = sum(e['score'] for e in final_emotions_list)
        if total_score > 0:
            for emotion in final_emotions_list:
                emotion['score'] = emotion['score'] / total_score
        elif final_emotions_list:
            equal_share = 1 / len(final_emotions_list)
            for emotion in final_emotions_list:
                emotion['score'] = equal_share

        # --- FIX: Re-sort the final list by score before sending it to the frontend ---
        final_emotions_list.sort(key=lambda x: x['score'], reverse=True)

        if not final_emotions_list:
            final_emotions_list = [{"label": "neutral", "score": 1, "explanation": "The text appears to be emotionally neutral.", "emoji": "😐"}]
            refined_analysis['summary'] = "No strong emotions were detected in the text."

        # Construct the final JSON
        final_analysis_payload = {
            "sentiment": sentiment_data,
            "summary": refined_analysis.get("summary", "Summary could not be generated."),
            "emotions": final_emotions_list
        }

        return jsonify(final_analysis_payload)

    except Exception as e:
        print(f"Error in final analysis endpoint: {e}")
        return jsonify({"error": "An internal error occurred during analysis."}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)