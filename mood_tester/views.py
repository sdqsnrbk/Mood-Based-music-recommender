# mood_tester/views.py
from django.shortcuts import render
from django.conf import settings
import os
import random
import pandas as pd  # Added for DataFrame operations
import torch  # Already imported but good to note for predict_mood

# --- Hugging Face Transformers (for mood detection) ---
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    HF_TRANSFORMERS_AVAILABLE = True
except ImportError:
    HF_TRANSFORMERS_AVAILABLE = False
    print("WARNING: Hugging Face Transformers library or PyTorch not installed. Mood detection will not work.")
    print("Please install them by running: pip install transformers torch pandas")  # Added pandas here

# --- Google Generative AI (for text generation with Gemini) ---
try:
    import google.generativeai as genai

    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False
    print("WARNING: Google Generative AI SDK not installed. Text generation with Gemini will not work.")
    print("Please install it by running: pip install google-generativeai")

# Mood Detection Model
MOOD_MODEL = None
MOOD_TOKENIZER = None

# Song Database
SONG_DATABASE_DF = None
# Define the path to your song database CSV.
# PLEASE UPDATE THIS PATH if your updated_db.csv is located elsewhere.
SONG_DATABASE_PATH = os.path.join(settings.BASE_DIR, 'mood_tester', 'data', 'updated_db.csv')
# If updated_db.csv is directly in your project root (dl_project folder):
# SONG_DATABASE_PATH = os.path.join(settings.BASE_DIR, 'updated_db.csv')


# Define the 28 emotion classes in the order your model outputs them
EMOTION_CLASSES = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity',
    'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
    'gratitude', 'grief', 'joy', 'love', 'nervousness', 'neutral', 'optimism', 'pride',
    'realization', 'relief', 'remorse', 'sadness', 'surprise'
]

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", getattr(settings, "GOOGLE_API_KEY", None))
GEMINI_MODEL = None
if GOOGLE_GENAI_AVAILABLE and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_MODEL = genai.GenerativeModel('gemini-1.5-flash-latest')
        print("Gemini API configured and model initialized.")
    except Exception as e:
        print(f"Error configuring Gemini API or initializing model: {e}")
        GEMINI_MODEL = None
        GOOGLE_GENAI_AVAILABLE = False
else:
    GEMINI_MODEL = None
    if GOOGLE_GENAI_AVAILABLE and not GEMINI_API_KEY:
        print("GOOGLE_API_KEY not found. Gemini text generation will not work.")
        GOOGLE_GENAI_AVAILABLE = False


def load_mood_model_and_tokenizer():
    global MOOD_MODEL, MOOD_TOKENIZER
    if not HF_TRANSFORMERS_AVAILABLE:
        print("Skipping mood model load: Hugging Face Transformers library not available.")
        return
    if MOOD_MODEL is None or MOOD_TOKENIZER is None:
        model_path = os.path.join(settings.BASE_DIR, 'bert(28 moods)')
        print(f"Attempting to load mood model from: {model_path}")
        try:
            if not os.path.exists(model_path) or not os.path.isdir(model_path):
                print(f"Error: Model path '{model_path}' does not exist or is not a directory.")
                return
            MOOD_TOKENIZER = AutoTokenizer.from_pretrained(model_path)
            MOOD_MODEL = AutoModelForSequenceClassification.from_pretrained(model_path)
            # Move model to GPU if available
            if torch.cuda.is_available():
                MOOD_MODEL.to('cuda')
            MOOD_MODEL.eval()
            print("Mood Model and Tokenizer loaded successfully.")
        except Exception as e:
            print(f"Error loading mood model/tokenizer from '{model_path}': {e}")
            MOOD_MODEL = None
            MOOD_TOKENIZER = None


def load_song_database():
    global SONG_DATABASE_DF
    if SONG_DATABASE_DF is None:
        try:
            print(f"Attempting to load song database from: {SONG_DATABASE_PATH}")
            SONG_DATABASE_DF = pd.read_csv(SONG_DATABASE_PATH)
            # Basic cleaning: ensure mood columns are lowercase to match EMOTION_CLASSES
            SONG_DATABASE_DF.columns = [col.lower() for col in SONG_DATABASE_DF.columns]
            # Verify that all emotion classes exist as columns in the song database
            missing_mood_cols = [mood for mood in EMOTION_CLASSES if mood not in SONG_DATABASE_DF.columns]
            if missing_mood_cols:
                print(f"WARNING: The following mood columns are missing in '{SONG_DATABASE_PATH}': {missing_mood_cols}")
                print(f"Available columns: {list(SONG_DATABASE_DF.columns)}")
            print(f"Song database loaded successfully. Shape: {SONG_DATABASE_DF.shape}")
        except FileNotFoundError:
            print(f"Error: Song database file not found at '{SONG_DATABASE_PATH}'. Please check the path.")
            SONG_DATABASE_DF = pd.DataFrame()  # Empty DataFrame to prevent errors later
        except Exception as e:
            print(f"Error loading song database: {e}")
            SONG_DATABASE_DF = pd.DataFrame()
    return SONG_DATABASE_DF


def predict_mood_probabilities(text):  # Renamed and modified
    if not HF_TRANSFORMERS_AVAILABLE or MOOD_MODEL is None or MOOD_TOKENIZER is None:
        print("Mood model not available for probability prediction.")
        # Return an empty DataFrame or a specific structure indicating failure
        return pd.DataFrame(columns=['id', 'label', 'prob'])

    try:
        MOOD_MODEL.eval()  # Ensure model is in eval mode
        with torch.no_grad():
            inputs = MOOD_TOKENIZER(text, return_tensors="pt", truncation=True, padding=True, max_length=128)  # Max length from your snippet
            # Move inputs to the same device as the model
            device = next(MOOD_MODEL.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = MOOD_MODEL(**inputs)
            logits = outputs.logits
            # Use sigmoid for multi-label probabilities
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        # Create DataFrame with probabilities
        results_df = pd.DataFrame({
            'id': list(range(len(EMOTION_CLASSES))),
            'prob': probs
        })
        # Add labels
        labels_df = pd.DataFrame({
            'id': list(range(len(EMOTION_CLASSES))),
            'label': EMOTION_CLASSES
        })
        results_df = results_df.merge(labels_df, on='id', how='left')
        results_df = results_df[['id', 'label', 'prob']]  # Ensure correct column order
        results_df = results_df.sort_values(by='prob', ascending=False).reset_index(drop=True)
        return results_df
    except Exception as e:
        print(f"Error during mood probability prediction: {e}")
        return pd.DataFrame(columns=['id', 'label', 'prob'])


def generate_introductory_text_with_gemini(mood_display_name, num_songs_found):
    if not GOOGLE_GENAI_AVAILABLE or GEMINI_MODEL is None:
        print("Gemini API not available. Using fallback message for song intro.")
        if num_songs_found > 0:
            return f"For your {mood_display_name} mood, here are some songs that might resonate!"
        else:
            return f"We looked for songs for your {mood_display_name} mood, but couldn't find a perfect match this time."

    prompt_base = f"You are a friendly and empathetic music recommender bot. A user is feeling {mood_display_name}."
    if num_songs_found > 0:
        prompt_detail = f"You have found {num_songs_found} {'song' if num_songs_found == 1 else 'songs'} for them."
        prompt_task = "Write a short, warm, and encouraging message (1-2 sentences, max 30 words) to introduce this song list."
    else:
        prompt_detail = "Unfortunately, you couldn't find any songs that perfectly matched this mood in the local collection."
        prompt_task = "Write a short, empathetic message (1-2 sentences, max 30 words) acknowledging this."

    prompt = (
        f"{prompt_base} {prompt_detail}\n"
        f"{prompt_task} Make the message directly address the user and be suitable for someone feeling {mood_display_name}."
    )
    print(f"Gemini Prompt for song intro: {prompt}")
    try:
        generation_config = genai.types.GenerationConfig(
            candidate_count=1, max_output_tokens=60, temperature=0.7, top_p=0.9, top_k=40
        )
        response = GEMINI_MODEL.generate_content(prompt, generation_config=generation_config)
        if response.candidates and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text.strip()
        else:
            print("Gemini API response for song intro was empty. Using fallback.")
            return f"Thinking about your {mood_display_name} mood. Here are some thoughts (and maybe songs if we found them)!"
    except Exception as e:
        print(f"Error during Gemini API call for song intro: {e}")
        return f"Thinking about your {mood_display_name} mood."


# --- Main View ---
def test_mood_view(request):
    if MOOD_MODEL is None or MOOD_TOKENIZER is None:
        load_mood_model_and_tokenizer()

    song_db = load_song_database()  # Load song database (cached)

    context = {
        'input_text': '',
        'predicted_mood_display': 'N/A',  # For showing the primary mood to the user
        'human_response': '',
        'recommended_songs': [],  # To store list of song dicts
        'error_message': ''
    }

    # Initial checks
    if not HF_TRANSFORMERS_AVAILABLE: context['error_message'] += " Mood detection library missing. "
    if not GOOGLE_GENAI_AVAILABLE: context['error_message'] += " Gemini text generation SDK missing. "
    if MOOD_MODEL is None or MOOD_TOKENIZER is None: context['error_message'] += " Mood model failed to load. "
    if GOOGLE_GENAI_AVAILABLE and GEMINI_MODEL is None: context['error_message'] += " Gemini model init failed. "
    if song_db.empty: context[
        'error_message'] += f" Song database ('{os.path.basename(SONG_DATABASE_PATH)}') is empty or failed to load. "

    if request.method == 'POST':
        input_text = request.POST.get('inputText', '').strip()
        context['input_text'] = input_text
        recommended_songs_list = []  # Initialize here

        if input_text and MOOD_MODEL and MOOD_TOKENIZER and not song_db.empty:
            mood_probabilities_df = predict_mood_probabilities(input_text)

            if not mood_probabilities_df.empty:
                # Display the top predicted mood
                top_mood_label = mood_probabilities_df.iloc[0]['label']
                context['predicted_mood_display'] = top_mood_label.replace("_", " ").capitalize()

                # Filter moods with probability > 0.5
                significant_moods_df = mood_probabilities_df[mood_probabilities_df['prob'] > 0.5]

                result_tracks_df = pd.DataFrame()  # Initialize empty DataFrame

                if not significant_moods_df.empty:
                    print(f"Significant moods (prob > 0.5): \n{significant_moods_df}")
                    for top_n in [3, 2, 1]:  # Try matching top 3, then top 2, then top 1 significant moods
                        # Get the labels of the top N significant moods
                        top_moods_to_match = significant_moods_df.head(top_n)['label'].tolist()

                        if not top_moods_to_match:  # Should not happen if significant_moods_df is not empty
                            continue

                        print(f"Attempting to match top {len(top_moods_to_match)} moods: {top_moods_to_match}")

                        filtered_songs = song_db.copy()  # Start with the full song database
                        possible_to_filter = True
                        for mood_label_to_match in top_moods_to_match:
                            if mood_label_to_match in filtered_songs.columns:
                                # Ensure the column is treated as numeric/boolean for filtering
                                try:
                                    filtered_songs[mood_label_to_match] = pd.to_numeric(
                                        filtered_songs[mood_label_to_match], errors='coerce').fillna(0).astype(int)
                                    filtered_songs = filtered_songs[filtered_songs[mood_label_to_match] == 1]
                                except Exception as e_filter:
                                    print(f"Error filtering by mood '{mood_label_to_match}': {e_filter}")
                                    possible_to_filter = False
                                    break
                            else:
                                print(
                                    f"Warning: Mood '{mood_label_to_match}' not found as a column in song_db. Skipping this mood for filtering.")
                                # If a critical mood for matching isn't in DB, this strategy might not work well for this top_n
                                # For simplicity, we continue, but this might lead to less specific matches if a mood is skipped.

                        if not possible_to_filter:
                            continue

                        if len(filtered_songs) >= 10:  # Aim for at least 10 songs (reduced from 20 for wider results)
                            result_tracks_df = filtered_songs
                            print(f"Found {len(result_tracks_df)} songs matching {len(top_moods_to_match)} moods.")
                            break  # Found enough songs

                    if result_tracks_df.empty and not significant_moods_df.empty:  # Fallback if multi-mood match yields nothing
                        print(
                            "Multi-mood match yielded no results or too few. Falling back to top 1 significant mood if available.")
                        # Try with just the single most significant mood if others failed
                        top_single_mood = significant_moods_df.iloc[0]['label']
                        if top_single_mood in song_db.columns:
                            try:
                                song_db[top_single_mood] = pd.to_numeric(song_db[top_single_mood],
                                                                         errors='coerce').fillna(0).astype(int)
                                result_tracks_df = song_db[song_db[top_single_mood] == 1]
                                print(
                                    f"Fallback: Found {len(result_tracks_df)} songs matching only the top mood '{top_single_mood}'.")
                            except Exception as e_single_filter:
                                print(f"Error in fallback filtering by mood '{top_single_mood}': {e_single_filter}")

                    # Sample songs from the final filtered DataFrame
                    if not result_tracks_df.empty:
                        num_to_sample = min(10, len(result_tracks_df))
                        # Sample up to 10 songs
                        sampled_tracks_df = result_tracks_df.sample(n=num_to_sample, replace=False)
                        sampled_tracks_df = sampled_tracks_df.sort_values(by=['popularity'], ascending=False)
                        # Convert DataFrame rows to list of dictionaries for the template
                        # Ensure you have 'track_name', 'artists', 'album_name' columns in your updated_db.csv
                        # Adjust these keys if your CSV column names are different
                        for index, row in sampled_tracks_df.iterrows():
                            recommended_songs_list.append({
                                'track_name': row.get('track_name'),  # Use .get for safety
                                'artists': row.get('artists'),
                                'album_name': row.get('album_name'),
                                'track_id': row.get('track_id')  # Get track_id, default to None if not found
                                # Add other song details you want to display
                            })
                        context['recommended_songs'] = recommended_songs_list
                        print(recommended_songs_list)
                    else:
                        print("No songs found after filtering.")

                else:  # No significant moods (all probs <= 0.5)
                    context['predicted_mood_display'] = "Mood is quite subtle or neutral."
                    print("No moods with probability > 0.5 found.")

                # Generate Gemini response based on the top mood and number of songs found
                # Use the display name for Gemini
                gemini_mood_input = context['predicted_mood_display'] if context[
                                                                             'predicted_mood_display'] != 'N/A' else "the current feeling"
                context['human_response'] = generate_introductory_text_with_gemini(gemini_mood_input,
                                                                                   len(recommended_songs_list))

            else:  # mood_probabilities_df is empty (error in prediction)
                context['predicted_mood_display'] = "Could not determine mood."
                context['human_response'] = "I'm having a little trouble understanding the mood right now."

        elif not input_text:
            context['predicted_mood_display'] = 'Please enter some text.'
            context['human_response'] = "Tell me how you're feeling to get started!"
        elif song_db.empty:
            context['human_response'] = "The song library isn't available right now, so I can't pick any tunes."


    else:  # Initial GET request
        context['human_response'] = "How are you feeling today? Tell me, and I might find some songs for you!"

    return render(request, 'mood_tester/test_page.html', context)
