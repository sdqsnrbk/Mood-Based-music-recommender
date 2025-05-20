import streamlit as st
import gdown  # For downloading from Google Drive
import os
import zipfile  # To extract the model if you zipped it
import pandas as pd  # For loading your track data (CSV)
import torch  # For PyTorch model (BERT)
import time  # For simulating delays or for actual model processing time

# --- Dependency Availability Flags ---
HF_TRANSFORMERS_AVAILABLE = False
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    HF_TRANSFORMERS_AVAILABLE = True
except ImportError:
    # Error will be shown in the app's main area
    pass

GOOGLE_GENAI_AVAILABLE = False
try:
    import google.generativeai as genai

    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    # Warning will be shown in the app's main area
    pass

# --- Configuration ---
# Model download and paths
MODEL_DOWNLOAD_DIR = "downloaded_model_cache"
EXTRACTED_MODEL_FOLDER_NAME = "bert(28 moods)"  # Ensure this EXACTLY matches the folder name inside your zip
PATH_TO_EXTRACTED_MODEL = os.path.join(MODEL_DOWNLOAD_DIR, EXTRACTED_MODEL_FOLDER_NAME)

# Track data path (relative to this app.py file)
# Assuming app.py is at the root, and your data is in 'mood_tester/data/'
DATA_FILE_PATH = os.path.join("mood_tester", "data", "updated_db.csv")

# Emotion classes (ensure these match your BERT model's output and your CSV columns)
EMOTION_CLASSES = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity',
    'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
    'gratitude', 'grief', 'joy', 'love', 'nervousness', 'neutral', 'optimism', 'pride',
    'realization', 'relief', 'remorse', 'sadness', 'surprise'
]


# --- Embedded CSS for styling ---
def local_css():
    st.markdown("""
        <style>
            /* General body style - Streamlit applies its own theme, so be careful with broad changes */
            /* .stApp { background-color: #your_choice; } */

            h1#mood-music-recommender {
                color: #2c3e50; /* Dark blue-grey */
                text-align: center;
                margin-bottom: 25px;
                font-weight: bold; /* Make title bolder */
            }

            .stTextArea label { /* Target Streamlit's text_area label */
                display:block;
                margin-bottom:8px;
                font-weight:bold;
                font-size: 1.1em;
                color: #333; /* Darker label color */
            }

            /* Custom styled boxes */
            .custom-box {
                padding: 15px;
                border-radius: 8px; /* Softer corners */
                margin-bottom: 15px;
                font-size: 16px; /* Consistent font size */
                line-height: 1.6;
                border-left-width: 5px;
                border-left-style: solid;
            }

            .human-response-box { /* For Gemini's message */
                background-color: #e8f4fd; /* Light blue */
                border-left-color: #3498db; /* Blue accent */
                color: #2980b9; /* Darker blue text */
            }

            .detected-mood-box { /* For primary mood */
                background-color: #f0f0f0; /* Light grey */
                border-left-color: #555; /* Dark grey accent */
                color: #2c3e50; /* Dark text */
            }
            .detected-mood-box strong { 
                color: #2c3e50; 
                font-weight: 600; /* Slightly bolder */
            }

            /* Styling for the expander (significant moods) */
            .stExpander {
                border: 1px solid #f39c12 !important; /* Orange border */
                border-left: 5px solid #f39c12 !important; /* Orange accent */
                background-color: #fef9e7 !important; /* Very light orange/yellow */
                border-radius: 8px !important; /* Softer corners */
                margin-bottom: 15px !important;
            }
            .stExpander header button { /* Expander header */
                color: #d35400 !important; /* Darker orange for title */
                font-weight: bold !important;
                font-size: 1.05em !important; /* Slightly larger header */
            }
            .stExpander [data-testid="stExpanderDetails"] p { /* Content within expander */
                 margin: 8px 0 8px 5px; /* Adjust spacing */
                 color: #555; /* Softer text color */
            }

            /* Song recommendations area */
            .song-recommendations-title { /* For the "Your Personalized Playlist:" subheader */
                margin-top: 25px;
                margin-bottom: 10px;
                color: #2c3e50;
                font-weight: bold;
            }
            /* Streamlit's dataframe styling is mostly internal. Can target cells if desperate with complex CSS */
            /* .stDataFrame div[role="gridcell"] { font-size: 14px; } */

            .spotify-player-container {
                margin-top: 20px;
                margin-bottom: 20px;
                width: 100%;
                max-width: 480px; /* Slightly wider for better fit */
                margin-left: auto;
                margin-right: auto;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1); /* Add subtle shadow */
                border-radius: 12px; /* Match iframe's expected radius */
            }
        </style>
    """, unsafe_allow_html=True)


# --- BERT Model Functions ---
def actual_bert_model_loader(model_path):
    if not HF_TRANSFORMERS_AVAILABLE:
        st.error(
            "Cannot load BERT model: Transformers library is missing. Please add 'transformers' and 'torch' to your requirements.txt and restart.")
        return None, None
    try:
        if not os.path.exists(model_path) or not os.path.isdir(model_path):
            st.error(
                f"BERT Model path '{model_path}' does not exist or is not a directory. Check EXTRACTED_MODEL_FOLDER_NAME.")
            return None, None

        st.write(f"Attempting to load tokenizer from: {model_path}")  # Use st.write for user-facing messages
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        st.write(f"Attempting to load model from: {model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

        model.eval()
        st.success("BERT Model and Tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading BERT model/tokenizer from '{model_path}': {e}")
        return None, None


@st.cache_resource  # Cache the loaded model and tokenizer
def get_bert_model_and_tokenizer():
    """Downloads (if needed), extracts, and loads the BERT model."""
    # IMPORTANT: Ensure 'google_drive_model_zip_url' is set in your Streamlit secrets!
    # Create a file .streamlit/secrets.toml and add:
    # google_drive_model_zip_url = "YOUR_GOOGLE_DRIVE_FILE_ID_OR_SHAREABLE_LINK"
    model_zip_url_or_id = st.secrets.get("google_drive_model_zip_url")
    if not model_zip_url_or_id:
        st.error("Model URL/ID ('google_drive_model_zip_url') not found in Streamlit secrets! Cannot download model.")
        st.info("Please create a file named .streamlit/secrets.toml in your app's root directory and add the key.")
        return None, None

    if not os.path.exists(PATH_TO_EXTRACTED_MODEL):
        os.makedirs(MODEL_DOWNLOAD_DIR, exist_ok=True)
        local_zip_path = os.path.join(MODEL_DOWNLOAD_DIR, "bert_model_archive.zip")

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.info(f"Downloading BERT model from Google Drive... (this can take a few minutes)")
            print(f"Starting download from GDrive to {local_zip_path}")  # Keep console log for debugging

            # gdown needs a direct download link or file ID.
            # If it's a shareable link, gdown tries to resolve it.
            gdown.download(id=model_zip_url_or_id, output=local_zip_path, quiet=False,
                           fuzzy=True)  # Use id for clarity if it's an ID

            progress_bar.progress(50)  # Update progress
            print("Download finished.")
            status_text.info(f"Extracting BERT model to '{MODEL_DOWNLOAD_DIR}'...")
            print(f"Extracting {local_zip_path} to {MODEL_DOWNLOAD_DIR}")
            with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
                zip_ref.extractall(MODEL_DOWNLOAD_DIR)
            progress_bar.progress(90)  # Update progress
            print("Extraction finished.")

            if os.path.exists(local_zip_path):  # Clean up zip file
                os.remove(local_zip_path)

            if not os.path.exists(PATH_TO_EXTRACTED_MODEL):
                st.error(f"CRITICAL: Extraction did not create the expected folder: {PATH_TO_EXTRACTED_MODEL}. "
                         f"Ensure your zip file creates a top-level folder named '{EXTRACTED_MODEL_FOLDER_NAME}'.")
                st.info(
                    f"Contents of {MODEL_DOWNLOAD_DIR} after extraction: {os.listdir(MODEL_DOWNLOAD_DIR) if os.path.exists(MODEL_DOWNLOAD_DIR) else 'Not found'}")
                status_text.empty()
                progress_bar.empty()
                return None, None

            progress_bar.progress(100)
            status_text.success("BERT model downloaded and extracted!")
            time.sleep(2)  # Give time for user to see success message
            status_text.empty()
            progress_bar.empty()

        except Exception as e:
            st.error(f"Error during BERT model download or extraction: {e}")
            status_text.empty()
            progress_bar.empty()
            return None, None
    else:
        st.info(f"BERT model folder found in local cache: {PATH_TO_EXTRACTED_MODEL}")

    return actual_bert_model_loader(PATH_TO_EXTRACTED_MODEL)


def predict_emotions_with_bert(text_input, model, tokenizer):
    if not HF_TRANSFORMERS_AVAILABLE or model is None or tokenizer is None:
        st.warning("BERT model not available for prediction.")
        return {}
    try:
        model.eval()  # Ensure model is in evaluation mode
        with torch.no_grad():  # Disable gradient calculations for inference
            inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True, max_length=128)

            # Move inputs to the same device as the model
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            logits = outputs.logits
            # Sigmoid for multi-label classification, then convert to numpy
            probabilities = torch.sigmoid(logits).cpu().numpy()[0]

            # Create a dictionary of emotion scores
        emotion_scores = {EMOTION_CLASSES[i]: float(probabilities[i]) for i in range(len(EMOTION_CLASSES))}
        # Sort by score in descending order
        return dict(sorted(emotion_scores.items(), key=lambda item: item[1], reverse=True))
    except Exception as e:
        st.error(f"Error during mood prediction with BERT: {e}")
        return {}


# --- Gemini API Function ---
@st.cache_data(ttl=3600)  # Cache Gemini responses for an hour
def generate_gemini_description(primary_mood, num_songs):
    if not GOOGLE_GENAI_AVAILABLE:
        st.info("Google Generative AI SDK not available. Using a fallback description.")
        return f"Hey there! Here's a playlist with {num_songs} {'song' if num_songs == 1 else 'songs'} for your {primary_mood} mood. Enjoy!"

    # IMPORTANT: Ensure 'gemini_api_key' is set in your Streamlit secrets!
    # In .streamlit/secrets.toml:
    # gemini_api_key = "YOUR_GEMINI_API_KEY"
    gemini_api_key = st.secrets.get("gemini_api_key")
    if not gemini_api_key:
        st.warning("Gemini API key ('gemini_api_key') not found in secrets. Using a fallback description.")
        st.info("To enable personalized messages, add your Gemini API key to .streamlit/secrets.toml.")
        return f"Hey there! Here's a playlist with {num_songs} {'song' if num_songs == 1 else 'songs'} for your {primary_mood} mood. Enjoy!"

    try:
        genai.configure(api_key=gemini_api_key)
        gemini_model_instance = genai.GenerativeModel('gemini-1.5-flash-latest')  # Or your preferred model

        prompt_base = f"You are a friendly and empathetic music recommender bot. A user is feeling {primary_mood}."
        if num_songs > 0:
            prompt_detail = f"You have found {num_songs} {'song' if num_songs == 1 else 'songs'} for them."
            prompt_task = "Write a short, warm, and encouraging message (1-2 sentences, max 30-40 words) to introduce this song list."
        else:
            prompt_detail = "Unfortunately, you couldn't find any songs that perfectly matched this mood in the local collection."
            prompt_task = "Write a short, empathetic message (1-2 sentences, max 30-40 words) acknowledging this."

        prompt = (
            f"{prompt_base} {prompt_detail}\n"
            f"{prompt_task} Make the message directly address the user and be suitable for someone feeling {primary_mood}."
        )
        print(f"Gemini Prompt: {prompt}")  # For server-side debugging

        generation_config = genai.types.GenerationConfig(  # Ensure this matches the SDK version you use
            candidate_count=1, max_output_tokens=60, temperature=0.7, top_p=0.9, top_k=40
        )
        response = gemini_model_instance.generate_content(prompt, generation_config=generation_config)

        # Handle response structure based on google-generativeai library
        if response.candidates and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text.strip()
        elif hasattr(response, 'text') and response.text:  # Older versions or simpler responses
            return response.text.strip()
        else:
            print(f"Gemini API response was empty or in unexpected format: {response}")
            return f"Thinking about your {primary_mood} mood. Here are some thoughts and {num_songs} {'song' if num_songs == 1 else 'songs'}!"
    except Exception as e:
        st.error(f"Error calling Gemini API: {e}")
        return f"Thinking about your {primary_mood} mood, and found {num_songs} {'song' if num_songs == 1 else 'songs'}."


# --- Music Recommendation Logic ---
def recommend_songs(detected_emotions_scores, track_df, num_to_recommend=10):
    # Ensure EMOTION_CLASSES is accessible globally.
    # If it's not defined in the scope where this app runs, this will cause a NameError.
    # This change assumes EMOTION_CLASSES is available globally as it was in the full script.
    global EMOTION_CLASSES
    if 'EMOTION_CLASSES' not in globals() or not EMOTION_CLASSES:
        st.error(
            "CRITICAL: EMOTION_CLASSES global variable is not defined or empty. Cannot proceed with recommendations.")
        return pd.DataFrame(), "Configuration Error - EMOTION_CLASSES missing"

    if track_df is None or track_df.empty:
        st.warning("Track database is empty. Cannot recommend songs.")
        return pd.DataFrame(), "No significant moods detected or database empty."

    # Filter for moods with score > 0.5 and sort them
    significant_moods = {
        mood: score for mood, score in detected_emotions_scores.items() if score > 0.6 and mood in EMOTION_CLASSES
    }
    # Sort significant moods by score, highest first
    sorted_significant_moods = dict(sorted(significant_moods.items(), key=lambda item: item[1], reverse=True))

    primary_mood_for_filtering = list(sorted_significant_moods.keys())[0] if sorted_significant_moods else None

    # If no significant moods are detected at all, fall back to popular tracks
    if not sorted_significant_moods:
        st.info("Moods are quite subtle or neutral. Showing some popular tracks from the database.")
        if 'popularity' in track_df.columns:
            return track_df.sort_values(by=['popularity'], ascending=False).head(num_to_recommend), "Neutral/Subtle"
        else:
            return track_df.head(num_to_recommend), "Neutral/Subtle (no popularity)"

    print(f"Significant moods (prob > 0.5) for recommendation: {sorted_significant_moods}")

    best_overall_match_df = pd.DataFrame()

    # Try matching multiple top moods, then fewer, down to one.
    # The goal is to find the most specific match that still yields a reasonable number of songs.
    for n_moods_to_match in range(min(3, len(sorted_significant_moods)), 0, -1):
        top_moods_labels = list(sorted_significant_moods.keys())[:n_moods_to_match]
        print(f"Attempting to match top {len(top_moods_labels)} moods: {top_moods_labels}")

        current_filter_df = track_df.copy()
        possible_to_filter_this_iteration = True
        for mood_label in top_moods_labels:
            if mood_label in current_filter_df.columns:
                try:
                    current_filter_df[mood_label] = pd.to_numeric(current_filter_df[mood_label],
                                                                  errors='coerce').fillna(0).astype(int)
                    current_filter_df = current_filter_df[current_filter_df[mood_label] == 1]
                    if current_filter_df.empty:  # If any mood filter in the combination results in no songs, this combination is not viable
                        possible_to_filter_this_iteration = False
                        break
                except Exception as e_filter:
                    print(f"Error filtering by mood '{mood_label}': {e_filter}")
                    possible_to_filter_this_iteration = False
                    break
            else:
                print(
                    f"Warning: Mood column '{mood_label}' not found in song_db. Skipping this mood for this {len(top_moods_labels)}-moods combination.")
                # If a mood column is missing, this specific combination might be invalid
                possible_to_filter_this_iteration = False
                break

        if not possible_to_filter_this_iteration or current_filter_df.empty:
            # If this combination of moods didn't work or resulted in an empty df,
            # continue to the next iteration (matching fewer moods).
            print(
                f"No songs found for {len(top_moods_labels)}-moods combination: {top_moods_labels}. Trying fewer moods.")
            continue

        # If this iteration found songs:
        print(f"Found {len(current_filter_df)} songs matching {len(top_moods_labels)} moods: {top_moods_labels}.")

        # We want the most specific match (more moods) that gives us enough songs.
        # If this current match (e.g. 3 moods) gives songs, it's better than a 2-mood match found later.
        # So, we take the first successful match from most specific to least specific.
        best_overall_match_df = current_filter_df
        st.info(f"Prioritizing match with {len(top_moods_labels)} mood(s) yielding {len(best_overall_match_df)} songs.")
        break  # Found the most specific match possible, so exit the loop.

    # --- Decision logic after attempting all levels of multi-mood matches ---

    # Case 1: A specific mood match (single or multi) was found
    if not best_overall_match_df.empty:
        st.info(f"Recommending based on best specific mood match which found {len(best_overall_match_df)} track(s).")
        return best_overall_match_df.sort_values(by=['popularity'], ascending=False).head(num_to_recommend).drop_duplicates(), primary_mood_for_filtering

    # Case 2: No specific mood combination (even single primary mood) yielded any songs.
    # This implies primary_mood_for_filtering (if it existed) didn't match any songs when n_moods_to_match was 1.
    st.info("No specific mood matches found after trying various combinations. Showing some generally popular tracks.")
    if 'popularity' in track_df.columns:
        return track_df.sort_values(by=['popularity'], ascending=False).head(num_to_recommend), "Popular Fallback"
    else:
        return track_df.head(num_to_recommend), "Popular Fallback (no popularity)"


# --- Track Data Loading Function ---
@st.cache_data  # Cache the loaded song data
def load_song_data(file_path):
    absolute_file_path = os.path.abspath(file_path)  # Get absolute path for clarity in error messages
    if not os.path.exists(file_path):
        st.error(f"Song database file not found at relative path: '{file_path}'")
        st.error(f"Attempted absolute path: '{absolute_file_path}'")
        st.error(f"Current working directory: {os.getcwd()}")
        st.info("Please ensure DATA_FILE_PATH is correct and the CSV file exists in that location.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_path)
        # Normalize column names: lowercase and strip whitespace
        df.columns = [str(col).strip().lower() for col in df.columns]

        # Ensure 'popularity' column exists and is numeric, otherwise add a dummy one
        if 'popularity' in df.columns:
            df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce').fillna(0)
        else:
            st.warning(
                "'popularity' column not found in track data. Adding a dummy 'popularity' column with 0s. Sorting by popularity will not be effective.")
            df['popularity'] = 0  # Add dummy popularity

        # Check for missing mood columns based on EMOTION_CLASSES
        # These columns are expected to be 0/1 flags for filtering
        missing_mood_cols = [mood for mood in EMOTION_CLASSES if mood not in df.columns]
        if missing_mood_cols:
            st.warning(
                f"The following mood columns are missing in '{os.path.basename(file_path)}' and are needed for filtering: {missing_mood_cols}. "
                "Mood-based song filtering might be severely affected or not work for these moods."
            )
        st.success(f"Song database loaded ('{os.path.basename(file_path)}', {len(df)} rows).")
        return df
    except pd.errors.EmptyDataError:
        st.error(f"The song database file at '{file_path}' is empty.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading song database from '{file_path}': {e}")
        return pd.DataFrame()


# --- Streamlit App Main UI ---
st.set_page_config(page_title="Mood Music Recommender", layout="centered", initial_sidebar_state="auto")
local_css()  # Apply custom CSS

# Initial checks for critical dependencies - display these prominently
if not HF_TRANSFORMERS_AVAILABLE:
    st.error(
        "CRITICAL: Hugging Face Transformers library not installed. Mood detection will NOT work. Please add 'transformers' and 'torch' to your requirements.txt and reinstall.")
if not GOOGLE_GENAI_AVAILABLE:
    st.warning(
        "INFO: Google Generative AI SDK not installed. Playlist descriptions will use a fallback. Add 'google-generativeai' to requirements.txt for this feature.")

st.markdown("<h1 id='mood-music-recommender'>✨ Mood Music Recommender ✨</h1>", unsafe_allow_html=True)

# Load resources (model and song data) - these will be cached
# Display loading messages for resources
with st.spinner("Initializing AI model and song database... This may take a moment on first run."):
    bert_model, bert_tokenizer = get_bert_model_and_tokenizer() if HF_TRANSFORMERS_AVAILABLE else (None, None)
    song_database_df = load_song_data(DATA_FILE_PATH)

# User input section
st.markdown(
    "<label for='mood_input_area' style='display:block; margin-bottom:8px; font-weight:bold; font-size: 1.1em;'>How are you feeling today?</label>",
    unsafe_allow_html=True)
user_text = st.text_area("", placeholder="e.g., I am feeling ecstatic and ready to party!", height=100, key="mood_input_area", label_visibility="collapsed")

if st.button("Get Recommendations 🚀", type="primary", use_container_width=True):
    if not user_text.strip():  # Check if input is empty or just whitespace
        st.warning("Please tell me how you're feeling!")
    elif HF_TRANSFORMERS_AVAILABLE and (bert_model is None or bert_tokenizer is None):
        st.error(
            "BERT Model not loaded. Cannot get recommendations. Check error messages above and ensure Transformers library is installed and model downloaded.")
    elif song_database_df.empty:
        st.error(
            "Song database not loaded or empty. Cannot get recommendations. Check data file path and error messages above.")
    else:
        with st.spinner("Analyzing your mood and finding songs... 🎶"):
            emotion_scores = predict_emotions_with_bert(user_text, bert_model, bert_tokenizer)

            if not emotion_scores:  # Check if dictionary is empty (error during prediction)
                st.error(
                    "Could not determine emotions from your input. Please try different phrasing or check model logs.")
            else:
                # Determine primary mood from the sorted scores
                # The recommend_songs function will also determine the mood used for filtering
                recommended_songs_df, mood_for_gemini = recommend_songs(emotion_scores, song_database_df)

                # If recommend_songs returns a mood_for_gemini, use it. Otherwise, fallback to top score.
                if not mood_for_gemini or mood_for_gemini == "Popular Fallback (no popularity)" or mood_for_gemini == "Neutral/Subtle (no popularity)":
                    # Fallback if recommend_songs didn't have a clear mood due to fallbacks
                    primary_mood_display = list(emotion_scores.keys())[0] if emotion_scores else "Unknown"
                else:
                    primary_mood_display = mood_for_gemini

                intro_message = generate_gemini_description(primary_mood_display.capitalize(),
                                                            len(recommended_songs_df))

                # Display results
                st.markdown(f"<div class='custom-box human-response-box'>{intro_message}</div>", unsafe_allow_html=True)

                st.markdown(f"""
                <div class='custom-box detected-mood-box'>
                    <p><strong>Detected Primary Mood for Playlist:</strong> {primary_mood_display.capitalize()}</p>
                </div>
                """, unsafe_allow_html=True)

                with st.expander("All Significant Detected Moods (Score > 0.5):", expanded=True):
                    significant_found = False
                    # Iterate through original emotion_scores to show all significant ones
                    for mood, score in emotion_scores.items():
                        if score > 0.7 and mood in EMOTION_CLASSES:  # Ensure mood is valid
                            st.markdown(f"<p>{mood.capitalize()}: ({score:.2f})</p>", unsafe_allow_html=True)
                            significant_found = True
                    if not significant_found:
                        st.markdown(
                            "<p>No specific strong moods detected (above 0.7 threshold). The primary mood or popular tracks were used.</p>",
                            unsafe_allow_html=True)

                st.markdown("<h3 class='song-recommendations-title'>Your Personalized Playlist:</h3>",
                            unsafe_allow_html=True)

                if not recommended_songs_df.empty:

                    display_cols = ['track_name', 'artists', 'album_name', 'popularity']
                    actual_display_cols = [col for col in display_cols if col in recommended_songs_df.columns]
                    st.dataframe(
                        recommended_songs_df[actual_display_cols] if actual_display_cols else recommended_songs_df,
                        use_container_width=True)

                    # --- MODIFIED SPOTIFY PLAYER SECTION TO LOOP THROUGH SONGS ---
                    st.markdown("---")
                    st.markdown("<h4>🎧 Play Recommended Tracks:</h4>", unsafe_allow_html=True)

                    player_displayed_count = 0
                    # Loop through the recommended_songs_df (which now contains up to 10 songs)
                    for index, row in recommended_songs_df.iterrows():
                        # Check if 'track_id' column exists and the value is not NaN
                        if 'track_id' in row and pd.notna(row['track_id']):
                            track_id = row['track_id']
                            track_name = row.get('track_name', 'Unknown Track')
                            artists = row.get('artists', 'Unknown Artist')

                            # Using the Spotify embed URL format from your original code.
                            # Note: Standard Spotify embed URL is typically: f"https://open.spotify.com/embed/track/{track_id}"
                            spotify_embed_url = f"https://open.spotify.com/embed/track/{track_id}"



                            # Embed Spotify player
                            st.markdown(f"""
                                            <div class='spotify-player-container'>
                                                <iframe title="Spotify Web Player for {track_name}"
                                                        style="border-radius:12px; width:100%;"
                                                        src="{spotify_embed_url}"
                                                        height="80" 
                                                        frameBorder="0" 
                                                        allowfullscreen="" 
                                                        allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" 
                                                        loading="lazy">
                                                </iframe>
                                            </div>
                                            """, unsafe_allow_html=True)
                            player_displayed_count += 1
                        # else:
                        #     # Optional: Log or inform if a track_id is missing for a recommended song
                        #     print(f"Skipping player for a song due to missing track_id: {row.get('track_name', 'N/A')}")

                    if player_displayed_count == 0:
                        st.info(
                            "Could not find 'track_id' for any of the top recommended songs to embed Spotify players.")
                    # --- END OF MODIFIED SPOTIFY PLAYER SECTION ---
                else:
                    st.info("No tracks found matching your current mood based on the available data.")

else:
    # Initial state message if resources are still loading or failed
    if (HF_TRANSFORMERS_AVAILABLE and (bert_model is None or bert_tokenizer is None) and not st.session_state.get(
            "get_recs_clicked", False)) \
            or (song_database_df.empty and not st.session_state.get("get_recs_clicked", False)):
        st.info("App is ready. Enter how you're feeling and click 'Get Recommendations'.")

# Add a session state to track if button was clicked to avoid showing initial loading error after interaction
if "get_recs_clicked" not in st.session_state:
    st.session_state.get_recs_clicked = False
if st.button:  # if any button is clicked, set it to true
    st.session_state.get_recs_clicked = True

# --- Sidebar ---
st.sidebar.header("About")
st.sidebar.info(
    "This app uses AI to understand your mood from your text input "
    "and recommends songs from a predefined dataset. It features a BERT-based model "
    "for mood analysis and can use Google's Gemini API for generating "
    "creative playlist descriptions."
)
st.sidebar.markdown("---")
st.sidebar.subheader("How it Works:")
st.sidebar.markdown("""
1.  **Enter Your Mood:** Type how you're feeling into the text box.
2.  **AI Analysis:** A BERT model analyzes your text to identify up to 28 different emotions.
3.  **Song Matching:** The app searches its music database for songs that match your detected primary and significant moods.
4.  **Playlist & Message:** You get a list of recommended songs and a personalized message (if Gemini API is configured).
""")
st.sidebar.markdown("---")
st.sidebar.subheader("Technical Details:")
st.sidebar.markdown(f"""
-   **Mood Model:** BERT ({EXTRACTED_MODEL_FOLDER_NAME})
-   **Song Data:** `updated_db.csv` ({len(song_database_df) if not song_database_df.empty else 'N/A'} tracks)
-   **Transformers Lib:** {'✅ Available' if HF_TRANSFORMERS_AVAILABLE else '❌ Not Installed'}
-   **Gemini API Lib:** {'✅ Available' if GOOGLE_GENAI_AVAILABLE else '⚠️ Not Installed (using fallback)'}
""")
if not os.path.exists(".streamlit/secrets.toml"):
    st.sidebar.warning(
        "Remember to create a `.streamlit/secrets.toml` file for API keys (Gemini, Google Drive model URL). See comments in the code.")
