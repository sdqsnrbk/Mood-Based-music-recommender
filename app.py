import streamlit as st
import gdown  # For downloading from Google Drive
import os
import zipfile  # To extract the model if you zipped it
import pandas as pd  # For loading your track data (CSV)
import time # To simulate model processing if needed

# --- Configuration for Model ---
# This is where your Streamlit app will store the downloaded and extracted model
MODEL_DOWNLOAD_DIR = "downloaded_model_cache"
# This should be the exact name of the folder that is created when you unzip your model.
# Based on your Google Drive screenshot (image_57a3e4.png), the folder is "bert (28 moods)"
EXTRACTED_MODEL_FOLDER_NAME = "bert(28 moods)"
PATH_TO_EXTRACTED_MODEL = os.path.join(MODEL_DOWNLOAD_DIR, EXTRACTED_MODEL_FOLDER_NAME)

# --- Configuration for Track Data ---
# Based on your PyCharm project structure (e.g., image_57c14f.png),
# if app.py is in 'dl_project - Copy', and your data is in 'dl_project - Copy/mood_tester/data/updated_db.csv'
# then the path would be:
# DATA_FILE_PATH = os.path.join("mood_tester", "data", "updated_db.csv")
# Or if it's in 'dl_project - Copy/dl_project/mood_tester/data/updated_db.csv':
# DATA_FILE_PATH = os.path.join("dl_project", "mood_tester", "data", "updated_db.csv")
# For simplicity, let's assume your 'data' folder (containing updated_db.csv) is at the same level as app.py
# C:\Users\sdqsn\Desktop\dl_project - Copy\data\updated_db.csv
# PLEASE ADJUST THIS PATH TO MATCH YOUR ACTUAL CSV FILE LOCATION:
DATA_FILE_PATH = os.path.join("mood_tester", "data", "updated_db.csv")

# --- Placeholder for your actual model loading function ---
# You will need to replace this with the code that loads your BERT model
# using PyTorch/TensorFlow and Hugging Face Transformers, or however it's built.
def your_actual_bert_model_loader_function(model_path):
    """
    Placeholder function to load your BERT model.
    Replace this with your actual model loading code.
    Example:
    from transformers import BertForSequenceClassification, BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    return model, tokenizer
    """
    st.write(f"Attempting to load model from: {model_path}")
    # Simulate model loading
    time.sleep(2)
    # Check if the path actually exists (it should if download/extraction worked)
    if os.path.exists(model_path) and os.path.isdir(model_path):
        # In a real scenario, you'd return the loaded model object
        # For now, just return a success message or the path
        return f"Successfully 'loaded' model from {model_path}"
    else:
        st.error(f"Model directory not found at {model_path} after download/extraction.")
        return None
# --- End of placeholder ---


# Function to download and prepare the model
@st.cache_resource  # Cache the resource (model)
def get_model():
    """
    Downloads the model from Google Drive (if not already present),
    extracts it, and then loads it using your specific loading function.
    """
    model_zip_url_from_secrets = st.secrets.get("google_drive_model_zip_url")

    if not model_zip_url_from_secrets:
        st.error("Model URL ('google_drive_model_zip_url') not found in Streamlit secrets!")
        st.info("Please add it to your .streamlit/secrets.toml locally, or in the Streamlit Cloud deployment settings.")
        return None

    # Check if the *extracted model folder* already exists
    if not os.path.exists(PATH_TO_EXTRACTED_MODEL):
        os.makedirs(MODEL_DOWNLOAD_DIR, exist_ok=True)  # Create download directory if it doesn't exist
        local_zip_path = os.path.join(MODEL_DOWNLOAD_DIR, "model_archive.zip") # Consistent name for the zip

        try:
            # Download the zip file
            with st.spinner(f"Downloading model from Google Drive... (this can take a few minutes for a large model)"):
                print(f"Starting download from: {model_zip_url_from_secrets} to {local_zip_path}")
                gdown.download(model_zip_url_from_secrets, local_zip_path, quiet=False, fuzzy=True)
                print("Download finished.")

            # Extract the zip file
            with st.spinner(f"Extracting model to '{MODEL_DOWNLOAD_DIR}'..."):
                print(f"Extracting {local_zip_path} to {MODEL_DOWNLOAD_DIR}")
                with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(MODEL_DOWNLOAD_DIR)
                print("Extraction finished.")

            os.remove(local_zip_path)  # Clean up the downloaded zip file
            st.success("Model downloaded and extracted successfully!")

            # Verify that the expected model folder was created by the extraction
            if not os.path.exists(PATH_TO_EXTRACTED_MODEL):
                st.error(f"Extraction did not create the expected folder: {PATH_TO_EXTRACTED_MODEL}")
                st.error(f"Please ensure your zip file contains a top-level folder named '{EXTRACTED_MODEL_FOLDER_NAME}'.")
                st.error(f"Contents of {MODEL_DOWNLOAD_DIR} after extraction: {os.listdir(MODEL_DOWNLOAD_DIR)}")
                return None

        except Exception as e:
            st.error(f"Error during model download or extraction: {e}")
            # Clean up partially downloaded/extracted files if an error occurs
            if os.path.exists(local_zip_path):
                os.remove(local_zip_path)
            if os.path.exists(PATH_TO_EXTRACTED_MODEL) and EXTRACTED_MODEL_FOLDER_NAME in PATH_TO_EXTRACTED_MODEL : # Be careful with recursive delete
                # import shutil # Use with caution
                # shutil.rmtree(PATH_TO_EXTRACTED_MODEL) # Remove extracted folder if an error occurred
                st.warning(f"Consider manually cleaning up {PATH_TO_EXTRACTED_MODEL} if it's corrupted.")
            return None
    else:
        st.info(f"Model folder '{EXTRACTED_MODEL_FOLDER_NAME}' found in local cache: {PATH_TO_EXTRACTED_MODEL}")

    # --- Load your model using your specific function ---
    loaded_model = your_actual_bert_model_loader_function(PATH_TO_EXTRACTED_MODEL)
    return loaded_model


# Function to load your track data
@st.cache_data  # Cache the loaded DataFrame
def load_track_data(file_path):
    """Loads the track data from the specified CSV file."""
    if not os.path.exists(file_path):
        st.error(f"Track data file not found at '{file_path}'.")
        st.info(f"Please ensure the file exists. Current working directory: {os.getcwd()}")
        st.info(f"Contents of current directory: {os.listdir('.')}")
        # Check one level up if common project structure is used
        if os.path.exists(os.path.join("..", file_path)):
             st.warning(f"File found at '../{file_path}'. Consider adjusting DATA_FILE_PATH.")
        return None
    try:
        df = pd.read_csv(file_path)
        st.success(f"Track data loaded successfully from '{file_path}' ({len(df)} rows).")
        return df
    except pd.errors.EmptyDataError:
        st.error(f"The track data file at '{file_path}' is empty.")
        return None
    except Exception as e:
        st.error(f"Error loading track data from '{file_path}': {e}")
        return None

# --- Main Streamlit App ---
st.set_page_config(page_title="Music Recommender", layout="wide", initial_sidebar_state="expanded")
st.title("🎵 Mood-Based Music Recommender 🎵")

# Sidebar for controls or information
st.sidebar.header("Controls & Info")
st.sidebar.info(
    "This app recommends music based on your selected mood. "
    "The model will be downloaded on the first run."
)

# Load model and data
# These functions will only run fully the first time or if the cache is cleared/secrets change.
# Subsequent runs will use the cached versions.
model_object = get_model()  # This will be your actual loaded model object or a placeholder
track_df = load_track_data(DATA_FILE_PATH)

# Main app content area
if model_object and track_df is not None and not track_df.empty:
    st.header("How are you feeling today?")

    # Example input: A select box for mood
    # You'll likely want to customize these moods or use a text input
    # if your model processes natural language.
    # Based on image_57ba0c.png, you have 28 emotion classes.
    # You should list them here or get them from your model/config if possible.
    EMOTION_CLASSES = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
        'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
        'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness',
        'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise',
        'neutral'
    ] # Example list, ensure it matches your model's classes

    user_mood_input = st.selectbox(
        "Select your current mood:",
        options=EMOTION_CLASSES,
        index=EMOTION_CLASSES.index('neutral') if 'neutral' in EMOTION_CLASSES else 0 # Default to neutral or first
    )

    if st.button("Get Music Recommendations 🎶", type="primary"):
        if user_mood_input:
            st.subheader(f"Recommendations for when you're feeling: {user_mood_input.capitalize()}")

            # --- !!! IMPORTANT !!! ---
            # --- Add your ACTUAL recommendation logic here ---
            # This section needs to:
            # 1. Take `user_mood_input`.
            # 2. If your `model_object` is a text processing model (like BERT),
            #    you might pass the mood text to it to get embeddings or a classification.
            #    (If it's just a mood string, you might use it directly to filter your tracks).
            # 3. Use the processed mood and the `track_df` to find and display matching songs.

            # Placeholder recommendation logic:
            # For now, let's just filter the DataFrame if there's a 'mood' or 'emotion' column,
            # or show random samples.
            # You'll need to adapt this based on your `track_df` columns and model output.

            # Example: if your track_df has an 'emotion' column that matches your model's output
            if 'emotion' in track_df.columns: # Replace 'emotion' with your actual column name
                recommended_tracks = track_df[track_df['emotion'].str.lower() == user_mood_input.lower()]
                if not recommended_tracks.empty:
                    st.write(f"Found {len(recommended_tracks)} tracks matching this mood:")
                    st.dataframe(recommended_tracks.head()) # Show top matches
                else:
                    st.warning(f"No tracks found directly matching '{user_mood_input}'. Showing some popular tracks instead.")
                    st.dataframe(track_df.sample(min(5, len(track_df))))
            else:
                st.info("Track data doesn't have a direct 'emotion' column for filtering. Showing random samples.")
                st.dataframe(track_df.sample(min(5, len(track_df)))) # Show 5 random tracks

        else:
            st.warning("Please select a mood.")
elif model_object and (track_df is None or track_df.empty):
    st.warning("Model loaded, but track data could not be loaded or is empty. Please check the data file and path.")
    st.info(f"Expected data file path: {os.path.abspath(DATA_FILE_PATH)}")
else:
    st.error("App cannot start. Model or track data failed to load. Please check messages and logs.")
    st.info("Ensure `google_drive_model_zip_url` is set in secrets for model download.")
    st.info(f"Expected data file path: {os.path.abspath(DATA_FILE_PATH)}")

# Add a footer or more info in the sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("Created with Streamlit.")
