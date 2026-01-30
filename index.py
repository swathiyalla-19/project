import cv2
import numpy as np
import datetime
import webbrowser
from tensorflow.keras.models import load_model
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# =========================
# 1. Load Emotion Model
# =========================
model = load_model("emotion_model.h5")   # keep model file in same folder
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# =========================
# 2. Spotify API Setup
# =========================
client_id = "YOUR_SPOTIFY_CLIENT_ID"
client_secret = "YOUR_SPOTIFY_CLIENT_SECRET"

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=client_id,
    client_secret=client_secret
))

# Mood → Spotify playlist mapping (you can update these IDs with your own)
mood_playlists = {
    "Happy": "37i9dQZF1DXdPec7aLTmlC",    # Happy Hits
    "Sad": "37i9dQZF1DX7qK8ma5wgG1",      # Sad Songs
    "Angry": "37i9dQZF1DWY4xHQp97fN6",    # Rock Hard
    "Neutral": "37i9dQZF1DX3rxVfibe1L0",  # Chill Vibes
    "Fear": "37i9dQZF1DX4sWSpwq3LiO",     # Relax & Unwind
    "Surprise": "37i9dQZF1DX4fpCWaHOned", # Discover Weekly
    "Disgust": "37i9dQZF1DX70RN3TfWWJh"   # Random playlist for disgust
}

def play_music(emotion):
    """Play a track from Spotify based on detected emotion"""
    playlist_id = mood_playlists.get(emotion, mood_playlists["Neutral"])
    results = sp.playlist_tracks(playlist_id)
    if results['items']:
        track_url = results['items'][0]['track']['external_urls']['spotify']
        print(f"[INFO] Playing {emotion} music → {track_url}")
        webbrowser.open(track_url)
    else:
        print("[WARNING] No tracks found for this mood.")

def log_mood(emotion):
    """Save detected mood with timestamp to history file"""
    with open("mood_history.txt", "a") as f:
        f.write(f"{datetime.datetime.now()} - {emotion}\n")

# =========================
# 3. Start Webcam for Emotion Detection
# =========================
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("[INFO] Press 'q' to quit the webcam.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype('float') / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        prediction = model.predict(roi, verbose=0)[0]
        max_index = int(np.argmax(prediction))
        emotion = emotion_labels[max_index]

        # Display emotion on screen
        cv2.putText(frame, emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h),
                      (0, 255, 0), 2)

        # Log and Play Music
        log_mood(emotion)
        play_music(emotion)

    cv2.imshow("Emotion-Based Music Recommendation", frame)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
