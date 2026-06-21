import cv2
import mediapipe as mp
from gtts import gTTS
from playsound import playsound
import json
import time
import os
import threading  # <-- new

# ------------------------
# Load gestures
# ------------------------
with open("gestures.json") as f:
    gestures = json.load(f)

# ------------------------
# Speak function (threaded 🔊)
# ------------------------
def speak(text):
    def run_speech(text):
        print("Speaking:", text)
        tts = gTTS(text=text, lang='en')
        tts.save("voice.mp3")
        playsound("voice.mp3")
        os.remove("voice.mp3")
    
    # Run in separate thread
    threading.Thread(target=run_speech, args=(text,), daemon=True).start()

# ------------------------
# Sentence memory
# ------------------------
sentence_words = []
last_add_time = 0
last_spoken = ""

# ------------------------
# Sentence builder
# ------------------------
def build_sentence(words):
    w = set(words)
    if "Help" in w and "Stop" in w:
        return "Emergency! I need help"
    if "Hello" in w and "Help" in w:
        return "Hello, I need help"
    if "Yes" in w and "Help" in w:
        return "Please help me"
    if "Water" in w:
        return "I am thirsty, I need water"
    if "Food" in w:
        return "I am hungry, I need food"
    if "Doctor" in w:
        return "Please call a doctor"
    if "Emergency" in w:
        return "This is an emergency situation"
    if "Hello" in w:
        return "Hello, how are you?"
    if "Thank You" in w:
        return "Thank you very much"
    return " ".join(words)

# ------------------------
# MediaPipe setup
# ------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
draw = mp.solutions.drawing_utils

# ------------------------
# Finger count
# ------------------------
def count_fingers(landmarks):
    if not landmarks:
        return 0
    tips = [4, 8, 12, 16, 20]
    fingers = []
    fingers.append(1 if landmarks[tips[0]][1] > landmarks[tips[0]-1][1] else 0)
    for i in range(1, 5):
        fingers.append(1 if landmarks[tips[i]][2] < landmarks[tips[i]-2][2] else 0)
    return sum(fingers)

# ------------------------
# Webcam
# ------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    landmarks_list = []

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            for id, lm in enumerate(hand.landmark):
                h, w, _ = frame.shape
                landmarks_list.append((id, int(lm.x * w), int(lm.y * h)))

    # Detect word
    finger_count = count_fingers(landmarks_list)
    word = gestures.get(str(finger_count), "")

    # Add words (delay control)
    if word and (time.time() - last_add_time > 1.5):
        sentence_words.append(word)
        last_add_time = time.time()

    # Build sentence
    final_sentence = build_sentence(sentence_words)

    # 🔥 Speak sentence (non-blocking)
    if final_sentence and final_sentence != last_spoken and len(sentence_words) >= 1:
        speak(final_sentence)
        last_spoken = final_sentence
        sentence_words = []

    # ------------------------
    # UI
    # ------------------------
    h, w, _ = frame.shape
    cv2.rectangle(frame, (0, 0), (w, 70), (30, 30, 30), -1)
    cv2.putText(frame, "SignSpeak AI", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Detected: {word}", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, f"Sentence: {final_sentence}", (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("SignSpeak AI", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
