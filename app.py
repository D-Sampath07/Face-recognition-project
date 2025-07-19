import cv2
import face_recognition
import os
import numpy as np
import winsound
import time
from telegram import Bot
from datetime import datetime
import asyncio

# --- Telegram Config ---
TELEGRAM_BOT_TOKEN = "Your_Bot_Token_Here"
TELEGRAM_CHAT_ID = "Your_Chat_ID_Here"
bot = Bot(token=TELEGRAM_BOT_TOKEN)

# --- Async Event Loop Setup ---
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

async def send_telegram_alert(message, image=None):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # Clean filename
    filename = None  # Ensure it's defined in case an exception hits early

    try:
        if image is not None:
            filename = f"alert_{timestamp}.jpg"
            cv2.imwrite(filename, image)
            with open(filename, "rb") as photo:
                await bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=photo, caption=f"{message}\n{timestamp}")
            os.remove(filename)  # Only delete if success
        else:
            await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"{message}\n{timestamp}")
        print(f"[TELEGRAM] Sent: {message}")
    except Exception as e:
        print(f"[ERROR] Telegram alert failed: {e}")
        if filename:
            print(f"[BACKUP] Alert image saved as {filename}")



# --- Alert Control ---
last_exam_alert_time = 0
last_security_alerts = {}  # name: timestamp
ALERT_COOLDOWN = 60  # seconds

# --- Directories ---
ENCODING_DIR = os.path.join("known_faces", "encodings")
THUMBNAIL_DIR = os.path.join("known_faces", "thumbnails")
os.makedirs(ENCODING_DIR, exist_ok=True)
os.makedirs(THUMBNAIL_DIR, exist_ok=True)

# --- Load Known Encodings ---
known_encodings = []
known_names = []
for file in os.listdir(ENCODING_DIR):
    if file.endswith(".npy"):
        encoding = np.load(os.path.join(ENCODING_DIR, file))
        known_encodings.append(encoding)
        known_names.append(os.path.splitext(file)[0])
print(f"[INFO] Loaded {len(known_names)} encoded face(s).")

# --- Mode Selection ---
mode_input = input("Enter mode: [1] Exam  [0] Security  [Anything else] Normal: ").strip()

mode = "exam" if mode_input == '1' else "security" if mode_input == '0' else "normal"
print(f"[MODE] {mode.upper()} MODE activated.")

# --- Start Webcam ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    face_names = []
    for encoding in face_encodings:
        name = "Unknown"
        if known_encodings:
            distances = face_recognition.face_distance(known_encodings, encoding)
            best_match_index = np.argmin(distances)
            if distances[best_match_index] < 0.5:
                name = known_names[best_match_index]
        face_names.append(name)

    # --- Exam Mode Alert ---
    if mode == "exam" and len(face_locations) > 1:
        if current_time - last_exam_alert_time > ALERT_COOLDOWN:
            winsound.Beep(1000, 200)
            loop.run_until_complete(send_telegram_alert("⚠️ Exam Mode Alert: Multiple faces detected!", frame))
            last_exam_alert_time = current_time

    # --- Security Mode Alert ---
    elif mode == "security" and face_locations:
        for name, (top, right, bottom, left) in zip(face_names, face_locations):
            top *= 4; right *= 4; bottom *= 4; left *= 4
            person_image = frame

            if name not in last_security_alerts or current_time - last_security_alerts[name] > ALERT_COOLDOWN:
                if name == "Unknown":
                    loop.run_until_complete(send_telegram_alert("🚨 Security Alert: Unknown person detected!", person_image))
                else:
                    loop.run_until_complete(send_telegram_alert(f"✅ Security Info: {name} detected.", person_image))
                last_security_alerts[name] = current_time

    # --- Drawing ---
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4; right *= 4; bottom *= 4; left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.putText(frame, f"Faces: {len(face_locations)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("FaceNotify - Face Recognition", frame)

    key = cv2.waitKey(1) & 0xFF

    # --- Save New Face ---
    if key == ord('s'):
        if face_locations:
            print("[INFO] Saving face...")
            name = input("Enter name for this face: ").strip()
            if name:
                top, right, bottom, left = face_locations[0]
                top *= 4; right *= 4; bottom *= 4; left *= 4

                face_image = frame[top:bottom, left:right]
                thumb_path = os.path.join(THUMBNAIL_DIR, f"{name}.jpg")
                enc_path = os.path.join(ENCODING_DIR, f"{name}.npy")

                cv2.imwrite(thumb_path, face_image)
                rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                encoding = face_recognition.face_encodings(rgb_face)
                if encoding:
                    np.save(enc_path, encoding[0])
                    print(f"[INFO] Saved encoding as {enc_path}")
                else:
                    print("[ERROR] Could not extract face encoding.")
            else:
                print("[WARNING] Name cannot be empty.")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
loop.close()
