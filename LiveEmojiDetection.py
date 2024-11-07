import cv2
from fer import FER
import time

# Initialize emotion detector
detector = FER()

# Open the default camera (index 0)
cap = cv2.VideoCapture(0)

# Check if the camera is successfully opened
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Load face detection model (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize FPS counter
fps = 0
prev_time = time.time()

# Start capturing video
while True:
    ret, frame = cap.read()

    # If the frame is not read correctly, break the loop
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Detect emotions in the frame using the FER library
    emotion, score = detector.top_emotion(frame)

    # Map the detected emotion to a corresponding emoji
    emoji_map = {
        "happy": "😊",
        "sad": "😞",
        "angry": "😡",
        "surprise": "😲",
        "fear": "😨",
        "disgust": "🤢",
        "neutral": "😐"
    }

    # Draw a rectangle around the faces and display the emotion with emoji
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, emoji_map.get(emotion, '') + " " + emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show a feedback bar for the emotion's intensity
    intensity = int(score * 100)  # Get the emotion intensity as percentage
    cv2.rectangle(frame, (10, 50), (intensity + 10, 70), (0, 255, 0), -1)
    cv2.putText(frame, f"Emotion Intensity: {intensity}%", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    # Calculate and display the FPS (Frames per second)
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with the detected faces, emotions, and feedback
    cv2.imshow('Live Emoji Detection', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
