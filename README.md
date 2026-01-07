AI-Powered Online Proctoring System 🛡️🤖
An automated remote proctoring solution designed to maintain the integrity of online examinations. This system uses Computer Vision and Deep Learning to monitor students in real-time, detecting suspicious activities without the need for a human supervisor.

🌟 Key Features
Face Verification: Matches the student's face with their registered ID before the exam starts.

Head Pose Estimation: Detects if the student is looking away from the screen (left, right, up, or down).

Multiple Person Detection: Alerts if more than one person is visible in the camera frame.

Object Detection: Identifies prohibited items like mobile phones, books, or earphones.

Mouth Opening/Speech Detection: Detects if the student is talking or whispering.

Active Tab Monitoring: (If integrated with a web app) Prevents switching tabs or windows.

Real-time Alerts: Provides a "Suspicion Score" and generates logs for the instructor.

🏗️ System Architecture
The system follows a modular pipeline to process video frames in real-time:

Preprocessing: Normalizing frames and resizing for the model.

Detection Layer: Using MediaPipe/OpenCV for facial landmarks and YOLO for object detection.

Analysis Layer: Logic to determine if the detected behavior violates exam rules.

Reporting: Storing evidence (screenshots/logs) in a database.

🛠️ Tech Stack
Language: Python

Libraries: OpenCV, MediaPipe, TensorFlow/Keras

Object Detection: YOLO (v5/v8)

Backend: Flask / FastAPI (Optional)

Frontend: React / Streamlit (Optional)

Database: SQLite / MongoDB (For logs)

🚀 Getting Started
Prerequisites

Python 3.8+

Webcam
