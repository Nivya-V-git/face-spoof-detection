# face-spoof-detection
# Real-Time Face Spoofing Detection Using CNN and Depth Analysis

## Project Overview
This project detects whether a face in front of a webcam is real or a spoof attack (photo or screen replay).  
The system combines CNN-based texture analysis and depth estimation to identify fake faces in real time.

## Key Features
- Real-time webcam face detection
- CNN-based spoof detection
- Depth estimation to detect flat surfaces
- Blink detection for liveness verification
- Detection of photo and screen replay attacks

## Technologies Used
- Python
- OpenCV
- TensorFlow / PyTorch
- MiDaS depth estimation model

## System Architecture
Camera → Face Detection → CNN Model → Depth Analysis → Liveness Check → Decision Output

## Folder Structure
dataset/ – real and fake face data  
cnn_model/ – CNN training and model files  
depth_module/ – depth estimation and analysis  
integration/ – webcam and final system  
docs/ – project documentation

## Team Members
- Member 1 – Dataset & Attack Collection
- Member 2 – CNN Model Development
- Member 3 – Depth Analysis & Liveness Detection
- Member 4 – System Integration & Testing
