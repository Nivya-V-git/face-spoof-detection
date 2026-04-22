import cv2
import os

video_root = "videos"
dataset_root = "dataset"
FRAME_SKIP = 10

def extract_frames(category):
    video_folder = os.path.join(video_root, category)
    output_folder = os.path.join(dataset_root, category)
    os.makedirs(output_folder, exist_ok=True)

    for video_name in os.listdir(video_folder):
        if not video_name.endswith(('.mp4', '.avi', '.mov')):
            continue
        video_path = os.path.join(video_folder, video_name)
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        saved_count = 0
        print(f"Processing {video_name}...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % FRAME_SKIP == 0:
                filename = f"{video_name}_frame_{saved_count}.jpg"
                cv2.imwrite(os.path.join(output_folder, filename), frame)
                saved_count += 1
            frame_count += 1
        cap.release()
        print(f"Saved {saved_count} frames from {video_name}")

extract_frames("real")
extract_frames("spoof")
print("Frame extraction completed.")