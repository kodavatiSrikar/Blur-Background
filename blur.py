import cv2
import os
import numpy as np

# Load the pre-trained face detection Haar Cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Root input and output directories
input_root = "study_videos"
output_root = "blur_videos"

# Create output root if it doesn't exist
os.makedirs(output_root, exist_ok=True)

# Define a fixed bounding box size
fixed_box_size = 200  # Fixed size for all face bounding boxes
zoom_factor = 2       # Zoom factor for all bounding boxes


def process_video(input_path, output_path):
    print(f"Processing video: {input_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        print(f"Error: Invalid FPS for video {input_path}")
        cap.release()
        return

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) > 0:
            # Select the largest detected face
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
            x, y, w, h = faces[0]

            if w >= fixed_box_size and h >= fixed_box_size:
                # If the bounding box is already larger than fixed size, use it directly
                cropped_frame = frame[y:y + h, x:x + w]
            else:
                # Adjust bounding box to fixed size centered on detected face
                cx, cy = x + w // 2, y + h // 2
                half_fixed_size = fixed_box_size // 2

                x1 = max(0, cx - half_fixed_size)
                y1 = max(0, cy - half_fixed_size)
                x2 = min(frame_width, cx + half_fixed_size)
                y2 = min(frame_height, cy + half_fixed_size)

                cropped_frame = frame[y1:y2, x1:x2]

            # Scale the cropped frame by zoom factor
            zoomed_frame = cv2.resize(
                cropped_frame,
                (fixed_box_size * zoom_factor, fixed_box_size * zoom_factor),
                interpolation=cv2.INTER_LINEAR
            )

            # Center zoomed frame on black canvas of original frame size
            canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

            canvas_cx, canvas_cy = frame_width // 2, frame_height // 2
            zoom_h, zoom_w = zoomed_frame.shape[:2]

            start_x = max(0, canvas_cx - zoom_w // 2)
            start_y = max(0, canvas_cy - zoom_h // 2)
            end_x = min(frame_width, start_x + zoom_w)
            end_y = min(frame_height, start_y + zoom_h)

            canvas[start_y:end_y, start_x:end_x] = zoomed_frame[0:end_y - start_y, 0:end_x - start_x]

            # Write output frame
            out.write(canvas)

        else:
            # No face detected: blur + darken frame
            blurred_frame = cv2.GaussianBlur(frame, (99, 99), 30)
            darkened_frame = cv2.addWeighted(blurred_frame, 0.5, np.zeros_like(frame), 0.5, 0)
            out.write(darkened_frame)

        frame_count += 1

    cap.release()
    out.release()

    if frame_count > 0:
        print(f"Saved: {output_path}")
    else:
        print(f"No frames processed for video: {input_path}")


# Recursively walk through input_root and process all .mp4 files
for root, dirs, files in os.walk(input_root):
    for file in files:
        if file.lower().endswith(".mp4"):
            input_path = os.path.join(root, file)

            # Get relative path from input_root (e.g., agreeableness/high/file.mp4)
            rel_path = os.path.relpath(input_path, input_root)

            # Build corresponding output path under output_root
            output_path = os.path.join(output_root, rel_path)

            process_video(input_path, output_path)

print("Processing completed.")