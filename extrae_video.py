import cv2
import os
import glob


def extract_keyframes(video_path, output_dir, frame_interval=30):
    """
    Extrae un frame cada N frames de un video.
    Retorna lista de paths a los frames extraídos.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved = 0
    frame_paths = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{saved:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            saved += 1

        count += 1

    cap.release()
    return frame_paths


def extract_all_videos(videos_dir, output_base_dir="data/frames", frame_interval=30):
    """
    Extrae keyframes de TODOS los videos MP4 en videos_dir.
    Organiza por nombre de video como etiqueta.
    Retorna dict: {video_name: [frame_paths]}
    """
    os.makedirs(output_base_dir, exist_ok=True)
    video_files = sorted(glob.glob(os.path.join(videos_dir, "*.mp4")))

    all_frames = {}
    for video_path in video_files:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(output_base_dir, video_name)
        frames = extract_keyframes(video_path, output_dir, frame_interval)
        all_frames[video_name] = frames
        print(f"  {video_name}: {len(frames)} frames extraídos")

    return all_frames
