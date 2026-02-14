import os

MAX_VIDEO_SIZE_MB = 20  # Ajustar según límites actuales de Gemini


def validate_video_size(video_path: str):
    """
    Valida que el video no exceda el tamaño máximo permitido.
    """
    size_mb = os.path.getsize(video_path) / (1024 * 1024)
    if size_mb > MAX_VIDEO_SIZE_MB:
        raise ValueError(
            f"El archivo {video_path} excede el límite de {MAX_VIDEO_SIZE_MB}MB."
        )
