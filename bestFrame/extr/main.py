import cv2

def calcular_nitidez(frame):
    """Calcula a nitidez de um frame usando a variação Laplaciana."""
    return cv2.Laplacian(frame, cv2.CV_64F).var()

def encontrar_frame_mais_nitido(video_path):
    """Encontra o frame mais nítido em um arquivo de vídeo."""
    cap = cv2.VideoCapture(video_path)
    max_nitidez = 0
    frame_mais_nitido = None
    frame_id = -1
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        nitidez = calcular_nitidez(frame)

        if nitidez > max_nitidez:
            max_nitidez = nitidez
            frame_mais_nitido = frame
            frame_id = frame_count

    cap.release()
    return frame_mais_nitido, frame_id, max_nitidez

# Caminho para o vídeo
video_path = "output_video.mp4"

# Processa o vídeo para encontrar o frame mais nítido
frame, id_frame, nitidez = encontrar_frame_mais_nitido(video_path)

# Salva o frame mais nítido encontrado
if frame is not None:
    print(f"Frame mais nítido encontrado no ID: {id_frame} com nitidez: {nitidez}")
    cv2.imwrite("frame_mais_nitido.jpg", frame)
    print("Frame salvo como 'frame_mais_nitido.jpg'")
else:
    print("Nenhum frame encontrado ou erro ao processar o vídeo.")
