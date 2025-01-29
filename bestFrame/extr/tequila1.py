import cv2
import os
import shutil
import numpy as np
from concurrent.futures import ThreadPoolExecutor


def calcular_nitidez(frame):
    """Calcula a nitidez usando a Transformada de Fourier com componentes de alta frequência."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)

    # Definir uma máscara para isolar as altas frequências
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2  # Coordenadas do centro
    mask = np.ones((rows, cols), dtype=np.uint8)
    mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0  # Bloqueia frequências baixas

    # Aplicar máscara
    high_freqs = magnitude_spectrum * mask

    # Calcular a nitidez como a média das altas frequências
    return np.mean(np.log1p(high_freqs))


def salvar_frames_ordenados(video_path, output_folder, reducao_percentual=10):
    """Extrai frames de um vídeo, calcula a nitidez e os salva em ordem decrescente, excluindo uma porcentagem dos primeiros e últimos frames."""
    cap = cv2.VideoCapture(video_path)
    frames_nitidez = []
    frame_count = 0

    # Obtém o total de frames no vídeo
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calcula os frames a serem excluídos no início e no final
    excluir_frames = int(total_frames * (reducao_percentual / 100))

    # Remove a pasta de saída, se existir, e recria
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)  # Remove a pasta e todo o conteúdo
    os.makedirs(output_folder)  # Cria a pasta novamente

    # Processa os frames
    def processar_frame(frame_id, frame):
        if frame_id <= excluir_frames or frame_id > total_frames - excluir_frames:
            return None
        nitidez = calcular_nitidez(frame)
        return (frame_id, nitidez, frame)

    # Paralelizar o processamento de frames
    with ThreadPoolExecutor() as executor:
        futures = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            futures.append(executor.submit(processar_frame, frame_count, frame))

        # Coletar resultados válidos
        for future in futures:
            result = future.result()
            if result:
                frames_nitidez.append(result)

    cap.release()

    # Ordena os frames pelo valor de nitidez em ordem decrescente
    frames_nitidez.sort(key=lambda x: x[1], reverse=True)

    # Salva os frames na pasta de saída
    for i, (frame_id, nitidez, frame) in enumerate(frames_nitidez):
        # Salvar imagem binarizada
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binarized_frame = cv2.threshold(gray_frame, 127, 255, cv2.THRESH_BINARY)
        
        # Nome do arquivo com sufixo "_binarizado"
        binarized_filename = f"{output_folder}/frame_{i+1:03d}_id_{frame_id}_nitidez_{nitidez:.2f}_binarizado.jpg"
        cv2.imwrite(binarized_filename, binarized_frame)
        
        # Salvar a versão colorida original
        filename = f"{output_folder}/frame_{i+1:03d}_id_{frame_id}_nitidez_{nitidez:.2f}.jpg"
        cv2.imwrite(filename, frame)

    print(f"Frames salvos em ordem de nitidez na pasta: {output_folder}")
    print(f"Foram ignorados os primeiros e últimos {excluir_frames} frames ({reducao_percentual}% do total).")


# Caminho para o vídeo
video_path = "output_video.mp4"
output_folder = "frames_ordenados"

# Porcentagem de frames a serem excluídos do início e do fim
reducao_percentual = 31  # Excluir 10% do início e 10% do final

# Processa o vídeo e salva os frames
salvar_frames_ordenados(video_path, output_folder, reducao_percentual)
