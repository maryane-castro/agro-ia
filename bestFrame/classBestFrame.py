import cv2
import os
import shutil
import numpy as np
from concurrent.futures import ThreadPoolExecutor


def calcular_nitidez(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)

    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), dtype=np.uint8)
    mask[crow - 120:crow + 120, ccol - 60:ccol + 60] = 0

    high_freqs = magnitude_spectrum * mask

    return np.mean(np.log1p(high_freqs))


def salvar_frames_ordenados(video_path, output_folder, reducao_percentual=10):
    cap = cv2.VideoCapture(video_path)
    frames_nitidez = []
    frame_count = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    excluir_frames = int(total_frames * (reducao_percentual / 100))

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    def processar_frame(frame_id, frame):
        if frame_id <= excluir_frames or frame_id > total_frames - excluir_frames:
            return None
        nitidez = calcular_nitidez(frame)
        return (frame_id, nitidez, frame)

    with ThreadPoolExecutor() as executor:
        futures = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            futures.append(executor.submit(processar_frame, frame_count, frame))

        for future in futures:
            result = future.result()
            if result:
                frames_nitidez.append(result)

    cap.release()

    frames_nitidez.sort(key=lambda x: x[1], reverse=True)

    for i, (frame_id, nitidez, frame) in enumerate(frames_nitidez):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rows, cols = gray_frame.shape
        crow, ccol = rows // 2, cols // 2

        height = 700
        width = 300

        mask = np.zeros((rows, cols), dtype=np.uint8)
        mask[crow - height // 2:crow + height // 2, ccol - width // 2:ccol + width // 2] = 255

        binarized_frame = cv2.bitwise_and(gray_frame, gray_frame, mask=mask)

        binarized_filename = f"{output_folder}/frame_{i+1:03d}_id_{frame_id}_nitidez_{nitidez:.2f}_gray.jpg"
        cv2.imwrite(binarized_filename, binarized_frame)
        
        filename = f"{output_folder}/frame_{i+1:03d}_id_{frame_id}_nitidez_{nitidez:.2f}.jpg"
        cv2.imwrite(filename, frame)

    print(f"Frames salvos em ordem de nitidez na pasta: {output_folder}")
    print(f"Foram ignorados os primeiros e últimos {excluir_frames} frames ({reducao_percentual}% do total).")