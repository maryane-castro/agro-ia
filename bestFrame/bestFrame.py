import os
import cv2
import torch
import numpy as np
import shutil
from ultralytics import YOLO

def measure_sharpness(image):
    """ Mede a nitidez do frame usando a variância do Laplaciano """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def process_frames(inputs_png, inputs_csv, output_base_dir, model_path, margem=5):
    model = YOLO(model_path)
    
    # Lista as subpastas dentro de inputs_png
    subfolders = [f for f in os.listdir(inputs_png) if os.path.isdir(os.path.join(inputs_png, f))]
    
    for svo_name in subfolders:
        input_dir = os.path.join(inputs_png, svo_name)
        input_csv_dir = os.path.join(inputs_csv, svo_name)
        output_dir_pngs = os.path.join(output_base_dir, "pngs", svo_name)
        output_dir_csvs = os.path.join(output_base_dir, "csvs", svo_name)
        output_dir_best = os.path.join(output_base_dir, "best_frames", svo_name)
        
        os.makedirs(output_dir_pngs, exist_ok=True)
        os.makedirs(output_dir_csvs, exist_ok=True)
        os.makedirs(output_dir_best, exist_ok=True)
        
        detected_frames = []
        png_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".png")])
        
        for png_path in png_files:
            frame = cv2.imread(png_path)
            original_shape = frame.shape  # Salva a forma original da imagem
            frame_resized = cv2.resize(frame, (640, 640))  # Redimensiona para 640x640
            
            frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0  
            frame_tensor = frame_tensor.unsqueeze(0)  # Adiciona uma dimensão de batch
            
            results = model(frame_tensor)  # Faz a detecção
            
            valid_detection = False
            best_frame = frame.copy()
            
            for result in results:
                for box in result.boxes:
                    conf = box.conf.item()
                    if conf > 0.8:
                        xmin, ymin, xmax, ymax = map(int, box.xyxy[0])  # Coordenadas do bounding box no tamanho 640x640
                        
                        # Redimensiona as coordenadas para o tamanho original da imagem
                        xmin = int(xmin * original_shape[1] / 640)
                        xmax = int(xmax * original_shape[1] / 640)
                        ymin = int(ymin * original_shape[0] / 640)
                        ymax = int(ymax * original_shape[0] / 640)

                        # Verifica se o bounding box toca no limite superior ou inferior
                        if ymin < margem or ymax > frame_resized.shape[0] - margem:  # 5px de margem
                            continue  # Ignora se o bounding box toca nas bordas da imagem

                        valid_detection = True
                        cv2.rectangle(best_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # Desenha a bbox
                        
            if valid_detection:
                detected_frames.append((png_path, best_frame))
        
        # Se menos de 5 frames forem detectados, não salva nada
        if len(detected_frames) < 5:
            continue
        
        # Seleciona entre 2 a 7 ou até o final
        if len(detected_frames) >= 10:
            selected_frames = [detected_frames[i] for i in [2, 3, 4, 5, 6, 7]]
        else:
            selected_frames = detected_frames[2:]
        
        # Salva os frames selecionados
        for frame_path, best_frame in selected_frames:
            frame_name = os.path.basename(frame_path)
            output_path = os.path.join(output_dir_pngs, frame_name)
            output_best_path = os.path.join(output_dir_best, frame_name)
            
            # Salva as imagens PNG
            cv2.imwrite(output_path, cv2.imread(frame_path))
            cv2.imwrite(output_best_path, best_frame)
            
            # Copia os arquivos CSV correspondentes
            csv_name = frame_name.replace(".png", ".csv")
            csv_source = os.path.join(input_csv_dir, csv_name)
            csv_dest = os.path.join(output_dir_csvs, csv_name)
            if os.path.exists(csv_source):
                shutil.copy(csv_source, csv_dest)
