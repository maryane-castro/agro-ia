import os
import pandas as pd
import numpy as np
import cv2
from scipy.spatial import distance as dist
import logging
import time
from ultralytics import YOLO

CAMERA_HEIGHT = 3060
MODEL_PATH = "./models/best.pt"  # Substitua pelo caminho do seu modelo treinado

# Carregar o modelo YOLO
model = YOLO(MODEL_PATH)
OUTPUT_DIR = "./segmented_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def detect_animal(image):
    results = model(image)
    if not results or not results[0].masks:
        return None  # Se não houver detecção, retorna None
    return results[0].masks.xy  # Retorna as coordenadas das máscaras

# volume

def calc_volume(df):
    '''
    Input:
    df: dataframe de mapa de profundidade após remoção dos outliers. 
    Output:
    volume: Soma das alturas de todos os pixels do animal.
    '''
    df["height"] = CAMERA_HEIGHT - df["dist"] #build new column named height
    volume = sum(df.height)
    return volume



# height

def calc_avg_height(fill_img, dfcsv):
    '''
    Input: 
    fill_img: imagem binária obtida a partir da segmentação.
    dfcsv: mapa de profundidade.
    Output:
    avg_height: altura média.
    df: mapa de profundidade removendo outliers.
    '''
    pixel = np.argwhere(fill_img == 255) #find pixels for white part
    dfcsv_rows = [] #combine pixel and distance
    for row, col in pixel:
      dfcsv_rows.append([row, col, dfcsv.iloc[row, col]])
    df = pd.DataFrame(dfcsv_rows, columns = ['row', 'col', 'dist'])
    df.dist.replace(to_replace=0, value = df.dist.mean(), inplace=True) #replace 0 with average distance
    avg_height = CAMERA_HEIGHT - df.dist.mean()
    return avg_height, df

def calc_centroid_height(cmax, dfcsv):
    '''
    Input: 
    cmax: contorno máximo da imagem segmentada
    dfcsv: mapa de profundidade
    Output:
    cen_height: Altura centroide
    '''
    M = cv2.moments(cmax)
    row_centroid = int(M["m01"] / M["m00"])
    col_centroid  = int(M["m10"] / M["m00"])
    print([row_centroid, col_centroid, dfcsv.iloc[row_centroid , col_centroid]])
    cen_height = CAMERA_HEIGHT - dfcsv.iloc[row_centroid , col_centroid]
    return cen_height


# width and Length 

def calc_width_legth(fill_img):
  '''
  Input: 
  fill_image: imagem segmentada
  Outputs:
  width: largura máxima do contorno em píxels
  length: comprimento máximo do contorno em píxels
  cmax: contorno máximo na imagem, necessário para os próximos passos.
  '''
  cnts, _ = cv2.findContours(fill_img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
  cmax = max(cnts, key=cv2.contourArea)
  rect = cv2.minAreaRect(cmax)
  box = cv2.boxPoints(rect)
  (A, B, C, D) = np.int0(box)
  d0 = dist.euclidean(B, C)
  d1 = dist.euclidean(A, B)
  width = min(d0, d1)
  length = max(d0, d1)
  return width, length, cmax


def save_image_with_bbox(image, masks, image_name):
    """
    Função para salvar a imagem com a caixa delimitadora
    """
    # Desenhando caixas delimitadoras em cada máscara
    for mask in masks:
        if len(mask) > 0:
            mask = np.array(mask, dtype=np.int32)
            cv2.fillPoly(image, [mask], (0, 255, 0))  # Preenche a máscara com verde
            x, y, w, h = cv2.boundingRect(mask)  # Caixa delimitadora
            #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Desenha a caixa em vermelho
    output_path = os.path.join(OUTPUT_DIR, image_name)
    cv2.imwrite(output_path, image)
    print(f"Imagem salva com caixa delimitadora em: {output_path}")

def extract_animal_features(depth_images, depth_csvs):
    results = []
    logging.basicConfig(level=logging.DEBUG)
    
    for i, image_path in enumerate(depth_images):
        start_time = time.time()
        img = cv2.imread(image_path)
        if img is None:
            logging.warning(f"Erro ao carregar imagem: {image_path}")
            results.append([np.nan] * 5)
            continue
        
        masks = detect_animal(img)
        logging.debug(f"--- Predict Took {time.time() - start_time} seconds ---")
        
        if masks is None:
            results.append([np.nan] * 5)
            continue


        #print(masks)
        
        # Salvar imagem com a máscara e a caixa delimitadora
        image_name = os.path.basename(image_path)
        save_image_with_bbox(img.copy(), masks, image_name)
        
        fill_img = np.zeros(img.shape[:2], dtype=np.uint8)
        for mask in masks:
            if len(mask) > 0:
                cv2.fillPoly(fill_img, [np.array(mask, dtype=np.int32)], 255)
        
        try:
            dfcsv = pd.read_csv(depth_csvs[i], header=None)
        except Exception as e:
            logging.warning(f"Erro ao ler CSV: {depth_csvs[i]} - {e}")
            results.append([np.nan] * 5)
            continue
        
        width, length, cmax = calc_width_legth(fill_img)
        centroid_height = calc_centroid_height(cmax, dfcsv) #if cmax is not None else np.nan
        avg_height, df = calc_avg_height(fill_img, dfcsv)
        volume = calc_volume(df)
        
        results.append([width, length, centroid_height, avg_height, volume])
    
    return results
