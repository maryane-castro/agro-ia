import os
import pandas as pd
import numpy as np
import cv2
from scipy.spatial import distance as dist
import skimage.io as io
import logging
import time
from ultralytics import YOLO

CAMERA_HEIGHT = 3060

CAMERA_HEIGHT = 3060
MODEL_PATH = "./models/best.pt"  # Substitua pelo caminho do seu modelo treinado

# Carregar o modelo YOLO
model = YOLO(MODEL_PATH)

def detect_animal(image):
    print(image)
    results = model(image)
    masks = results[0].masks  # Obtém as máscaras das detecções
    if masks is not None:
        return masks.xy  # Retorna as coordenadas das máscaras
    return None

def calc_volume(df, camera_height):
    # Calcula o volume usando a altura validada
    df["height"] = camera_height - df["dist"]
    df = df[df["height"].notna()]  # Remove linhas com valores nulos de altura
    return df["height"].sum()  # Retorna a soma das alturas

def calc_avg_height(fill_img, dfcsv):
    pixel = np.argwhere(fill_img == 255)
    dfcsv_rows = [[row, col, dfcsv.iloc[row, col]] for row, col in pixel]
    df = pd.DataFrame(dfcsv_rows, columns=['row', 'col', 'dist'])
    df["dist"] = df["dist"].replace(to_replace=0, value=df["dist"].mean())
    height1 = CAMERA_HEIGHT - df.dist.mean()
    return height1, df

def calc_centroid_height(cmax, dfcsv):
    M = cv2.moments(cmax)
    row_centroid = int(M["m01"] / M["m00"])
    col_centroid = int(M["m10"] / M["m00"])
    return CAMERA_HEIGHT - dfcsv.iloc[row_centroid, col_centroid]

def calc_width_length(fill_img):
    cnts, _ = cv2.findContours(fill_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.nan, np.nan, None
    cmax = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(cmax)
    box = cv2.boxPoints(rect)
    (A, B, C, D) = np.intp(box)
    d0 = dist.euclidean(B, C)
    d1 = dist.euclidean(A, B)
    return min(d0, d1), max(d0, d1), cmax

def extract_animal_features(depth_images, depth_csvs):
    results = []
    logging.basicConfig(level=logging.DEBUG)
    for i, image_path in enumerate(depth_images):
        start_time = time.time()
        img = cv2.imread(image_path)
        masks = detect_animal(img)
        print("--- Predict Took %s seconds ---" % (time.time() - start_time))
        
        if masks is None:
            results.append([np.nan] * 5)  # Se não detectar, retorna NaN
            continue
        
        fill_img = np.zeros(img.shape[:2], dtype=np.uint8)
        for mask in masks:
            cv2.fillPoly(fill_img, [np.array(mask, dtype=np.int32)], 255)
        
        dfcsv = pd.read_csv(depth_csvs[i], header=None)
        width, length, cmax = calc_width_length(fill_img)
        centroid_height = calc_centroid_height(cmax, dfcsv) if cmax is not None else np.nan
        avg_height, df = calc_avg_height(fill_img, dfcsv)
        volume = calc_volume(df, camera_height=CAMERA_HEIGHT)  # Corrigido para usar a função de volume
        
        results.append([width, length, centroid_height, avg_height, volume])
    
    return results
