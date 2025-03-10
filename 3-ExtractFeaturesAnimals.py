
import os
import pandas as pd
from featuresAnimals.extract3 import extract_animal_features

inputs_paste_pngs = "output/melhoresFrames/pngs"
inputs_paste_csvs = "output/melhoresFrames/csvs"
output_dataset_dir = "datasets"

# Cria o diretório do dataset, caso não exista
os.makedirs(output_dataset_dir, exist_ok=True)

# Cria uma lista para armazenar os dados do dataset
dataset = []

# Itera sobre as pastas dentro do diretório de PNGs
for folder_name in os.listdir(inputs_paste_pngs):
    folder_path_png = os.path.join(inputs_paste_pngs, folder_name)
    folder_path_csv = os.path.join(inputs_paste_csvs, folder_name)
    
    if os.path.isdir(folder_path_png):
        # Verifica se o nome da pasta tem o formato esperado (animal_id_peso_real)
        parts = folder_name.split("_")
        
        if len(parts) != 3:
            print(f"Nome da pasta inválido: {folder_name}. Esperado: animal_id_peso_real. Pulando essa pasta.")
            continue  # Pula a pasta se o formato não for válido
        
        # Extrai o nome do animal, ID e peso real do nome da pasta
        animal, id_unico, peso_real = parts
        
        # Itera sobre as imagens dentro da pasta
        for frame_num, png_name in enumerate(sorted(os.listdir(folder_path_png))):
            if png_name.endswith(".png"):
                png_path = os.path.join(folder_path_png, png_name)
                
                # Acha o CSV correspondente
                csv_name = png_name.replace(".png", ".csv")
                csv_path = os.path.join(folder_path_csv, csv_name)
                
                if os.path.exists(csv_path):
                    # Extraí as características do animal
                    [width, length, height_centroid, height_average, volume] = extract_animal_features([png_path], [csv_path])[0]
                    
                    # Verifica se algum valor é negativo
                    if width < 0 or length < 0 or height_centroid < 0 or height_average < 0 or volume < 0:
                        continue  # Pula essa iteração se algum valor for negativo
                    
                    # Adiciona os dados à lista do dataset
                    dataset.append([
                        animal, 
                        id_unico, 
                        frame_num, 
                        width, 
                        length, 
                        height_centroid, 
                        height_average, 
                        volume, 
                        peso_real
                    ])

# Cria o DataFrame do Pandas com os dados coletados
df = pd.DataFrame(dataset, columns=["Animal", "ID", "Frame", "Width", "Length", "Height_Centroid", "Height_average", "Volume", "Real"])

# Salva o DataFrame como um arquivo CSV
dataset_csv_path = os.path.join(output_dataset_dir, "dataset1.csv")
df.to_csv(dataset_csv_path, index=False)

print(f"Dataset salvo em: {dataset_csv_path}")
