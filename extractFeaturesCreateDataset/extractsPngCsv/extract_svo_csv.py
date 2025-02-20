# modificado

import sys
import pyzed.sl as sl
import csv
import os
import numpy as np

def extract_depth(svo_path, output_dir):
    # Configurar a inicialização da câmera virtual ZED
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(svo_path)
    init_params.coordinate_units = sl.UNIT.MILLIMETER

    zed = sl.Camera()
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Erro ao abrir o arquivo SVO: {status}")
        exit(1)

    frame = sl.Mat()
    frame_count = 0

    # Criar pasta de saída, se não existir
    os.makedirs(output_dir, exist_ok=True)

    # Loop para extrair quadros
    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_measure(frame, sl.MEASURE.DEPTH)
            depth_data = frame.get_data()

            # Verifica se depth_data é um ndarray 2D
            if isinstance(depth_data, np.ndarray) and depth_data.ndim == 2:
                # Substituir valores NaN por 0
                depth_data = np.nan_to_num(depth_data, nan=0.0).astype(np.float32)  # Substitui NaN por 0 e converte para float32
                
                filename = f"{output_dir}/frame_{frame_count:03}.csv"
                
                # Salvar os dados usando a biblioteca CSV
                with open(filename, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    for row in depth_data:
                        writer.writerow(row)
                
                print(f"Quadro salvo: {filename}")
                frame_count += 1
            else:
                print(f"Erro: dados inesperados no frame {frame_count} - Tipo: {type(depth_data)}, Dimensões: {depth_data.shape}")

        else:
            break

    zed.close()
    print(f"Extração concluída. Total de quadros: {frame_count}")

#if _#_name__ == "__main__":
    ##if len(sys.argv) != 3:
    ##    print("Uso: python extract_depth.py <caminho_para_arquivo.svo> <pasta_de_saida>")
    ##    exit(1)
##
    ##svo_path = "svos-sc/9_1734629366_355.svo"
    ##output_dir = "out/csv/"
    ##extract_depth(svo_path, output_dir)
