from extractsPngCsv.extract_svo_png import extract_images
from extractsPngCsv.extract_svo_csv import extract_depth
import os

svo_directory = "svos-testes"
output_base_dir = "output/extracoes"

svos_teste = [os.path.join(svo_directory, f) for f in os.listdir(svo_directory) if f.endswith(".svo")]

for svo_path in svos_teste:
    svo_name = os.path.splitext(os.path.basename(svo_path))[0]
    output_dir_pngs = os.path.join(output_base_dir, "pngs", svo_name)
    output_dir_csvs = os.path.join(output_base_dir, "csvs", svo_name)
    
    os.makedirs(output_dir_pngs, exist_ok=True)
    os.makedirs(output_dir_csvs, exist_ok=True)
    
    extract_images(svo_path, output_dir_pngs)
    extract_depth(svo_path, output_dir_csvs)
