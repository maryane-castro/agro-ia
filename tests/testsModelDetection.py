import sys
import os
import time
import logging
import json
import shutil
from glob import glob
from roboflow import Roboflow
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from classModels.classYOLO import YOLOModel

# Configuração do logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename="logs/performance_test.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Carregar o projeto e dataset do Roboflow
rf = Roboflow(api_key="UmrhwHmxixp7hrYrUSIB")
project = rf.workspace("agroia").project("dataset-v1-s3fwi")
version = project.version(9)
dataset = version.download("yolov11")

# Caminho da pasta de imagens
image_folder = os.path.join(dataset.location, "test", "images")  
output_folder = "output/"
os.makedirs(output_folder, exist_ok=True)

# Instanciando o modelo YOLO
yolo_model = YOLOModel(
    model_path='checkpoints/checkpoints/v2/weights/YOLO-MEDIUM/runs/segment/train/weights/best.pt',
    conf=0.50
)

MAX_IMAGES = 100  # Defina o máximo de imagens a processar
image_paths = (glob(os.path.join(image_folder, "*.jpg")) + glob(os.path.join(image_folder, "*.png")))[:MAX_IMAGES]

# Verificar se as imagens foram encontradas
print(f"🔎 Imagens encontradas: {len(image_paths)}")
print(f"🔎 Caminho das imagens: {image_paths}")

results = []

for image_path in image_paths:
    print(f"🔄 Processando a imagem: {image_path}")
    start_time = time.time()

    try:
        # Faz a predição
        annotated_image, detections_len = yolo_model.predict(image_path)
        inference_time = time.time() - start_time  # Tempo de inferência
        
        # Salvar a imagem anotada na pasta 'output'
        output_path = os.path.join(output_folder, os.path.basename(image_path))
        
        # Usar o método save_annotated_image para salvar a imagem
        yolo_model.save_annotated_image(annotated_image, output_path)

        # Log dos resultados
        log_info = {
            "image": os.path.basename(image_path),
            "detections": detections_len,
            "inference_time": inference_time
        }
        results.append(log_info)

        logging.info(f"Processed {image_path} | Detections: {detections_len} | Time: {inference_time:.4f}s")

    except Exception as e:
        logging.error(f"Error processing {image_path}: {str(e)}")

# Verificar se resultados foram preenchidos
print(f"📊 Resultados: {results}")

# Salvar métricas em arquivo JSON
if results:
    with open(os.path.abspath("logs/performance_metrics.json"), "w") as f:
        json.dump(results, f, indent=4)

print("✅ Processamento concluído. Logs e métricas salvos.")

# Remover o dataset baixado
print("🗑️ Removendo o dataset baixado...")
try:
    shutil.rmtree(dataset.location)  # Remove a pasta do dataset
    print(f"✅ Dataset removido com sucesso: {dataset.location}")
except Exception as e:
    print(f"❌ Erro ao remover o dataset: {str(e)}")
