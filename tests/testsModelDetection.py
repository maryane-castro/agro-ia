import sys
import os
import time
import logging
import json
import shutil
from glob import glob
from roboflow import Roboflow
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from classModels.classYOLODetection import YOLOInference  


def delete_existing_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"🔴 Arquivo existente removido: {file_path}")


log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)


log_file = os.path.join(log_dir, "performance_test_detec.log")


delete_existing_file(log_file)


logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


rf = Roboflow(api_key="UmrhwHmxixp7hrYrUSIB")
project = rf.workspace("agroia").project("dataset-v1-s3fwi")
version = project.version(9)
dataset = version.download("yolov11")


image_folder = os.path.join(dataset.location, "test", "images")  
output_folder = "output/"
os.makedirs(output_folder, exist_ok=True)


yolo_model = YOLOInference(
    model_path='checkpoints/checkpoints/v2/weights/YOLO-MEDIUM/runs/segment/train/weights/best.pt',
    conf=0.80
)

MAX_IMAGES = 100  
image_paths = (glob(os.path.join(image_folder, "*.jpg")) + glob(os.path.join(image_folder, "*.png")))[:MAX_IMAGES]


print(f"🔎 Imagens encontradas: {len(image_paths)}")
print(f"🔎 Caminho das imagens: {image_paths}")

results = []

for image_path in image_paths:
    print(f"🔄 Processando a imagem: {image_path}")
    start_time = time.time()

    try:
        
        annotated_image, detections, confidences = yolo_model.predict(image_path)
        inference_time = time.time() - start_time  
        
        
        output_path = os.path.join(output_folder, os.path.basename(image_path))
        
        
        yolo_model.save_annotated_image(annotated_image, output_path)

        
        if isinstance(confidences, (int, float)):
            confidences = [confidences]  

        
        log_info = {
            "image": os.path.basename(image_path),
            "detections": len(detections),
            "inference_time": inference_time,
            "confidences": confidences  
        }
        results.append(log_info)

        logging.info(f"Processed {image_path} | Detections: {len(detections)} | Time: {inference_time:.4f}s | Confidences: {confidences}")

    except Exception as e:
        logging.error(f"Error processing {image_path}: {str(e)}")

print(f"📊 Resultados: {results}")

metrics_file = os.path.join(log_dir, "performance_metrics_detec.json")


delete_existing_file(metrics_file)


if results:
    with open(metrics_file, "w") as f:
        json.dump(results, f, indent=4)

print(f"✅ Processamento concluído. Logs e métricas salvos em: {log_file}, {metrics_file}")


print("🗑️ Removendo o dataset baixado...")
try:
    shutil.rmtree(dataset.location)  
    print(f"✅ Dataset removido com sucesso: {dataset.location}")
except Exception as e:
    print(f"❌ Erro ao remover o dataset: {str(e)}")
