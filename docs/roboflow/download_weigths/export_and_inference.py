import os
import cv2
import inference
import supervision as sv
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()
os.environ["MODEL_CACHE_DIR"] = "./models_cache"

model_id = os.getenv("MODEL_ID")
api_key = os.getenv("API_KEY")
image_path = "<path_your_image>"

if not model_id:
    raise ValueError("A variável de ambiente 'MODEL_ID' não está configurada.")
if not api_key:
    raise ValueError("A variável de ambiente 'API_KEY' não está configurada.")


model = inference.get_model(model_id, api_key=api_key)
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Imagem não encontrada no caminho: {image_path}")

results = model.infer(image)[0]
detections = sv.Detections.from_inference(results)

mask_annotator = sv.MaskAnnotator()
label_annotator = sv.LabelAnnotator()
annotated_image = mask_annotator.annotate(scene=image, detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 10))
plt.imshow(annotated_image_rgb)
plt.axis('off') 
plt.title("Annotated Image")
plt.show()
