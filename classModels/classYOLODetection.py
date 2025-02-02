import supervision as sv
from ultralytics import YOLO
import numpy as np
from PIL import Image

class YOLOInference:
    def __init__(self, model_path, conf=0.5):
        """
        Inicializa o modelo YOLO.

        :param model_path: Caminho para o arquivo de pesos do modelo YOLO.
        :param conf: Limite de confiança para detecções.
        """
        self.model = YOLO(model_path)
        self.conf = conf
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK)

    def predict(self, image_path):
        """
        Faz a predição em uma imagem.

        :param image_path: Caminho para a imagem.
        :return: Imagem anotada, número de detecções e confianças das detecções.
        """
        image = Image.open(image_path)
        if isinstance(image, Image.Image):
            image = np.array(image)

        result = self.model.predict(image, conf=self.conf)[0]
        detections = sv.Detections.from_ultralytics(result)

        # Extrair as confianças corretamente
        confidences = detections.confidence.tolist() if len(detections) > 0 else []

        # Verificar se confidences é um valor único (int ou float) e converter
        if isinstance(confidences, (int, float)):
            confidences = [confidences]  # Garantir que seja uma lista

        annotated_image = image.copy()
        annotated_image = self.box_annotator.annotate(annotated_image, detections=detections)
        annotated_image = self.label_annotator.annotate(annotated_image, detections=detections)

        return annotated_image, detections, confidences


    def plot(self, image, size=(10, 10)):
        """
        Exibe a imagem anotada.

        :param image: Imagem anotada.
        :param size: Tamanho da figura para plotagem.
        """
        sv.plot_image(image, size=size)
    
    def save_annotated_image(self, annotated_image, save_path):
        """
        Salva a imagem anotada em um caminho especificado.

        :param annotated_image: A imagem anotada a ser salva.
        :param save_path: O caminho onde a imagem será salva.
        """
        annotated_image_pil = Image.fromarray(annotated_image)  # Converte de numpy para PIL
        annotated_image_pil.save(save_path)
        print(f"Imagem salva em: {save_path}")
