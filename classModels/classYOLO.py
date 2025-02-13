import supervision as sv
from ultralytics import YOLO
import numpy as np
from PIL import Image

class YOLOModel:
    def __init__(self, model_path, name_dictionary=None, conf=0.5):
        """
        Inicializa o modelo YOLO.

        :param model_path: Caminho para o arquivo de pesos do modelo YOLO.
        :param name_dictionary: Dicionário opcional para renomear classes detectadas.
        :param conf: Limite de confiança para detecções.
        """
        self.model = YOLO(model_path)
        self.name_dictionary = name_dictionary or {}
        self.conf = conf
        self.mask_annotator = sv.MaskAnnotator()
        self.label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK, text_position=sv.Position.CENTER)

    def _rename_classes(self, detections):
        """
        Renomeia as classes das detecções usando o dicionário fornecido.

        :param detections: Objeto de detecções do Supervision.
        :return: Detecções com classes renomeadas.
        """
        detections.data["class_name"] = [
            self.name_dictionary.get(class_name, class_name) 
            for class_name in detections.data["class_name"]
        ]
        return detections

    def predict(self, image_path):
        """
        Faz a predição em uma imagem.

        :param image: Imagem no formato PIL ou NumPy.
        :return: Imagem anotada com máscaras e labels, número de detecções e as confidências.
        """

        image = Image.open(image_path).convert("RGB")
        if isinstance(image, Image.Image):
            image = np.array(image)

        result = self.model.predict(image, conf=self.conf)[0]
        detections = sv.Detections.from_ultralytics(result)
        confidences = detections.confidence.tolist() if len(detections) > 0 else []

        if self.name_dictionary:
            detections = self._rename_classes(detections)

        annotated_image = image.copy()
        self.mask_annotator.annotate(annotated_image, detections=detections)
        self.label_annotator.annotate(annotated_image, detections=detections)

        return annotated_image, len(detections), confidences

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

    def get_annotated_image(self, image_path):
        """
        Retorna a imagem anotada para ser exibida no Gradio.

        :param image_path: Caminho da imagem de entrada.
        :return: Imagem anotada no formato PIL.Image para o Gradio.
        """
        # Chama a função predict para processar a imagem
        annotated_image, _, _ = self.predict(image_path)

        # Converte de NumPy para PIL (se necessário)
        return Image.fromarray(annotated_image)
