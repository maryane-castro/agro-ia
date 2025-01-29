import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from scipy.spatial.distance import euclidean
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

class SAMModel:
    def __init__(self, model_cfg, checkpoint_path, device=None):
        """
        Initialize the SAM model.

        :param model_cfg: Path to the SAM model configuration file.
        :param checkpoint_path: Path to the SAM model checkpoint file.
        :param device: Device for computation ('cuda', 'mps', or 'cpu'). If None, it will auto-detect.
        """
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        if self.device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif self.device.type == "mps":
            print(
                "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                "give numerically different outputs and sometimes degraded performance on MPS. "
                "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
            )

        self.model = build_sam2(model_cfg, checkpoint_path, device=self.device, apply_postprocessing=False)
        self.mask_generator = SAM2AutomaticMaskGenerator(self.model)

    def generate_masks(self, image_path):
        """
        Generate masks for the given image and add a confidence score based on the center of the mask.

        :param image_path: Path to the input image file.
        :return: List of generated masks with confidence score.
        """
        image = Image.open(image_path)
        image = np.array(image.convert("RGB"))
        masks = self.mask_generator.generate(image)
        
        image_center = np.array([image.shape[1] // 2, image.shape[0] // 2])  # Centro da imagem

        # Adicionar a pontuação de confiança baseada na distância ao centro da imagem
        for mask in masks:
            # Calcular o centro da máscara (pode ser o centro de sua bounding box)
            mask_center = self._get_mask_center(mask['segmentation'])
            distance = euclidean(image_center, mask_center)
            max_distance = np.linalg.norm(image_center)  # Distância máxima (canto da imagem)
            mask['confidence'] = 1 - (distance / max_distance)  # Quanto mais perto do centro, maior a confiança

        return image, masks

    def _get_mask_center(self, mask):
        """
        Calcula o centro da máscara. Para simplificação, usamos o centro de sua bounding box.

        :param mask: A máscara binária.
        :return: O centro da máscara como (x, y).
        """
        # Encontrar os contornos da máscara e calcular a bounding box
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        center = np.array([x + w // 2, y + h // 2])
        return center

    @staticmethod
    def show_masks(image, masks, borders=True, min_confidence=0.1):
        """
        Display the image with the generated masks overlaid, filtering by confidence, and showing only the most central mask.

        :param image: The input image as a NumPy array.
        :param masks: List of generated masks.
        :param borders: Whether to display borders for the masks.
        :param min_confidence: Minimum confidence threshold to display masks.
        """
        if len(masks) == 0:
            print("No masks generated.")
            return

        # Filtra as máscaras pela confiança
        masks = [mask for mask in masks if mask['confidence'] >= min_confidence]

        if len(masks) == 0:
            print("No masks meet the confidence threshold.")
            return

        # Encontrar a máscara mais central (a que tem maior confiança)
        central_mask = max(masks, key=lambda x: x['confidence'])

        # Exibir a máscara mais central
        m = central_mask['segmentation']
        img_overlay = np.ones((m.shape[0], m.shape[1], 4))
        img_overlay[:, :, 3] = 0  # Inicialmente com fundo transparente

        # Definir a cor da máscara
        color_mask = np.concatenate([np.random.random(3), [0.8]])  # Cor aleatória e opacidade alta
        img_overlay[m] = color_mask

        # Se necessário, desenhar o contorno da máscara
        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            
            #cv2.drawContours(img_overlay, contours, -1, (0, 0, 0, 1), thickness=3)

        # Exibir a imagem original com a máscara sobreposta
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.imshow(img_overlay, alpha=0.6)  # Sobreposição com opacidade ajustada
        plt.axis('off')
        plt.show()



