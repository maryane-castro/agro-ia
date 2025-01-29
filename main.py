"""
INFERENCE MODELS
"""

from classModels.classSAM import SAMModel
from classModels.classYOLO import YOLOModel


if __name__ == "__main__":

    image_path = "images/frame_000004_png.rf.2162d5c0f0689d91bf25d39c03f97569.jpg"


    # sam
    absolute_path_sam2_checkpoint = "/home/nuvenpreto01/Documentos/Github/sam2-inference/checkpoints/v2/weights/SAM/checkpoint.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    sam_model = SAMModel(model_cfg, absolute_path_sam2_checkpoint)
    image, masks = sam_model.generate_masks(image_path)
    print(f"Generated {len(masks)} masks.")
    SAMModel.show_masks(image, masks) # retorna a mais central caso detecte mais máscaras


    # yolo
    yolo_model = YOLOModel(
        model_path='checkpoints/v2/weights/YOLO-MEDIUM/runs/segment/train/weights/best.pt',
        conf=0.50
    )
    annotated_image, detections_len = yolo_model.predict(image_path)
    yolo_model.plot(annotated_image)