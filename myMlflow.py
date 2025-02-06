# mlflow ui 


import time
import mlflow
import mlflow.pyfunc
from classModels.classSAM import SAMModel
from classModels.classYOLO import YOLOModel

mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("SAM_YOLO_Experiments")

# Definir o wrapper para o modelo SAM
class SAMModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model_cfg, checkpoint_path):
        self.model = SAMModel(model_cfg, checkpoint_path)

    def predict(self, context, model_input):
        image_path = model_input[0]
        image, masks = self.model.generate_masks(image_path)
        return masks

# Função para logar o modelo SAM
def log_sam_model(sam_model, image_path, model_cfg, checkpoint_path):
    # Medindo o tempo de inferência
    start_time = time.time()
    image, masks = sam_model.generate_masks(image_path)
    inference_time = time.time() - start_time

    # Logar no MLflow
    with mlflow.start_run(run_name="SAM_Model_Logging", nested=True):
        print("Logging SAM model...")
        mlflow.log_param("model_type", "SAM")
        mlflow.log_param("model_config", model_cfg)
        mlflow.log_param("checkpoint", checkpoint_path)
        mlflow.log_metric("num_detections", len(masks))  # Número de máscaras geradas
        mlflow.log_metric("inference_time", inference_time)  # Tempo de inferência
        print(f"Generated {len(masks)} masks logged to MLflow.")
        print(f"Inference time: {inference_time:.4f} seconds")

        # Salvar o modelo SAM no MLflow (usando PyTorch)
        model = sam_model.model  # Assumindo que sam_model tem um atributo 'model' do tipo torch.nn.Module
        mlflow.pytorch.log_model(model, "SAM_Model")  # Logando o modelo PyTorch diretamente

        # Adicionando a descrição de inferência como artefato de texto
        inference_description_sam = """
        # Descrição de Inferência - SAM Model

        O modelo SAM (Segment Anything Model) é utilizado para gerar máscaras de objetos em imagens. A inferência é realizada utilizando um caminho de imagem fornecido, e o modelo gera máscaras para cada objeto detectado na imagem.

        Processo de inferência:
        1. O modelo SAM carrega a configuração e o checkpoint especificados.
        2. A imagem é passada para o modelo.
        3. O modelo gera as máscaras de segmentação para a imagem.
        4. O número de máscaras geradas é registrado junto com o tempo de inferência.

        Resultado esperado:
        - O número de máscaras geradas e o tempo total para a inferência são registrados no MLflow.
        """

        # Salvar a descrição em um arquivo de texto local
        with open("inference_description_sam.txt", "w") as f:
            f.write(inference_description_sam)

        # Agora logue o arquivo como um artefato
        mlflow.log_artifact("inference_description_sam.txt")


# Definir o wrapper para o modelo YOLO
class YOLOModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model_path, name_dict, conf):
        self.model = YOLOModel(model_path, name_dict, conf)

    def predict(self, context, model_input):
        image_path = model_input[0]
        annotated_image, detections_len = self.model.predict(image_path)
        return annotated_image, detections_len

# Função para logar o modelo YOLO
def log_yolo_model(yolo_model, image_path, model_path):
    # Medindo o tempo de inferência
    start_time = time.time()
    annotated_image, detections_len, _ = yolo_model.predict(image_path)
    inference_time = time.time() - start_time

    # Logar no MLflow
    with mlflow.start_run(run_name="YOLO_Model_Logging", nested=True):
        print("Logging YOLO model...")
        mlflow.log_param("model_type", "YOLO")
        mlflow.log_param("confidence_threshold", yolo_model.conf)
        mlflow.log_param("model_path", model_path)
        mlflow.log_metric("num_detections", detections_len)  # Número de detecções
        mlflow.log_metric("inference_time", inference_time)  # Tempo de inferência
        print(f"Inference time: {inference_time:.4f} seconds")
        print("YOLO annotated image logged to MLflow.")

        # Registrar o modelo YOLO
        model_wrapper = YOLOModelWrapper(model_path, yolo_model.name_dictionary, yolo_model.conf)
        mlflow.pyfunc.log_model("YOLO_Model", python_model=model_wrapper)

        # Adicionando a descrição de inferência como artefato de texto
        inference_description_yolo = """
        # Descrição de Inferência - YOLO Model

        O modelo YOLO (You Only Look Once) é utilizado para detectar objetos em imagens. Ele realiza a inferência em tempo real, retornando o número de objetos detectados e suas respectivas classes.

        Processo de inferência:
        1. O modelo YOLO carrega o modelo treinado e a configuração do limiar de confiança.
        2. A imagem é passada para o modelo.
        3. O modelo realiza a detecção de objetos na imagem.
        4. O número de objetos detectados e o tempo total de inferência são registrados no MLflow.

        Resultado esperado:
        - O número de detecções e o tempo total para a inferência são registrados no MLflow.
        """

        # Salvar a descrição em um arquivo de texto local
        with open("inference_description_yolo.txt", "w") as f:
            f.write(inference_description_yolo)

        # Agora logue o arquivo como um artefato
        mlflow.log_artifact("inference_description_yolo.txt")


if __name__ == "__main__":
    sam2_checkpoint = "/home/nuvenpreto01/Documentos/Github/agro-ia/checkpoints/checkpoints/v2/weights/SAM/checkpoint.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    sam_model = SAMModel(model_cfg, sam2_checkpoint)

    image_path = "images/frame_000004_png.rf.2162d5c0f0689d91bf25d39c03f97569.jpg"
    log_sam_model(sam_model, image_path, model_cfg, sam2_checkpoint)

    model_path = 'checkpoints/checkpoints/v2/weights/YOLO-MEDIUM/runs/segment/train/weights/best.pt'
    yolo_model = YOLOModel(
        model_path=model_path,
        conf=0.50
    )

    log_yolo_model(yolo_model, image_path, model_path)
