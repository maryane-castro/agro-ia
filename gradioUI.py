import gradio as gr
from PIL import Image
from classModels.classSAM import SAMModel
from classModels.classYOLO import YOLOModel


# Função para processar a imagem com base no modelo selecionado (SAM ou YOLO)
def process_image(image_path, selected_model, confidence):
    if not image_path:
        return None

    # Se o modelo selecionado for SAM
    if selected_model == "SAM":
        sam_model = SAMModel(
            model_cfg="configs/sam2.1/sam2.1_hiera_b+.yaml",  # Caminho do arquivo de configuração do SAM
            checkpoint_path="/home/nuvenpreto01/Documentos/Github/agro-ia/checkpoints/checkpoints/v2/weights/SAM/checkpoint.pt"  # Caminho do checkpoint
        )
        # Gerar imagem anotada com a máscara
        annotated_image = sam_model.get_annotated_image(image_path)
        return annotated_image

    # Se o modelo selecionado for YOLO
    elif selected_model == "YOLO":
        yolo_model = YOLOModel(
            model_path='checkpoints/checkpoints/v2/weights/YOLO-MEDIUM/runs/segment/train/weights/best.pt',  # Caminho para o modelo YOLO
            conf=confidence
        )
        annotated_image = yolo_model.get_annotated_image(image_path)
        return annotated_image

    return None


# Interface Gradio
with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown("### Input Image")
        gr.Markdown("### Processed Image")
    
    with gr.Row():
        # Componente de upload de imagem
        input_image = gr.Image(type="filepath", label="Upload Image")

        # Componente para exibir a imagem processada
        processed_image = gr.Image(label="Processed Image")

    with gr.Row():
        # Coluna de configurações
        with gr.Column():
            gr.Markdown("**Settings**")
            
            # Componente para selecionar o modelo
            selected_model = gr.Radio(
                choices=["YOLO", "SAM"],  # Agora temos a opção para YOLO também
                value="YOLO",  # Valor padrão
                label="Select Model"
            )

            # Slider para selecionar o valor de confiança
            confidence = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.5,
                step=0.1,
                label="Confidence Threshold"
            )

        # Botão para processar a imagem
        generate_button = gr.Button("Generate", variant="primary")

    # Exemplos de imagens para carregar automaticamente
    examples = [
        "images/depth_280_1712318419_224_030.png",
        "images/frame_000004_png.rf.2162d5c0f0689d91bf25d39c03f97569.jpg",
        "images/teste1.png",
        # Adicione os caminhos para suas imagens de exemplo aqui
    ]
    
    gr.Examples(examples=examples, inputs=[input_image])

    # Conexão do botão com a função de processamento
    generate_button.click(
        fn=process_image,
        inputs=[input_image, selected_model, confidence],  # Entradas da função
        outputs=[processed_image]  # Saídas da função
    )

# Iniciar o Gradio
demo.launch()
