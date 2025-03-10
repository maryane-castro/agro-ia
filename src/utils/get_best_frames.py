from src.utils.best_frame_selection.best_frame import process_frames

def master_best_frame(inputs_png="output/extracoes/pngs", inputs_csv="output/extracoes/csvs"):
    #inputs_png = "output/extracoes/pngs"  # Contém várias pastas, cada uma com PNGs
    #inputs_csv = "output/extracoes/csvs"  # Contém as mesmas pastas, mas com CSVs correspondentes

    # Diretório de saída
    output_base_dir = "data/processed/melhoresFrames"

    model_path = "models/weights/best.pt"

    # Executa a seleção dos melhores frames
    process_frames(inputs_png, inputs_csv, output_base_dir, model_path)


# 2