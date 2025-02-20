from bestFrame.bestFrame import process_frames

inputs_png = "output/extracoes/pngs"  # Contém várias pastas, cada uma com PNGs
inputs_csv = "output/extracoes/csvs"  # Contém as mesmas pastas, mas com CSVs correspondentes

# Diretório de saída
output_base_dir = "output/melhoresFrames"

model_path = "models/best.pt"

# Executa a seleção dos melhores frames
process_frames(inputs_png, inputs_csv, output_base_dir, model_path)

