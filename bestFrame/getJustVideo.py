import os
import subprocess
import cv2

# path video
input_folder = "./bestFrame/svos/"
output_folder = "./bestFrame/out_video"


os.makedirs(output_folder, exist_ok=True)
os.makedirs(input_folder, exist_ok=True)


svo_files = [f for f in os.listdir(input_folder) if f.endswith(".svo")]

if not svo_files:
    print("Nenhum arquivo .svo encontrado na pasta de entrada.")
    exit()


process_svo_script = """
import pyzed.sl as sl
import os
import sys

input_svo = sys.argv[1]
output_folder = sys.argv[2]
start_index = int(sys.argv[3])
max_frames = int(sys.argv[4])


zed = sl.Camera()

init_params = sl.InitParameters()
init_params.svo_real_time_mode = False
init_params.set_from_svo_file(input_svo)

status = zed.open(init_params)
if status != sl.ERROR_CODE.SUCCESS:
    print(f"Erro ao abrir o arquivo SVO '{input_svo}': {status}")
    sys.exit(1)

runtime = sl.RuntimeParameters()
image = sl.Mat()

frame_index = start_index
current_frame = 0
while current_frame < max_frames:
    if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)
        filename = f"{output_folder}/frame_{frame_index:06d}.png"
        image.write(filename)
        print(f"Quadro salvo: {filename}")
        frame_index += 1
        current_frame += 1
    else:
        print(f"Fim do arquivo SVO detectado: {input_svo}")
        break

zed.close()
print(f"Exportação concluída para o arquivo: {input_svo}")
print(frame_index)
"""


script_path = "process_svo_temp.py"
with open(script_path, "w") as f:
    f.write(process_svo_script)


max_frames = 40


frame_index = 0


for svo_file in svo_files:
    input_svo_path = os.path.join(input_folder, svo_file)
    print(f"\nIniciando processamento do arquivo: {input_svo_path}")

    try:
        
        result = subprocess.run(
            ["python3", script_path, input_svo_path, output_folder, str(frame_index), str(max_frames)],
            check=True,
            capture_output=True,
            text=True
        )
        
        
        for line in result.stdout.splitlines():
            if line.strip().isdigit():
                frame_index = int(line.strip())
                break

    except subprocess.CalledProcessError as e:
        print(f"Erro ao processar '{svo_file}': {e}")
        print(f"Saída do erro: {e.stderr}")
    except Exception as ex:
        print(f"Erro inesperado: {ex}")


print("Iniciando a criação do vídeo...")


video_output = os.path.join(output_folder, "output_video.mp4")  


image_files = [f for f in os.listdir(output_folder) if f.endswith(".png")]
image_files.sort()  


if not image_files:
    print("Nenhum quadro encontrado para criar o vídeo.")
    exit()


first_frame = cv2.imread(os.path.join(output_folder, image_files[0]))
height, width, _ = first_frame.shape


fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
video_writer = cv2.VideoWriter(video_output, fourcc, 30, (width, height))  


for image_file in image_files:
    img_path = os.path.join(output_folder, image_file)
    frame = cv2.imread(img_path)
    video_writer.write(frame)  


video_writer.release()

print(f"Vídeo criado com sucesso em: {video_output}")


for image_file in image_files:
    img_path = os.path.join(output_folder, image_file)
    os.remove(img_path)

print("Imagens apagadas da pasta de saída.")


os.remove(script_path)

print("Processamento de todos os arquivos concluído!")
