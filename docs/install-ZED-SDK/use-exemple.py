import pyzed.sl as sl
import os

input_svo = "src/input_svo"
output_folder = "src/output_frames"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(input_svo, exist_ok=True)


zed = sl.Camera()
init_params = sl.InitParameters()
init_params.svo_real_time_mode = False  
init_params.set_from_svo_file(input_svo) 

status = zed.open(init_params)
if status != sl.ERROR_CODE.SUCCESS:
    print(f"Erro ao abrir o arquivo SVO: {status}")
    exit()

runtime = sl.RuntimeParameters()
image = sl.Mat()

frame_index = 0
while True:
    if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)
        filename = f"{output_folder}/frame_{frame_index:06d}.png"
        image.write(filename)
        print(f"Quadro salvo: {filename}")
        frame_index += 1
    else:
        break

zed.close()
print("Exportação concluída!")