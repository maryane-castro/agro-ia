import sys
import pyzed.sl as sl


def extract_images(svo_path, output_dir):
    # Configurar a inicialização da câmera virtual ZED
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(svo_path)
    init_params.coordinate_units = sl.UNIT.MILLIMETER

    zed = sl.Camera()
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Erro ao abrir o arquivo SVO: {status}")
        exit(1)

    image = sl.Mat()
    frame_count = 0

    # Loop para extrair quadros
    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Para obter uma imagem rgb
            zed.retrieve_image(image, sl.VIEW.LEFT)
            # Para obter uma imagem de profundidade
            #zed.retrieve_image(image, sl.VIEW.DEPTH)
            filename = f"{output_dir}/frame_{frame_count:03}.png"
            image.write(filename)
            print(f"Quadro salvo: {filename}")
            frame_count += 1
        else:
            break

    zed.close()
    print(f"Extração concluída. Total de quadros: {frame_count}")

#if __name__ == "__main__":
#    svo_path = "svos-testes/3_1734629197_329.svo"
#    output_dir = "out_png"
#    extract_images(svo_path, output_dir)
