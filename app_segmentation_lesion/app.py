from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import shutil
import cv2
import sys
import numpy as np
import torch
import utils
import detect
import generate_EDSR as gen_EDSR
import tool_SVM
import tool
import generate_feature_map as gen_vgg16

import time 
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm

# Inicialización de modelos y herramientas
device = "cuda"
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

app = Flask(__name__)

# Directorio para guardar las imágenes cargadas
uploads_dir = os.path.join('data', 'images')
os.makedirs(uploads_dir, exist_ok=True)

@app.route("/")
def index():
    return render_template('index.html')


model_vgg16_layer_5 = gen_vgg16.load_vgg16_model()


sr = cv2.dnn_superres.DnnSuperResImpl_create()
path = 'EDSR_x4.pb' # 2, 3, 4
sr.readModel(path)
sr.setModel("edsr",4) # 2, 3, 4


@app.route("/detect_lesion", methods=['POST'])
def detect_lesion():
    start_time = time.time()  # Tiempo de inicio
    ruta_carpeta = 'predicts_RGB_VGG_VER'
    tool.create_new_file(ruta_carpeta)
    tool.create_new_file('crops')
    tool.create_new_file('{}/{}'.format(ruta_carpeta, 'overlay'))
    tool.create_new_file('{}/{}'.format(ruta_carpeta, 'crops_filter'))
    tool.create_new_file('{}/{}'.format(ruta_carpeta, 'marcas_crop'))
    tool.create_new_file('{}/{}'.format(ruta_carpeta, 'marcas_images'))
    tool.create_new_file('{}/{}'.format(ruta_carpeta, 'slic_boundaries'))
    tool.create_new_file('{}/{}'.format(ruta_carpeta, 'file_images_centroide_sample'))
    tool.create_new_file('{}/{}'.format(ruta_carpeta, 'images_bbox'))
    if 'file' not in request.files:
        return jsonify(error="No se proporcionó imagen")

    uploaded_file = request.files['file']

    # Limpiar el directorio de imágenes anteriores
    for filename in os.listdir(uploads_dir):
        file_path = os.path.join(uploads_dir, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

    # Guardamos la nueva imagen en el directorio de 'data/images'
    filepath = os.path.join(uploads_dir, secure_filename(uploaded_file.filename))
    uploaded_file.save(filepath)

    # Procesamiento de la imagen con el nuevo código
    method = 3  # 1: RGB, 2: VGG, 3: RGB and VGG, 4: unet
    confiabilidad = 0.25
    Path = os.path.join('.', 'runs/train/exp/weights/best.pt')
    
    
    # Realizar detección con YOLOv5
    start_yolo_time = time.time()
    bboxes = detect.run(weights=Path, conf_thres=confiabilidad, save_crop=False)
    end_yolo_time = time.time()
    total_yolo_time = end_yolo_time - start_yolo_time
    
    # Inicializar acumuladores de tiempo para EDSR y SAM
    total_edsr_time = 0
    total_extraction_time = 0
    total_sam_time = 0


    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

    start_sam_time = time.time()
    predictor.set_image(img)
    end_sam_time = time.time()
    total_sam_time += end_sam_time - start_sam_time


    h, w, _ = img.shape
    masks_pred = []
    

    # Inicializar img_view_2 y img_bbox antes del bucle
    img_view_2 = img.copy()
    img_bbox = img.copy()

    pixel_counts = []  # Lista para almacenar el recuento de píxeles por máscara



    
    for bbox in bboxes:
        bbox = bbox[0]
        bbox = np.asarray(bbox, dtype=np.int32)
        
        
        x1, y1, x2, y2 = bbox
        
        
        # Realizar recorte
        recorte = img[y1:y2, x1:x2, :]
        h_crop, w_crop, _ = recorte.shape

        # Aplicar EDSR solo si el recorte es menor de 512x512
        if h_crop < 200 and w_crop < 200:
            #cv2.imwrite('crops/crop.jpg', recorte)
            start_edsr_time = time.time()
            recorte_sr = gen_EDSR.SR_EDSR_simplify(recorte, sr)
            end_edsr_time = time.time()
            total_edsr_time += end_edsr_time - start_edsr_time
            
            cv2.imwrite('generate_images_sr_img/crop.jpg', recorte_sr)
            h_crop_sr, w_crop_sr, _ = recorte_sr.shape
            fh = h_crop / h_crop_sr
            fw = w_crop / w_crop_sr
        else:
            recorte_sr = recorte
            cv2.imwrite('generate_images_sr_img/crop.jpg', recorte_sr)
            fh = 1
            fw = 1

        n_segment = 100
        compactness = 10
        sigma = 1 
        threshold = 180
        layer = 5

        if (method == 1 or method == 2 or method == 3):
            start_extraction_time = time.time()
            input_point_all, input_label_all = tool_SVM.prediction_SVM_predict('.', ruta_carpeta, 'generate_images_sr_img', n_segment, compactness, sigma, threshold , layer, [uploaded_file.filename], method, predictor, uploaded_file.filename, model_vgg16_layer_5)
            end_extraction_time = time.time()
            total_extraction_time += end_extraction_time - start_extraction_time

        #actualizacion de coordenadas
        for coord in input_point_all:
            coord[0] = x1 + coord[0]*fw
            coord[1] = y1 + coord[1]*fh

        start_sam_time = time.time()
        mask_pred = tool.segment_image(img, input_point_all, input_label_all, bbox, predictor)
        end_sam_time = time.time()
        total_sam_time += end_sam_time - start_sam_time

        masks_pred.append(mask_pred)

        # Cuenta los píxeles con valor 1 y almacénalo
        pixel_count = np.sum(mask_pred == 1)
        pixel_counts.append(pixel_count)

        # Actualizar img_view_2 y img_bbox dentro del bucle
        img_view_2 = tool.show_points_on_image(img_view_2, input_point_all, input_label_all, complete=True)

    print(f"Tiempo total de YOLOv5: {total_yolo_time} segundos")
    print(f"Tiempo total de EDSR: {total_edsr_time} segundos")
    print(f"Tiempo total de extracción de características: {total_extraction_time} segundos")
    print(f"Tiempo total de SAM: {total_sam_time} segundos")

    img_bbox = tool.draw_bbox_with_numbered_labels(img_view_2, bboxes, color=(255, 0, 0))

    if masks_pred:
        mask_predict_final = tool.unir_mascaras(masks_pred)
        mask_predict_final = np.squeeze(mask_predict_final)
        mask_predict_final_gray = mask_predict_final.astype(np.uint8) * 255
        mask_predict_final = cv2.cvtColor(mask_predict_final_gray, cv2.COLOR_GRAY2RGB)
        img_overlay = tool.overlay_mask(img, mask_predict_final_gray)
        cv2.imwrite('{}/{}/{}'.format(ruta_carpeta, 'overlay', uploaded_file.filename), cv2.cvtColor(img_overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite('{}/{}/mask_{}'.format(ruta_carpeta, 'marcas_images', uploaded_file.filename), mask_predict_final_gray)  # Asumiendo que esta es una imagen en escala de grises, no se necesita conversión
        cv2.imwrite('{}/{}/bbox_points_{}'.format(ruta_carpeta, 'images_bbox', uploaded_file.filename), cv2.cvtColor(img_bbox, cv2.COLOR_RGB2BGR))


        
    # Mover imágenes procesadas al directorio 'static'
    destination_folder = os.path.join('static', 'detect_results')
    for filename in os.listdir(destination_folder):
        file_path = os.path.join(destination_folder, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    
    shutil.move('{}/{}/{}'.format(ruta_carpeta, 'overlay', uploaded_file.filename), destination_folder)
    shutil.move('{}/{}/mask_{}'.format(ruta_carpeta, 'marcas_images',uploaded_file.filename), destination_folder)
    shutil.move('{}/{}/bbox_points_{}'.format(ruta_carpeta, 'images_bbox', uploaded_file.filename), destination_folder)

    pixel_counts = [int(count) for count in pixel_counts]
    end_time = time.time()  # Tiempo de finalización
    processing_time = end_time - start_time  # Calcula el tiempo total
    print(f"Tiempo de procesamiento: {processing_time} segundos")
    
    return jsonify({
            "pixel_counts": pixel_counts,
            "processing_time": processing_time,
            "overlay_path": os.path.join('detect_results', uploaded_file.filename),
            "mask_path": os.path.join('detect_results', 'mask_' + uploaded_file.filename),
            "bbox_path": os.path.join('detect_results', 'bbox_points_'+uploaded_file.filename)
        })

@app.route("/check_results")
def check_results():
    destination_folder = os.path.join('static', 'detect_results')
    image_files = os.listdir(destination_folder)
    print('holi a')
    print(image_files)
    
    if image_files:
        overlay_path = os.path.join('detect_results', image_files[0])
        mask_path = os.path.join('detect_results', 'mask_'+image_files[0])
        bbox_path = os.path.join('detect_results', 'bbox_points_'+image_files[0])
        
        return jsonify({
            "has_images": True,
            "overlay_path": overlay_path,
            "mask_path": mask_path,
            "bbox_path": bbox_path
        })
    else:
        return jsonify({"has_images": False})

@app.route('/download_image/<filename>', methods=['GET'])
def download_image(filename):
    destination_folder = os.path.join('static', 'detect_results')
    return send_from_directory(destination_folder, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)