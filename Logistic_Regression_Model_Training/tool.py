# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 23:15:25 2022

@author: Nicolas
"""
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import pandas as pd
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import shutil
from scipy import ndimage
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import extractor_features_rgb_and_vgg16 as vector_slic_rgb_vgg16
import extractor_features_vgg16 as vector_slic_vgg16
import extractor_features_rgb as vector_slic_rgb
import csv
from tqdm import tqdm
from time import time
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import learning_curve
import detect_noise
from matplotlib.patches import Polygon
from scipy import stats
# from skimage.feature import greycomatrix, graycoprops
# from skimage.feature import graycomatrix, graycoprops
from numpy.random import choice
from scipy.stats import multivariate_normal
'''
<-----TOOL General----->
'''


def copy_images_from_csv(csv_path, images_folder, output_folder):
    with open(csv_path, "r") as csv_file:
        reader = csv.reader(csv_file)
        total_files = sum(1 for line in csv_file)  # Contamos el número total de archivos en el CSV
        csv_file.seek(0)  # Volvemos al principio del archivo CSV
        with tqdm(total=total_files, desc="Copiando imágenes") as pbar:
            for row in reader:
                filename = row[0]
                source_path = os.path.join(images_folder, filename)
                target_path = os.path.join(output_folder, filename)
                shutil.copy(source_path, target_path)
                pbar.update(1)  # Actualizamos la barra de progreso después de copiar cada archivo


def create_file_data(output_folder_images, output_folder_masks, name_csv):
    images_folder = r'C:\Users\nicol\OneDrive\Escritorio\Monitoring_of_Wound\SELECT_DATA\Data_procesada_delete_specular\all\images'
    masks_folder = r'C:\Users\nicol\OneDrive\Escritorio\Monitoring_of_Wound\SELECT_DATA\Data_procesada_delete_specular\all\masks'
    ruta_csv = r'C:\Users\nicol\OneDrive\Escritorio\Monitoring_of_Wound\SELECT_DATA\Data_procesada_delete_specular\all'
    # name_csv = '200_images.csv'
    csv_path = os.path.join(ruta_csv, name_csv)
    copy_images_from_csv(csv_path, images_folder, output_folder_images)
    copy_images_from_csv(csv_path, masks_folder, output_folder_masks)


def load_data(path_img, path_mask):
    is_path = isinstance(path_img, str) and isinstance(path_mask, str)

    if is_path:
        y_val = encoder(path_img)
        y_val_mask = encoder(path_mask, b_mask=False)
    else:
        y_val = path_img
        y_val_mask = path_mask

    return y_val, y_val_mask


def pre_procesing(I):

    # formato RGB
    # I = I[:,:,(2,1,0)]

    bins_edges_min_max = [0, 256]
    num_bins = 256
    bin_count, bins_edges = np.histogram(I, num_bins, bins_edges_min_max)

    pdf = bin_count / np.sum(bin_count)
    cdf = np.cumsum(pdf)
    f_eq = np.round(cdf * 300).astype(int)

    I_eq = f_eq[I]

    I_eq = np.array(I_eq, dtype=np.uint8)

    return I_eq
"""

def encoder(PATH, b_mask=False):
    X = []
    names = os.listdir(PATH)
    names.sort()
    for i in range(len(names)):
        img_path = "{}/{}".format(PATH, names[i])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if b_mask:
            img = pre_procesing(img)
        X.append([img])
    X = np.array(X)
    X = np.squeeze(X)
    return X
"""
def encoder(PATH, b_mask=False):
    X = []
    names = os.listdir(PATH)
    names.sort()
    X = np.empty(len(names), dtype=object)
    for i in range(len(names)):
        img_path = "{}/{}".format(PATH, names[i])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if b_mask:
            img = pre_procesing(img)
        X[i] = img
   
    #X = np.array(X)
    X = np.squeeze(X)
    return X

def new_file(PATH, file, i):
    files = os.listdir(PATH)
    if file in files and i == 0:
        shutil.rmtree('{}/{}'.format(PATH, file))
        os.mkdir('{}/{}'.format(PATH, file))
    else:
        if i == 0:
            os.mkdir('{}/{}'.format(PATH, file))


def calculate_iou(img_ref, img_sp):
    img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    img_sp = cv2.cvtColor(img_sp, cv2.COLOR_BGR2GRAY)

    _, img_ref = cv2.threshold(img_ref, 127, 255, cv2.THRESH_BINARY)
    _, img_sp = cv2.threshold(img_sp, 127, 255, cv2.THRESH_BINARY)

    # Calcular la intersección y la unión de las dos imágenes
    intersection = cv2.bitwise_and(img_ref, img_sp)
    union = cv2.bitwise_or(img_ref, img_sp)

    # Calcular el IoU
    iou = cv2.countNonZero(intersection) / cv2.countNonZero(union)
    # print("Intersection over Union (IoU):", iou)
    return iou

# Convertir las imágenes a binarias (0 y 255)


'''
<-----TOOL for SLIC----->
'''


def centroide(masks, segments_slic):
    # Centoride
    x_c = []
    y_c = []
    for n in range(np.max(segments_slic) + 1):
        cen = ndimage.center_of_mass(masks[n])
        x_c.append(int(cen[1]))
        y_c.append(int(cen[0]))
    return x_c, y_c


def create_img(img, mask_ref, segments_slic, clas, path_out, wound, name_id, predict=False, gc=False, sam=False, mask_sam=[]):
    if predict:
        aux_bool = False
        mask_sp, mask_edge_sp, mask_edge_manual, img_comparative = labels_visual(img, mask_ref, segments_slic, clas)

        m_prueba = mask_sp[:, :, 0]
        a = np.where(m_prueba == 255)

        if gc:
            if a[0].size != 0:
                aux_bool = True
                mask_sp = grub_cut(img, mask_sp)  # la imagen es procesada por
            else:
                aux_bool = True
        if sam:
            if a[0].size != 0:
                mask_sp = mask_sam
        images = [mask_sp, mask_ref, mask_edge_sp, mask_edge_manual]
        labels = ['mask SP', 'mask Manual', 'Edge SP', 'Edge Manual']
        iou = save_img(images, labels, path_out, wound, name_id, predict=True)
        if aux_bool:
            return iou, mask_sp
        else:
            return iou
    else:
        mask_sp, mask_edge_sp, mask_edge_manual, img_comparative = labels_visual(img, mask_ref, segments_slic, clas)

        images = [mask_sp, mask_ref, mask_edge_sp, mask_edge_manual]
        labels = ['mask SP', 'mask Manual', 'Edge SP', 'Edge Manual']
        save_img(images, labels, path_out, wound, name_id)


def create_masks(img, segments_slic):
    superPixels = []
    masks = []
    for i in range(np.max(segments_slic) + 1):
        superPixels.append(np.where(segments_slic == i))
        mask = np.zeros(img.shape, dtype="uint8")
        mask[superPixels[i]] = 255
        masks.append(mask[:, :, 0])
    return masks, superPixels


# Extraer caracteristicas de color para una sola imagen
def feature_color(img, ch, masks, sup_pxls, segments_slic):
    var_ch = []
    me_ch = []
    # as_ch = []
    # contrast_ch = []

    for i in range(np.max(segments_slic) + 1):
        img_ch = img[:, :, ch][sup_pxls[i]]
        var_ch.append(np.var(img_ch))
        me_ch.append(np.mean(img_ch))
        # mean_ch = np.mean(img_ch)
        # diff_sq = (img_ch - mean_ch) ** 2
        # contrast = np.sqrt(np.mean(diff_sq))
        # contrast_ch.append(contrast)

    return var_ch, me_ch


def detector(mask_ref, superPixels, segments_slic, mean_min):
    clas = []  # se guardara el número de superpixel que contiene lesión
    wound = []  # Se etiquetara al superpixel
    for i in range(np.max(segments_slic) + 1):
        if np.mean(mask_ref[superPixels[i]]) > mean_min:
            clas.append(i)
            wound.append(1)
        else:
            wound.append(0)
    ctd_sup = len(clas)  # cantidad de superpixeles necesarios para cubrir la lesión
    return clas, wound, ctd_sup

# función solo para crear mascara con superpixeles


def create_mask_sp(img, segments_slic, clas):
    marcadores = []
    for i in clas:
        marcadores.append(np.where(segments_slic == i))

    # creamos la máscara usando superpixeles
    mask_sp = np.zeros(img.shape, dtype="uint8")
    for idx in range(len(marcadores)):
        mask_sp[marcadores[idx]] = 255
    return mask_sp

# Función para extraer imagen por superpixeles y comparación
# Extraemos todos los pixeles que contienen la lesión


def labels_visual(img, mascara, segments_slic, clas):

    marcadores = [np.where(segments_slic == i) for i in clas]

    # creamos la máscara usando superpixeles
    # creamos la máscara usando superpixeles
    mask_sp = np.zeros(img.shape, dtype="uint8")
    for idx in range(len(marcadores)):
        mask_sp[marcadores[idx]] = 255

    # contorno de mascara con sp
    S = cv2.Canny(mask_sp, 250, 255)
    # contorno de mascara
    Q = cv2.Canny(mascara, 250, 255)

    # Graficando contorno de mascara manual
    mask_edge_manual = mark_boundaries(img, segments_slic)[:, :, (2, 1, 0)]

    mask_edge_manual[Q == 255] = [255, 0, 0]
    # Graficando contorno de mascara creada con superpixeles
    mask_edge_sp = mark_boundaries(img, segments_slic)[:, :, (2, 1, 0)]
    mask_edge_sp[S == 255] = [0, 0, 255]
    img_comparative = mask_edge_manual.copy()
    img_comparative[S == 255] = [0, 0, 255]

    return mask_sp, mask_edge_sp, mask_edge_manual, img_comparative


def create_df(name_c1, name_c2, data_c1, data_c2):
    df = pd.DataFrame()
    datos = {
        "{}".format(name_c1): data_c1,
        "{}".format(name_c2): data_c2
    }
    df = pd.DataFrame(datos)
    return df


def save_pd(df, file, id):
    path = "/content/drive/MyDrive/FootUlcerSegmentationChallenge/Segmentation_Estadistical/Evaluaciones/{}".format(file)
    df.to_csv("{}/prueba_{}.csv".format(path, id), header=True, index=False)


def create_df_feature_color(var_, med_, label, names, x_c_, y_c_):
    # ingresar la imágen en formator RGB
    df = pd.DataFrame()
    datos = {
        'N_img': names,
        'x_c': x_c_,
        'y_c': y_c_
    }
    # "R-G-B"
    for i in range(len(var_)):
        datos["var_ch_{}".format(i + 1)] = var_[i]
    for i in range(len(var_)):
        datos["mean_ch_{}".format(i + 1)] = med_[i]

    datos["Wound"] = label

    df = pd.DataFrame(datos)
    return df


def contador_pxls(segments_slic):
    return np.bincount(segments_slic.flat)


def flatten(arr):
    return [element for sublist in arr for element in sublist]


def save_img(images, labels, path_out, wound, name_id, predict=False):
    # visualizando
    r, c = 2, 2
    cont = 0
    fig = plt.figure(figsize=(5 * c, 5 * r))
    for _r in range(r):
        for _c in range(c):
            plt.subplot(r, c, _r * c + _c + 1)
            img = images[cont]
            img = img[:, :, [2, 1, 0]]
            img_aux = img.astype(int)
    # img_aux = img_aux*255
            plt.imshow(img)
            label = labels[cont]
            plt.title(label)
            plt.axis(False)
            cont += 1
    iou = calculate_iou(images[1], images[0])
    if predict:
        name_id_arr = name_id.split('_')
        title_id = name_id_arr[-1]
        fig.suptitle("{}\n\n IoU = {:.2f} and N_spxl = {}".format(title_id, iou, len(wound)))
        plt.savefig("{}/{}.jpg".format(path_out, name_id))
        plt.close()
        return iou

    else:
        fig.suptitle("IoU = {:.2f} and N_spxl = {}".format(iou, len(wound)))
        plt.savefig("{}/{}.jpg".format(path_out, name_id))
        plt.close()


'''
<-----TOOL for traning----->
'''


def separate_data_train(df):
    # Loading data
    # df = pd.read_csv('{}'.format(name_feature_csv))
    # df.head()

    del (df['N_img'], df['Cantidad'])
    # df.head()

    """## Scaling data"""

    array = df.values
    # separate array into input and output components

    lim_in = 0
    lim_sup = array.shape[1] - 1

    X = array[:, lim_in:lim_sup]

    # Rescale
    scaler = MaxAbsScaler().fit(X)  # applicate (positive scale [0,1]) for now
    rescaledX = scaler.transform(X)

    # Create of dateframe
    df2 = pd.DataFrame()
    df2 = pd.DataFrame(rescaledX)

    # Addd one column of labels
    df2['Wound'] = df.Wound

    # Rename the columnos
    df2.columns = df.columns

    df2.head()

    # df = df[0:int(df.shape[0]*5/100)]
    # df2.shape

    """### **Analyzing data**"""

    # We visualize the distribution of the data
    '''
    fig = plt.figure(figsize = (20,20))
    ax = fig.gca()
    df2.hist(ax=ax)
    plt.show()
    # Correlatioin of the features
    correlation = df2.corr()
    plt.figure(figsize=(20,20))
    ax = sns.heatmap(correlation,vmax=1,square=True,annot=True,cmap='viridis')
    plt.title('Matriz de correlación')
    plt.show()
    '''
    """## Creando SVM"""

    # Train Data
    # X = df.drop(['Wound','N_img'], axis='columns')
    X = df2.drop(['Wound'], axis='columns')
    y = df2.Wound
    # X = df.drop(['Wound'], axis='columns')
    # y = df.Wound

    return df2


#############################################################
# Prediction
#############################################################
def detector_predict(wound, segments_slic):
    return np.where(wound == 1)[0]


def separate_data_train(df):
    # Loading data
    # df.head()

    del (df['N_img'])
    # df.head()

    """## Scaling data"""

    array = df.values
    # separate array into input and output components

    lim_in = 0
    lim_sup = array.shape[1] - 1

    X = array[:, lim_in:lim_sup]

    # Rescale
    scaler = MaxAbsScaler().fit(X)
    rescaledX = scaler.transform(X)

    # Create of dateframe
    df2 = pd.DataFrame()
    df2 = pd.DataFrame(rescaledX)

    # Addd one column of labels
    df2['Wound'] = df.Wound

    # Rename the columnos
    df2.columns = df.columns

    # df2.head()

    # df = df[0:int(df.shape[0]*5/100)]
    # df2.shape

    """### **Analyzing data**"""

    # We visualize the distribution of the data
    '''
    fig = plt.figure(figsize = (20,20))
    ax = fig.gca()
    df2.hist(ax=ax)
    plt.show()

    # Correlatioin of the features

    correlation = df2.corr()
    plt.figure(figsize=(20,20))
    ax = sns.heatmap(correlation,vmax=1,square=True,annot=True,cmap='viridis')
    plt.title('Matriz de correlación')
    plt.show()
    '''
    """## Creando SVM"""

    # Train Data
    # X = df.drop(['Wound','N_img'], axis='columns')
    X = df2.drop(['Wound'], axis='columns')
    y = df2.Wound
    # X = df.drop(['Wound'], axis='columns')
    # y = df.Wound
    return X, y


def encoder_files(PATH):
    # lista los elementos que contiene la carpeta con los experimentos
    return os.listdir(PATH)


def split_dot(lista):
    before_dot = [name.split('.')[0] for name in lista]
    return before_dot


def train_LogisticRegression(path_out, name_training, file_train, test_n):
    pass


def train_SVM(path_out, name_training, file_train, test_n):
    # arreglo de modelos entrenados
    models = []
    # ruta donde se encuentran las imagenes de entrenamiento, i.e. todo los resultados que entregó los experimentos
    file_train = '{}/{}'.format(path_out, file_train)
    # cargando los datos
    data_train = encoder_files(file_train)
    # extrae el nombre de la carpeta
    data_whithout_dot = split_dot(data_train)
    # se cargan los datos
    file_save_data = 'Data_train'

    # temporal:
    obs = []

    tiempo_train = []
    predictions_arr = []
    percentage_arr = []
    res_arr = []
    len_test = []
    # itera sobre cada los datos que hay en cada experimento 3 RGB, VGG y RGB_VGG
    for i in range(len(data_train)):
        tiempo_inicial = time()
        new_file(path_out, name_training, i)
        new_file('{}/{}'.format(path_out, name_training), file_save_data, i)
        # path_save = "{}/{}/{}".format(path_out,name_training, 'Model_'+data_whithout_dot[i])
        # os.mkdir(path_save)

        # Loading data
        filename = os.path.join(file_train, data_train[i])
        df = pd.read_csv('{}'.format(filename), index_col=0)
        obs.append(df)
        # X feature vectors
        X, y = separate_data_train(df)
        # se utilizan los mismos datos para cada experimento
        if i == 0:
            # Model Training
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test_n))
            # I have save theese files
            id_X_train = X_train.index
            id_y_train = y_train.index
            id_X_test = X_test.index
            id_y_test = y_test.index
            X_train.to_csv("{}/{}/{}/X_train.csv".format(path_out, name_training, file_save_data))
            X_test.to_csv("{}/{}/{}/X_test.csv".format(path_out, name_training, file_save_data))
            y_train.to_csv("{}/{}/{}/y_train.csv".format(path_out, name_training, file_save_data))
            y_test.to_csv("{}/{}/{}/y_test.csv".format(path_out, name_training, file_save_data))
        else:
            # la siguiente iteracion extrae lo que está arriba no lo hace nuevamente
            X_train = X.loc[id_X_train]
            y_train = y.loc[id_y_train]
            X_test = X.loc[id_X_test]
            y_test = y.loc[id_y_test]
        # se entrena el modelo
        model = SVC(kernel='linear', probability=True)

        # filtro
        X_train = X_train.fillna(0)

        model.fit(X_train, y_train)
        # Creamos una gráfica para mostrar la función de costo en cada iteración del entrenamiento
        train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)

        train_scores_mean = train_scores.mean(axis=1)
        test_scores_mean = test_scores.mean(axis=1)

        plt.plot(train_sizes, train_scores_mean, label='Training accuracy')
        plt.plot(train_sizes, test_scores_mean, label='Validation accuracy')
        plt.xlabel('Training examples')
        plt.ylabel('Accuracy')
        plt.legend(loc='best')
        plt.savefig('{}/svm_training_curve_{}.png'.format(path_out, data_whithout_dot[i]), dpi=500)
        plt.close()

        name = "SVM_model_{}".format(data_whithout_dot[i])
        # Save model
        joblib.dump(model, "{}/{}/{}.joblib".format(path_out, name_training, name))  # guarda el modelo en models y en memoria local
        models.append(model)

        # Performance
        X_test = X_test.fillna(0)
        predictions = model.predict(X_test)
        percentage = model.score(X_test, y_test)
        res = confusion_matrix(y_test, predictions)

        # Save
        predictions_arr.append(predictions)
        percentage_arr.append(percentage * 100)
        res_arr.append(res)
        len_test.append(len(X_test))

        tiempo_final = time()
        tiempo_ejecucion = tiempo_final - tiempo_inicial
        tiempo = format_time(tiempo_ejecucion)
        tiempo_train.append(tiempo)
    name = 'README.txt'
    README = '{}/{}/{}'.format(path_out, name_training, name)
    # almacena los datos del modelo en memoria
    with open(README, 'w') as f:
        iden = 0
        for name_model in data_whithout_dot:
            f.write('Training_model: {}\n'.format(name_model))
            f.write("Confusion Matrix: \n{}\n".format(res_arr[iden]))
            f.write('Test Set: {} \n'.format(len_test[iden]))
            f.write('Accuracy = {:.2f} %'.format(percentage_arr[iden]))
            f.write('Tiempo de entreanmiento = {} %\n\n\n'.format(tiempo_train[iden]))

            iden = iden + 1
        f.close()

    # devuelve la tabla de datos
    return models, obs


def format_time(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60

    if hours == 0:
        if minutes == 0:
            return "{:.2f} s".format(seconds)
        else:
            return "{} m : {:.2f} s".format(minutes, seconds)
    else:
        return "{} h : {} m : {:.2f} s".format(hours, minutes, seconds)


'''
    #Performance
    predictions = model.predict(X_test)
    print(predictions)
    percentage = model.score(X_test, y_test)
    res = confusion_matrix(y_test, predictions)
    print("Confusion Matrix")
    print(res)
    print(f"Test Set: {len(X_test)}")
    print(f"Accuracy = {percentage*100} %")
    
    #almacenando y cargando el modelo
    
    
    #Load model
    #model = joblib.load("{}/Muestra_2.joblib".format(PATH)) #carga el modelo
'''


def prediction_SVM(path_out_origin, path_out, path_img, path_mask, name_prediction, file_models, n_segment, compactness, sigma, threshold, layer, name_image, predictor):
    file_visual_prediction = 'Vista'
    file_visual_grabcut = 'Vista_grabcut'
    file_visual_sam = 'Vista_sam'
    file_images_centroide = 'Images_centroides'
    file_images_centroide_sample = 'Images_centroides_sample'
    masks_predict_matrix = 'masks_predict_matrix'
    PATH = '{}/{}'.format(path_out_origin, file_models)
    data_models = encoder_files(PATH)
    data_models.remove('Data_train')
    data_models.remove('README.txt')
    name_models = split_dot(data_models)
    img = encoder(path_img)
    mask = encoder(path_mask)
    # new_file(path_out, name_prediction, 0)

    file_per_image = True
    file_iou_image = True

    # PATH = '{}/{}'.format(path_out,name_prediction)
    PATH = path_out

    iou_n = []
    iou_g = []
    iou_s = []
    list_sensitivity = []
    list_specificity = []
    list_precision = []
    list_f1_score = []
    list_accuracy = []
    list_sensitivity_g = []
    list_specificity_g = []
    list_precision_g = []
    list_f1_score_g = []
    list_accuracy_g = []
    list_sensitivity_s = []
    list_specificity_s = []
    list_precision_s = []
    list_f1_score_s = []
    list_accuracy_s = []

    Lista_n = []
    Lista_g = []
    Lista_s = []
    y_val, y_val_mask = load_data(path_img, path_mask)

    for i in range(len(name_models)):
        arr_id = name_models[i].split('_')
        idx_end = arr_id[-1]
        if (idx_end == 'RGB'):
            name_experiment = "Results_RGB"
            df = vector_slic_rgb.feature_slic(name_experiment, y_val, y_val_mask, n_segment, compactness, sigma, threshold, path_img, PATH, file_per_image, file_iou_image)
        if (idx_end == 'VGG'):
            name_experiment = "Results_VGG16"
            df = vector_slic_vgg16.feature_slic(name_experiment, y_val, y_val_mask, n_segment, compactness, sigma, threshold, layer, path_img, PATH, file_per_image, file_iou_image)
        if (idx_end == 'RGB-VGG'):
            name_experiment = "Results_RGB_VGG16"
            df = vector_slic_rgb_vgg16.feature_slic(name_experiment, y_val, y_val_mask, n_segment, compactness, sigma, threshold, layer, path_img, PATH, file_per_image, file_iou_image)

        X, y_test = separate_data_train(df)

        # Load model
        model = joblib.load("{}/{}/{}".format(path_out_origin, file_models, data_models[i]))

        X = X.fillna(0)
        predictions = model.predict(X)

        # Metricas de desempeno
        sensitivity, specificity, precision, f1_score, accuracy = calculate_metrics(y_test, predictions)
        list_sensitivity.append(sensitivity)
        list_specificity.append(specificity)
        list_precision.append(precision)
        list_f1_score.append(f1_score)
        list_accuracy.append(accuracy)

        # generate_segments_slic
        segments_slic = slic(img, n_segment, compactness=compactness, sigma=sigma)
        if segments_slic[0][0] != 0:
            segments_slic = segments_slic - 1  # segments_slic has to start at 0

        # cont_pxl = contador_pxls(segments_slic)
        masks, positions_pxl = create_masks(img, segments_slic)

        array_centro = []
        x, y = centroide(masks, segments_slic)
        for i_c in range(len(x)):
            input_point = [x[i_c], y[i_c]]
            array_centro.append(input_point)
        array_centro = np.array(array_centro)
        # la imagen debe estar en formato RGB
        clas = detector_predict(predictions, segments_slic)

        # Obtenemos las posiciones donde el arreglo de etiquetas tiene el valor 1
        np_predictions = np.array(predictions)
        indices = np_predictions == 1
        indices_n = np_predictions == 0
        # Obtenemos las filas del arreglo original en esas posiciones

        # coordenadas de centroides positivos
        input_point = array_centro[indices]
        # coordenadas de centroides negativos
        input_point_n = array_centro[indices_n]

        input_label = np.ones(len(input_point))
        input_label_n = np.zeros(len(input_point_n))
        # concatenando
        input_point_all = np.concatenate((input_point, input_point_n))
        input_label_all = np.concatenate((input_label, input_label_n))

        Path_model = '{}/{}/{}'.format(path_out_origin, 'Predictions', name_prediction)

        new_file(Path_model, file_images_centroide, i)
        edge_manual = mark_boundaries(img, segments_slic)[:, :, (2, 1, 0)]
        ruta_save = '{}/{}/{}.jpg'.format(Path_model, file_images_centroide, name_models[i])
        mark_centroide(ruta_save, edge_manual, input_point_all, input_label_all)

        # sampleo de 10 puntos aleatorios
        # selected_points = input_point
        '''
        selected_points = delete_external_points(input_point)
        selected_points_n = delete_external_outpoints(input_point_n, img)
        '''
        # prueba de sample jordi
        selected_points, selected_points_n = sample_points(input_point, input_point_n, img)

        input_label = np.ones(len(selected_points))
        input_label_n = np.zeros(len(selected_points_n))
        # concatenando
        input_point_all = np.concatenate((selected_points, selected_points_n))
        input_label_all = np.concatenate((input_label, input_label_n))

        new_file(Path_model, file_images_centroide_sample, i)
        edge_manual = mark_boundaries(img, segments_slic)[:, :, (2, 1, 0)]
        ruta_save = '{}/{}/{}.jpg'.format(Path_model, file_images_centroide_sample, name_models[i])
        mark_centroide(ruta_save, edge_manual, input_point_all, input_label_all)

        # mask_sp, mask_edge_sp, mask_edge_manual, img_comparative = labels_visual(img, mask, segments_slic, predictions)

        # Path_model = '{}/{}/{}'.format(path_out_origin,'Predictions',name_prediction)
        new_file(Path_model, file_visual_prediction, i)
        iou = create_img(img, mask, segments_slic, clas, '{}/{}'.format(Path_model, file_visual_prediction), predictions, name_models[i], predict=True)
        iou_n.append(iou)
        '''
        GrabCut
        '''
        # Path_model = '{}/{}/{}'.format(path_out_origin,'Predictions',name_prediction)
        new_file(Path_model, file_visual_grabcut, i)
        iou, mask_sp = create_img(img, mask, segments_slic, clas, '{}/{}'.format(Path_model, file_visual_grabcut), predictions, name_models[i], predict=True, gc=True)
        iou_g.append(iou)

        '''
        Modelo SAM by META
        '''
        mask_sam_p = predictor_SAM(img, predictor, input_point_all, input_label_all)
        new_file(Path_model, file_visual_sam, i)
        iou = create_img(img, mask, segments_slic, clas, '{}/{}'.format(Path_model, file_visual_sam), predictions, name_models[i], predict=True, sam=True, mask_sam=mask_sam_p)
        iou_s.append(iou)

        segments_slic = slic(img, n_segment, compactness=compactness, sigma=sigma)
        if segments_slic[0][0] != 0:
            segments_slic = segments_slic - 1  # segments_slic has to start at 0
        masks, positions_pxl = create_masks(img, segments_slic)
        # la imagen debe estar en formato RGB
        clas, wound, ctd_s_pxl = detector(mask_sp, positions_pxl, segments_slic, threshold)

        df_wound = pd.DataFrame({'Wound': wound})
        y_test_g = df_wound.Wound

        clas, wound, ctd_s_pxl = detector(mask_sam_p, positions_pxl, segments_slic, threshold)
        df_wound = pd.DataFrame({'Wound': wound})
        y_test_s = df_wound.Wound

        # metric by grabcut
        sensitivity, specificity, precision, f1_score, accuracy = calculate_metrics(y_test, y_test_g)
        list_sensitivity_g.append(sensitivity)
        list_specificity_g.append(specificity)
        list_precision_g.append(precision)
        list_f1_score_g.append(f1_score)
        list_accuracy_g.append(accuracy)

        # metric by sam model
        new_file(Path_model, masks_predict_matrix, i)
        img_rgb_mask_matrix = generator_masks_matrix(y_test_s, y_test, segments_slic)
        ruta_save = '{}/{}/{}.jpg'.format(Path_model, masks_predict_matrix, name_models[i])
        cv2.imwrite(ruta_save, img_rgb_mask_matrix)

        sensitivity, specificity, precision, f1_score, accuracy = calculate_metrics(y_test, y_test_s)
        list_sensitivity_s.append(sensitivity)
        list_specificity_s.append(specificity)
        list_precision_s.append(precision)
        list_f1_score_s.append(f1_score)
        list_accuracy_s.append(accuracy)

    Lista_n.append(iou_n)
    Lista_n.append(list_sensitivity)
    Lista_n.append(list_specificity)
    Lista_n.append(list_precision)
    Lista_n.append(list_f1_score)
    Lista_n.append(list_accuracy)
    Lista_g.append(iou_g)
    Lista_g.append(list_sensitivity_g)
    Lista_g.append(list_specificity_g)
    Lista_g.append(list_precision_g)
    Lista_g.append(list_f1_score_g)
    Lista_g.append(list_accuracy_g)
    Lista_s.append(iou_s)
    Lista_s.append(list_sensitivity_s)
    Lista_s.append(list_specificity_s)
    Lista_s.append(list_precision_s)
    Lista_s.append(list_f1_score_s)
    Lista_s.append(list_accuracy_s)
    return Lista_n, Lista_g, Lista_s


def generator_masks_matrix(y_test_s, predictions, segments_slic):
    confusion_labels = assign_confusion_labels(y_test_s, predictions)
    mask_matrix = asignar_valores(segments_slic, confusion_labels)
    img_rgb_mask_matrix = crear_imagen_rgb(mask_matrix)
    img_rgb_mask_matrix = mark_boundaries(img_rgb_mask_matrix, segments_slic)
    img_rgb_mask_matrix = img_rgb_mask_matrix * 255
    return img_rgb_mask_matrix


def crear_imagen_rgb(imagen_asignada):
    # Crear una matriz vacía con las dimensiones de la imagen asignada
    imagen_rgb = np.zeros((imagen_asignada.shape[0], imagen_asignada.shape[1], 3), dtype=np.uint8)

    # Definir los colores para cada valor asignado en la imagen
    colores = {
        0: [0, 0, 255],     # Azul
        5: [255, 0, 0],     # Rojo
        15: [0, 255, 0],    # Verde
        20: [255, 255, 0]   # Amarillo
    }

    # Asignar los colores correspondientes a cada pixel de la imagen
    for i in range(imagen_asignada.shape[0]):
        for j in range(imagen_asignada.shape[1]):
            valor = imagen_asignada[i, j]
            color = colores[valor]
            imagen_rgb[i, j] = color

    return imagen_rgb


def asignar_valores(imagen, etiquetas):
    # Crear una copia de la imagen para preservar el original
    imagen_asignada = np.copy(imagen)

    # Asignar valores según las etiquetas
    for i, etiqueta in enumerate(etiquetas):
        if etiqueta == 0:  # 'tp':
            imagen_asignada[imagen == i] = 0
        elif etiqueta == 3:  # 'tn':
            imagen_asignada[imagen == i] = 5
        elif etiqueta == 1:  # 'fp':
            imagen_asignada[imagen == i] = 15
        elif etiqueta == 2:  # 'fn':
            imagen_asignada[imagen == i] = 20

    return imagen_asignada


def assign_confusion_labels(y_true, y_pred):
    confusion_labels = np.empty_like(y_pred)
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            confusion_labels[i] = 0  # "tp"
        elif y_true[i] == 0 and y_pred[i] == 1:
            confusion_labels[i] = 1  # "fp"
        elif y_true[i] == 1 and y_pred[i] == 0:
            confusion_labels[i] = 2  # "fn"
        elif y_true[i] == 0 and y_pred[i] == 0:
            confusion_labels[i] = 3  # "tn"
    return confusion_labels


def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = 0
    specificity = 0
    precision = 0
    f1_score = 0
    accuracy = 0

    # Validamos si el divisor es cero
    if tp + fn > 0:
        sensitivity = tp / (tp + fn)
    if tn + fp > 0:
        specificity = tn / (tn + fp)
    if tp + fp > 0:
        precision = tp / (tp + fp)
    if precision + sensitivity > 0:
        f1_score = 2 * precision * sensitivity / (precision + sensitivity)
    if tp + tn + fp + fn > 0:
        accuracy = (tp + tn) / (tp + tn + fp + fn)

    return sensitivity, specificity, precision, f1_score, accuracy


####
def view_clas_wound(df_aux, clas):
    df = df_aux.copy()
    for i in range(len(df[clas])):
        if df[clas][i] == 0:
            df[clas][i] = 'no wound'
        else:
            df[clas][i] = 'wound'
    sns.set_style("whitegrid")
    sns.countplot(x=clas, data=df, hue=clas)

    return df


def predictions(path_out_origin, path_out, path_img_test, path_mask_test, file_models, n_segment, compactness, sigma, threshold, layer, predictor):
    file_visual_prediction = 'Predictions'
    new_file(path_out, file_visual_prediction, 0)
    PATH = '{}/{}'.format(path_out_origin, path_img_test)
    PATH_mask = '{}/{}'.format(path_out_origin, path_mask_test)
    data_images = encoder_files(PATH)
    name_images = split_dot(data_images)

    # img = encoder(path_img_test)
    # mask = encoder(path_mask_test)

    iou_rgb = []
    iou_vgg = []
    iou_rgb_vgg = []
    iou_rgb_grab = []
    iou_vgg_grab = []
    iou_rgb_vgg_grab = []
    iou_rgb_sam = []
    iou_vgg_sam = []
    iou_rgb_vgg_sam = []

    sensitivity_rgb = []
    specificity_rgb = []
    precision_rgb = []
    f1_score_rgb = []
    accuracy_rgb = []
    sensitivity_vgg = []
    specificity_vgg = []
    precision_vgg = []
    f1_score_vgg = []
    accuracy_vgg = []
    sensitivity_rgb_vgg = []
    specificity_rgb_vgg = []
    precision_rgb_vgg = []
    f1_score_rgb_vgg = []
    accuracy_rgb_vgg = []

    sensitivity_rgb_grab = []
    specificity_rgb_grab = []
    precision_rgb_grab = []
    f1_score_rgb_grab = []
    accuracy_rgb_grab = []
    sensitivity_vgg_grab = []
    specificity_vgg_grab = []
    precision_vgg_grab = []
    f1_score_vgg_grab = []
    accuracy_vgg_grab = []
    sensitivity_rgb_vgg_grab = []
    specificity_rgb_vgg_grab = []
    precision_rgb_vgg_grab = []
    f1_score_rgb_vgg_grab = []
    accuracy_rgb_vgg_grab = []

    sensitivity_rgb_sam = []
    specificity_rgb_sam = []
    precision_rgb_sam = []
    f1_score_rgb_sam = []
    accuracy_rgb_sam = []
    sensitivity_vgg_sam = []
    specificity_vgg_sam = []
    precision_vgg_sam = []
    f1_score_vgg_sam = []
    accuracy_vgg_sam = []
    sensitivity_rgb_vgg_sam = []
    specificity_rgb_vgg_sam = []
    precision_rgb_vgg_sam = []
    f1_score_rgb_vgg_sam = []
    accuracy_rgb_vgg_sam = []

    for i in range(len(name_images)):
        # new_file(PATH, file, 0)
        name_prediction = '{}_{}'.format("Prediction", name_images[i])
        new_path_out = '{}/{}'.format(path_out, "Predictions")
        new_file(new_path_out, name_prediction, 0)

        new_file('{}/{}'.format(new_path_out, name_prediction), 'image', 0)
        shutil.copy('{}/{}'.format(PATH, data_images[i]), '{}/{}/{}/{}'.format(new_path_out, name_prediction, 'image', data_images[i]))
        path_img = '{}/{}/{}'.format(new_path_out, name_prediction, 'image')

        new_file('{}/{}/{}'.format(path_out, "Predictions", name_prediction), 'mask', 0)
        shutil.copy('{}/{}'.format(PATH_mask, data_images[i]), '{}/{}/{}/{}'.format(new_path_out, name_prediction, 'mask', data_images[i]))
        path_mask = '{}/{}/{}'.format(new_path_out, name_prediction, 'mask')

        path_origin_aux = path_out
        path_out_2 = '{}/{}'.format(new_path_out, name_prediction)

        Lista_n, Lista_g, Lista_s = prediction_SVM(path_origin_aux, path_out_2, path_img, path_mask, name_prediction, file_models,
                                                   n_segment, compactness, sigma, threshold, layer, name_images[i], predictor)

        list_iou = Lista_n[0]
        list_sensitivity = Lista_n[1]
        list_specificity = Lista_n[2]
        list_precision = Lista_n[3]
        list_f1_score = Lista_n[4]
        list_accuracy = Lista_n[5]
        list_iou_g = Lista_g[0]
        list_sensitivity_g = Lista_g[1]
        list_specificity_g = Lista_g[2]
        list_precision_g = Lista_g[3]
        list_f1_score_g = Lista_g[4]
        list_accuracy_g = Lista_g[5]
        list_iou_s = Lista_s[0]
        list_sensitivity_s = Lista_s[1]
        list_specificity_s = Lista_s[2]
        list_precision_s = Lista_s[3]
        list_f1_score_s = Lista_s[4]
        list_accuracy_s = Lista_s[5]

        # el orden importa
        iou_rgb.append(list_iou[0])
        iou_rgb_grab.append(list_iou_g[0])
        iou_rgb_sam.append(list_iou_s[0])
        iou_rgb_vgg.append(list_iou[1])
        iou_rgb_vgg_grab.append(list_iou_g[1])
        iou_rgb_vgg_sam.append(list_iou_s[1])
        iou_vgg.append(list_iou[2])
        iou_vgg_grab.append(list_iou_g[2])
        iou_vgg_sam.append(list_iou_s[2])

        sensitivity_rgb.append(list_sensitivity[0])
        specificity_rgb.append(list_specificity[0])
        precision_rgb.append(list_precision[0])
        f1_score_rgb.append(list_f1_score[0])
        accuracy_rgb.append(list_accuracy[0])
        sensitivity_vgg.append(list_sensitivity[1])
        specificity_vgg.append(list_specificity[1])
        precision_vgg.append(list_precision[1])
        f1_score_vgg.append(list_f1_score[1])
        accuracy_vgg.append(list_accuracy[1])
        sensitivity_rgb_vgg.append(list_sensitivity[2])
        specificity_rgb_vgg.append(list_specificity[2])
        precision_rgb_vgg.append(list_precision[2])
        f1_score_rgb_vgg.append(list_f1_score[2])
        accuracy_rgb_vgg.append(list_accuracy[2])

        sensitivity_rgb_grab.append(list_sensitivity_g[0])
        specificity_rgb_grab.append(list_specificity_g[0])
        precision_rgb_grab.append(list_precision_g[0])
        f1_score_rgb_grab.append(list_f1_score_g[0])
        accuracy_rgb_grab.append(list_accuracy_g[0])
        sensitivity_vgg_grab.append(list_sensitivity_g[1])
        specificity_vgg_grab.append(list_specificity_g[1])
        precision_vgg_grab.append(list_precision_g[1])
        f1_score_vgg_grab.append(list_f1_score_g[1])
        accuracy_vgg_grab.append(list_accuracy_g[1])
        sensitivity_rgb_vgg_grab.append(list_sensitivity_g[2])
        specificity_rgb_vgg_grab.append(list_specificity_g[2])
        precision_rgb_vgg_grab.append(list_precision_g[2])
        f1_score_rgb_vgg_grab.append(list_f1_score_g[2])
        accuracy_rgb_vgg_grab.append(list_accuracy_g[2])

        sensitivity_rgb_sam.append(list_sensitivity_s[0])
        specificity_rgb_sam.append(list_specificity_s[0])
        precision_rgb_sam.append(list_precision_s[0])
        f1_score_rgb_sam.append(list_f1_score_s[0])
        accuracy_rgb_sam.append(list_accuracy_s[0])
        sensitivity_vgg_sam.append(list_sensitivity_s[1])
        specificity_vgg_sam.append(list_specificity_s[1])
        precision_vgg_sam.append(list_precision_s[1])
        f1_score_vgg_sam.append(list_f1_score_s[1])
        accuracy_vgg_sam.append(list_accuracy_s[1])
        sensitivity_rgb_vgg_sam.append(list_sensitivity_s[2])
        specificity_rgb_vgg_sam.append(list_specificity_s[2])
        precision_rgb_vgg_sam.append(list_precision_s[2])
        f1_score_rgb_vgg_sam.append(list_f1_score_s[2])
        accuracy_rgb_vgg_sam.append(list_accuracy_s[2])

    iou_prom = [np.mean(iou_rgb), np.mean(iou_rgb_grab), np.mean(iou_rgb_sam), np.mean(iou_vgg), np.mean(iou_vgg_grab),
                np.mean(iou_vgg_sam), np.mean(iou_rgb_vgg), np.mean(iou_rgb_vgg_grab), np.mean(iou_rgb_vgg_sam)]
    sensitivity_prom = [np.mean(sensitivity_rgb), np.mean(sensitivity_rgb_grab), np.mean(sensitivity_rgb_sam), np.mean(sensitivity_vgg), np.mean(sensitivity_vgg_grab),
                        np.mean(sensitivity_vgg_sam), np.mean(sensitivity_rgb_vgg), np.mean(sensitivity_rgb_vgg_grab), np.mean(sensitivity_rgb_vgg_sam)]
    specificity_prom = [np.mean(specificity_rgb), np.mean(specificity_rgb_grab), np.mean(specificity_rgb_sam), np.mean(specificity_vgg), np.mean(specificity_vgg_grab),
                        np.mean(specificity_vgg_sam), np.mean(specificity_rgb_vgg), np.mean(specificity_rgb_vgg_grab), np.mean(specificity_rgb_vgg_sam)]
    precision_prom = [np.mean(precision_rgb), np.mean(precision_rgb_grab), np.mean(precision_rgb_sam), np.mean(precision_vgg), np.mean(precision_vgg_grab),
                      np.mean(precision_vgg_sam), np.mean(precision_rgb_vgg), np.mean(precision_rgb_vgg_grab), np.mean(precision_rgb_vgg_sam)]
    f1_score_prom = [np.mean(f1_score_rgb), np.mean(f1_score_rgb_grab), np.mean(f1_score_rgb_sam), np.mean(f1_score_vgg), np.mean(f1_score_vgg_grab),
                     np.mean(f1_score_vgg_sam), np.mean(f1_score_rgb_vgg), np.mean(f1_score_rgb_vgg_grab), np.mean(f1_score_rgb_vgg_sam)]
    accuracy_prom = [np.mean(accuracy_rgb), np.mean(accuracy_rgb_grab), np.mean(accuracy_rgb_sam), np.mean(accuracy_vgg), np.mean(accuracy_vgg_grab),
                     np.mean(accuracy_vgg_sam), np.mean(accuracy_rgb_vgg), np.mean(accuracy_rgb_vgg_grab), np.mean(accuracy_rgb_vgg_sam)]

    # Crear un DataFrame de pandas a partir de los arreglos
    df_rgb = pd.DataFrame({'Name_img': name_images,
                           'iou_rgb': iou_rgb,
                           'sensitivity_rgb': sensitivity_rgb,
                           'specificity_rgb': specificity_rgb,
                           'precision_rgb': precision_rgb,
                           'f1_score_rgb': f1_score_rgb,
                           'accuracy_rgb': accuracy_rgb,
                           'iou_rgb_grab': iou_rgb_grab,
                           'sensitivity_rgb_grab': sensitivity_rgb_grab,
                           'specificity_rgb_grab': specificity_rgb_grab,
                           'precision_rgb_grab': precision_rgb_grab,
                           'f1_score_rgb_grab': f1_score_rgb_grab,
                           'accuracy_rgb_grab': accuracy_rgb_grab,
                           'iou_rgb_sam': iou_rgb_sam,
                           'sensitivity_rgb_sam': sensitivity_rgb_sam,
                           'specificity_rgb_sam': specificity_rgb_sam,
                           'precision_rgb_sam': precision_rgb_sam,
                           'f1_score_rgb_sam': f1_score_rgb_sam,
                           'accuracy_rgb_sam': accuracy_rgb_sam
                           })

    df_vgg = pd.DataFrame({'Name_img': name_images,
                           'iou_vgg': iou_vgg,
                           'sensitivity_vgg': sensitivity_vgg,
                           'specificity_vgg': specificity_vgg,
                           'precision_vgg': precision_vgg,
                           'f1_score_vgg': f1_score_vgg,
                           'accuracy_vgg': accuracy_vgg,
                           'iou_vgg_grab': iou_vgg_grab,
                           'sensitivity_vgg_grab': sensitivity_vgg_grab,
                           'specificity_vgg_grab': specificity_vgg_grab,
                           'precision_vgg_grab': precision_vgg_grab,
                           'f1_score_vgg_grab': f1_score_vgg_grab,
                           'accuracy_vgg_grab': accuracy_vgg_grab,
                           'iou_vgg_sam': iou_vgg_sam,
                           'sensitivity_vgg_sam': sensitivity_vgg_sam,
                           'specificity_vgg_sam': specificity_vgg_sam,
                           'precision_vgg_sam': precision_vgg_sam,
                           'f1_score_vgg_sam': f1_score_vgg_sam,
                           'accuracy_vgg_sam': accuracy_vgg_sam
                           })

    df_rgb_vgg = pd.DataFrame({'Name_img': name_images,
                               'iou_rgb_vgg': iou_rgb_vgg,
                               'sensitivity_rgb_vgg': sensitivity_rgb_vgg,
                               'specificity_rgb_vgg': specificity_rgb_vgg,
                               'precision_rgb_vgg': precision_rgb_vgg,
                               'f1_score_rgb_vgg': f1_score_rgb_vgg,
                               'accuracy_rgb_vgg': accuracy_rgb_vgg,
                               'iou_rgb_vgg_grab': iou_rgb_vgg_grab,
                               'sensitivity_rgb_vgg_grab': sensitivity_rgb_vgg_grab,
                               'specificity_rgb_vgg_grab': specificity_rgb_vgg_grab,
                               'precision_rgb_vgg_grab': precision_rgb_vgg_grab,
                               'f1_score_rgb_vgg_grab': f1_score_rgb_vgg_grab,
                               'accuracy_rgb_vgg_grab': accuracy_rgb_vgg_grab,
                               'iou_rgb_vgg_sam': iou_rgb_vgg_sam,
                               'sensitivity_rgb_vgg_sam': sensitivity_rgb_vgg_sam,
                               'specificity_rgb_vgg_sam': specificity_rgb_vgg_sam,
                               'precision_rgb_vgg_sam': precision_rgb_vgg_sam,
                               'f1_score_rgb_vgg_sam': f1_score_rgb_vgg_sam,
                               'accuracy_rgb_vgg_sam': accuracy_rgb_vgg_sam

                               })

    # Guardar el DataFrame como un archivo CSV en la memoria local
    direccion = '{}/{}'.format(path_out, 'Predictions')
    df_rgb.to_excel('{}/Predicciones_RGB.xlsx'.format(direccion).format(), index=False)
    df_vgg.to_excel('{}/Predicciones_VGG.xlsx'.format(direccion), index=False)
    df_rgb_vgg.to_excel('{}/Predicciones_RGB_VGG.xlsx'.format(direccion), index=False)

    name = 'README.txt'
    README = '{}/{}/{}'.format(path_out, 'Predictions', name)
    with open(README, 'w') as f:
        f.write('Predictions:\n\n')
        f.write('number of test images: {}'.format(len(name_images)))
        f.write('\nIOU_rgb: {:.2f} || IOU_rgb_grab: {:.2f} || IOU_rgb_sam: {:.2f}\n IOU_vgg: {:.2f} || IOU_vgg_grab: {:.2f} || IOU_vgg_sam: {:.2f}  \n IOU_rgb_vgg: {:.2f} || IOU_rgb_vgg_grab: {:.2f} || IOU_rgb_vgg_sam: {:.2f}\n'.format(
            iou_prom[0], iou_prom[1], iou_prom[2], iou_prom[3], iou_prom[4], iou_prom[5], iou_prom[6], iou_prom[7], iou_prom[8]))
        f.write('\nSENSITIVITY_rgb: {:.2f} || SENSITIVITY_rgb_grab: {:.2f} || SENSITIVITY_rgb_sam: {:.2f}\n SENSITIVITY_vgg: {:.2f} || SENSITIVITY_vgg_grab: {:.2f} || SENSITIVITY_vgg_sam: {:.2f}  \n SENSITIVITY_rgb_vgg: {:.2f} || SENSITIVITY_rgb_vgg_grab: {:.2f} || SENSITIVITY_rgb_vgg_sam: {:.2f} \n'.format(
            sensitivity_prom[0], sensitivity_prom[1], sensitivity_prom[2], sensitivity_prom[3], sensitivity_prom[4], sensitivity_prom[5], sensitivity_prom[6], sensitivity_prom[7], sensitivity_prom[8]))
        f.write('\nSPECIFICITY_rgb: {:.2f} || SPECIFICITY_rgb_grab: {:.2f} || SPECIFICITY_rgb_sam: {:.2f}\n SPECIFICITY_vgg: {:.2f} || SPECIFICITY_vgg_grab: {:.2f} || SPECIFICITY_vgg_sam: {:.2f}  \n SPECIFICITY_rgb_vgg: {:.2f} || SPECIFICITY_rgb_vgg_grab: {:.2f} || SPECIFICITY_rgb_vgg_sam: {:.2f}\n'.format(
            specificity_prom[0], specificity_prom[1], specificity_prom[2], specificity_prom[3], specificity_prom[4], specificity_prom[5], specificity_prom[6], specificity_prom[7], specificity_prom[8]))
        f.write('\nPRESICION_rgb: {:.2f} || PRESICION_rgb_grab: {:.2f} || PRESICION_rgb_sam: {:.2f}\n PRESICION_vgg: {:.2f} || PRESICION_vgg_grab: {:.2f}  || PRESICION_vgg_sam: {:.2f}  \n PRESICION_rgb_vgg: {:.2f} || PRESICION_rgb_vgg_grab: {:.2f} || PRESICION_rgb_vgg_sam: {:.2f} \n'.format(
            precision_prom[0], precision_prom[1], precision_prom[2], precision_prom[3], precision_prom[4], precision_prom[5], precision_prom[6], precision_prom[7], precision_prom[8]))
        f.write('\nF1_SCORE_rgb: {:.2f} || F1_SCORE_rgb_grab: {:.2f} || F1_SCORE_rgb_sam: {:.2f}\n F1_SCORE_vgg: {:.2f} || F1_SCORE_vgg_grab: {:.2f} || F1_SCORE_vgg_sam: {:.2f}  \n F1_SCORE_rgb_vgg: {:.2f} || F1_SCORE_rgb_vgg_grab: {:.2f}  || F1_SCORE_rgb_vgg_sam: {:.2f} \n'.format(
            f1_score_prom[0], f1_score_prom[1], f1_score_prom[2], f1_score_prom[3], f1_score_prom[4], f1_score_prom[5], f1_score_prom[6], f1_score_prom[7], f1_score_prom[8]))
        f.write('\nACCURACY_rgb: {:.2f} || ACCURACY_rgb_grab: {:.2f} || ACCURACY_rgb_sam: {:.2f}\n ACCURACY_vgg: {:.2f} || ACCURACY_vgg_grab: {:.2f} || ACCURACY_vgg_sam: {:.2f}  \n ACCURACY_rgb_vgg: {:.2f} || ACCURACY_rgb_vgg_grab: {:.2f} || ACCURACY_rgb_vgg_sam: {:.2f} \n'.format(
            accuracy_prom[0], accuracy_prom[1], accuracy_prom[2], accuracy_prom[3], accuracy_prom[4], accuracy_prom[5], accuracy_prom[6], accuracy_prom[7], accuracy_prom[8]))
        f.close()


def grub_cut(img, mask):
    bool_processing = True
    newmask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)  # pasar a escala de gris para que solo sea una matriz, pero no esta normalizado

    ret, orig_mask = cv2.threshold(newmask, 20, 255, cv2.THRESH_BINARY)  # normalizanos, es decir los pixeles solo tomaran 0 o 255
    orig_mask = orig_mask / 255  # los pixeles estaran entre 0 y 1

    orig_mask = np.array(orig_mask, dtype=np.uint8)

    orig_mask_new = np.zeros(img.shape[:2], np.uint8)
    # donde sea que esté marcado en blanco (primer plano seguro), cambiar mask=1
    # donde sea que esté marcado en negro (fondo seguro), cambiar mask=0
    orig_mask_new[orig_mask == 0] = 2
    orig_mask_new[orig_mask == 1] = 3

    # esto es fijo
    bgdModel = np.zeros((1, 65), dtype=np.float64)
    fgdModel = np.zeros((1, 65), dtype=np.float64)

    cv2.grabCut(img, orig_mask_new, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

    mask_grab = np.where((orig_mask_new == 2) | (orig_mask_new == 0), 0, 1).astype('uint8')

    mask_grab = detect_noise.denoise(mask_grab)

    return mask_grab

    '''
    mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask[:,:,np.newaxis]
    plt.imshow(img[:,:,(2,1,0)]),plt.colorbar(),plt.show()
    '''


def save_img_unet(images, labels, path_out, name_id, predict=False):
    # visualizando
    r, c = 2, 2
    cont = 0
    fig = plt.figure(figsize=(5 * c, 5 * r))
    for _r in range(r):
        for _c in range(c):
            plt.subplot(r, c, _r * c + _c + 1)
            img = images[cont]
            img = img[:, :, [2, 1, 0]]
            img_aux = img.astype(int)
    #  img_aux = img_aux*255
            plt.imshow(img)
            label = labels[cont]
            plt.title(label)
            plt.axis(False)
            cont += 1
    iou = calculate_iou(images[1], images[0])
    if predict:
        name_id_arr = name_id.split('_')
        title_id = name_id_arr[-1]
        fig.suptitle("{}\n\n IoU = {:.2f}".format(title_id, iou))
        plt.savefig("{}/{}.jpg".format(path_out, name_id))
        plt.close()
        return iou

    else:
        fig.suptitle("IoU = {:.2f}".format(iou))
        plt.savefig("{}/{}.jpg".format(path_out, name_id))
        plt.close()


def labels_visual_unet(img, mask, mask_sp):

    img_unet = img.copy()
    # contorno de mascara con sp
    S = cv2.Canny(mask_sp, 250, 255)
    # contorno de mascara
    Q = cv2.Canny(mask, 250, 255)

    # Graficando contorno de mascara manual
    (N, M) = img[:, :, 0].shape
    img[Q == 255] = [255, 0, 0]
    img_unet[S == 255] = [0, 0, 255]
    img_comparative = img.copy()
    img_comparative[S == 255] = [0, 0, 255]

    return mask, mask_sp, img, img_comparative


def show_points(coords, labels, ax, marker_size=50):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def mark_centroide(ruta_save, image, input_point, input_label):

    # Creamos la figura
    image = image[:, :, (2, 1, 0)]
    plt.imshow(image)
    show_points(input_point, input_label, plt.gca())
    plt.axis('off')
    # Guardamos la imagen sin los ejes y sin marcos blancos
    plt.savefig(ruta_save, dpi=300, bbox_inches='tight', transparent=True)
    # Cerramos la figura
    plt.close()

    # Evitamos que la figura se muestre en la consola antes de guardarla
    plt.show()


def create_region(points):
    """
    Given a list of 2D points, returns a Polygon object that defines the region
    defined by the extreme points.
    """
    # Convert the points to a NumPy array for easy indexing
    points = np.array(points)

    # Find the indices of the extreme points (i.e. the minimum and maximum x and y values)
    min_x_idx = np.argmin(points[:, 0])
    max_x_idx = np.argmax(points[:, 0])
    min_y_idx = np.argmin(points[:, 1])
    max_y_idx = np.argmax(points[:, 1])

    # Create an array of the extreme points in order
    extreme_points = points[[min_x_idx, max_y_idx, max_x_idx, min_y_idx, min_x_idx], :]

    # Create a Polygon object from the extreme points
    polygon = Polygon(extreme_points, closed=True)

    return polygon


'''
def delete_external_points(points, points_n):
    ix = 0
    lim = 5
    while points.shape[0]>lim:
        # Create the polygon defined by the extreme points
        polygon = create_region(points)
        # Get the points inside the polygon
        inside_points = []
        for point in points:
            if polygon.contains_point(point):
                inside_points.append(point)
        inside_points = np.array(inside_points)
        
        new_points_n = []
        for point_n in points_n:
            if not polygon.contains_point(point_n):
                new_points_n.append(point_n)
        points_n = np.array(new_points_n)
        points_n = new_points_n

            
        if inside_points.shape[0]>lim:
            inside_points = eliminar_elementos_comunes(inside_points, polygon.xy)
        points = inside_points
        ix += 1
        
    if points.shape[0]>lim:
        points = seleccionar_aleatorio(points)
    return points, points_n
'''


def seleccionar_aleatorio(arreglo):
    # Obtener la longitud del arreglo original
    n = arreglo.shape[0]

    # Usar la función numpy.random.choice() para seleccionar 10 índices aleatorios
    indices_aleatorios = np.random.choice(n, size=5, replace=False)

    # Usar los índices aleatorios para seleccionar las 10 coordenadas del arreglo original
    coordenadas_seleccionadas = arreglo[indices_aleatorios]

    return coordenadas_seleccionadas


def eliminar_elementos_comunes(arr1, arr2):
    # Convertir arreglos a conjuntos
    set1 = set(map(tuple, arr1))
    set2 = set(map(tuple, arr2))

    # Realizar la operación de resta entre conjuntos
    diferencia = set1 - set2

    # Convertir conjunto a arreglo de NumPy
    resultado = np.array(list(diferencia))

    return resultado


def circulo_en_centro(img):
    # Obtenemos el centro de la imagen
    alto, ancho, _ = img.shape
    centro_x, centro_y = ancho // 2, alto // 2

    # Definimos el radio del círculo como la mitad del tamaño de la imagen
    radio = min(alto, ancho) // 2

    # Obtenemos las coordenadas de la circunferencia
    angulos = np.linspace(0, 2 * np.pi, 100)
    x = centro_x + radio * np.cos(angulos)
    y = centro_y + radio * np.sin(angulos)
    extreme_points = np.array(list(zip(x, y)), dtype=np.int32)

    polygon = Polygon(extreme_points, closed=True)

    # Devolvemos las coordenadas de la circunferencia
    return polygon


def delete_external_outpoints(points, img):
    polygon = circulo_en_centro(img)
    # Get the points inside the polygon
    out_points = []
    for point in points:
        if not polygon.contains_point(point):
            out_points.append(point)
    out_points = np.array(out_points)
    return out_points


def delete_external_points(points):
    ix = 0
    lim = 15
    while points.shape[0] > lim:
        # Create the polygon defined by the extreme points
        polygon = create_region(points)
        # Get the points inside the polygon
        inside_points = []
        for point in points:
            if polygon.contains_point(point):
                inside_points.append(point)
        inside_points = np.array(inside_points)
        if inside_points.shape[0] > lim:
            inside_points = eliminar_elementos_comunes(inside_points, polygon.xy)
        points = inside_points
        ix += 1

    if points.shape[0] > lim:
        points = seleccionar_aleatorio(points)
    return points


# Algortimo de Sampleo de Jordi


def sample_points(X_p, X_n, img, k=5):
    segments_slic = slic(img, n_segments=100, compactness=10, sigma=1)

    numb_of_positive = [segments_slic[y, x] for x, y in X_p]
    numb_of_negative = [segments_slic[y, x] for x, y in X_n]

    mean_of_positive = [np.average(img[segments_slic == i], axis=0) for i in numb_of_positive]
    mean_of_negative = [np.average(img[segments_slic == i], axis=0) for i in numb_of_negative]

    data = np.r_[np.c_[mean_of_positive, X_p, np.ones(len(mean_of_positive))],
                 np.c_[mean_of_negative, X_n, np.zeros(len(mean_of_negative))]]

    np.random.shuffle(data)

    rgb_values = data[:, [0, 1, 2]]

    # Y tienes una lista de clases con la forma (num_pixeles,)
    classes = data[:, -1]

    # Entrenas un modelo de Random Forest con tus datos
    model = RandomForestClassifier(n_estimators=100, max_depth=5)
    model.fit(rgb_values, classes)

    # Usas el método `predict_proba` para obtener las probabilidades para cada clase
    probabilities = model.predict_proba(rgb_values)

    # Obtenemos los índices que ordenarían la segunda columna de 'B' de mayor a menor
    indices_0 = np.argsort(probabilities[:, 0])[::-1]
    indices_1 = np.argsort(probabilities[:, 1])[::-1]

    # Usamos estos índices para reordenar 'A'
    data_0 = data[indices_0]
    data_1 = data[indices_1]

    return data_1[:k, [3, 4]], data_0[:k, [3, 4]]


'''
def sample_points(X_p, X_n, k=5):
    # Convertir a arreglo numpy
    X_p = np.array(X_p)
    X_n = np.array(X_n)

    # Crear la distribución normal estandar centrada en la media de todos los puntos X_p
    xp_mean_point = X_p.mean(axis=0)
    sigma = np.array([[3000, 0],
                      [0, 5000]])
    rv = multivariate_normal(xp_mean_point, sigma)

    # Calcular las probabilidades de cada punto
    pp = rv.pdf(X_p[:, :2])
    pp /= pp.sum()

    pn = 1 / rv.pdf(X_n[:, :2])
    pn /= pn.sum()

    selected_points_p = select_k_points(X_p, k, pp)
    selected_points_n = select_k_points(X_n, k, pn)
    return selected_points_p, selected_points_n

'''

# Número de puntos a seleccionar


def select_k_points(data, k, p):
    # Obtener una muestra sin remplazo los índices basados en las probabilidades calculadas previamente
    indices = np.arange(len(data))
    chosen_indices = np.random.choice(indices, size=k, replace=False, p=p)
    return data[chosen_indices]


def predictor_SAM(image, predictor, input_point, input_label):
    predictor.set_image(image)

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    # seleccionando a la mejor prediccion
    mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best

    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        mask_input=mask_input[None, :, :],
        multimask_output=False,
    )

    mask_pre = masks[0, :, :]
    # Convertir la máscara booleana a una imagen en escala de grises
    gray_mask = np.uint8(mask_pre) * 255

    # probando
    gray_mask = detect_noise.denoise(gray_mask)

    # gray_mask = cv2.cvtColor(gray_mask, cv2.COLOR_GRAY2BGR)

    # Multiplicar elemento a elemento la imagen original y la máscara
    # result = cv2.multiply(image, gray_mask)

    # Convertir la imagen resultante a escala de grises
    # result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    return gray_mask


def encontrar_elementos_diferentes(arreglo1, arreglo2):
    conjunto1 = set(arreglo1)
    conjunto2 = set(arreglo2)

    elementos_diferentes = list(conjunto1 ^ conjunto2)

    return elementos_diferentes
