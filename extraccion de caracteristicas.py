import os
import torch
from facenet_pytorch import MTCNN
import cv2
from pymongo import MongoClient
import numpy as np
from facenet_pytorch import InceptionResnetV1

# Crear una instancia del detector de rostros MTCNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
base_folder = "clientes"  # Carpeta base que contiene carpetas de clientes

#Iniciar una conexion con MongoDB
client = MongoClient('localhost', 27017)
db = client["clientes_db"]
clientes_collection = db["clientes"]

# Definir funciones de carga y preprocesamiento de imágenes
def load_image(image_path):
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)#se convierte la imagen de BGR a RGB y es retornada por la funcion load_image(ruta)

def preprocess_image(image):
    return image / 255.0

# Detección de rostros aplicando MTCNN a la imagen para obtener las coordenadas de los rostros detectados

model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=40,#la imagen es de 160 px y se detectan rostros de minimo 20px
    thresholds=[0.6, 0.7, 0.7], factor=0.9, post_process=True,#atributos predeterminados
    device=device
)

# Función para extraer características de una imagen
# Función para extraer características de una imagen
def extract_features(frame):
    x_aligned, prob = mtcnn(frame, return_prob=True)
    x_embed = None  # Inicializar x_embed con None
    if x_aligned is not None:
        print('Rostro detectado con probabilidad: {:8f}'.format(prob))
        x_aligned = torch.stack([x_aligned]).to(device)
        x_embed = model(x_aligned).detach().cpu()
        x_embed = x_embed.numpy()

    # Comprobar si x_embed no es None antes de llamar a .ravel()
    if x_embed is not None:
        return x_embed.ravel()
    else:
        return None


# Extraer características de todas las imágenes en las carpetas de clientes
for client_folder in os.listdir(base_folder):#por cada cliente en la carpeta
    client_folder_path = os.path.join(base_folder, client_folder)#se define la ruta del cliente
    if os.path.isdir(client_folder_path):#si existe una carpeta en la ruta del cliente
        print(f"Procesando imágenes en la carpeta del cliente: {client_folder}")#se imprime alerta de procesando
        for filename in os.listdir(client_folder_path):#por cada imagen en la carpeta cliente
            if filename.endswith(".png"):#si la imagen esta en formato png
                image_path = os.path.join(client_folder_path, filename)#se define la ruta de la imagen
                frame = cv2.imread(image_path)#se recorta ek area de interes y se define como frame
                x_embed = extract_features(frame)#Extrae las caracteristicas del recorte
                print(f"Características extraídas de {filename}: {x_embed}")#imprime el vector de caracteristicas

                #Insertar caracteristicas en la base de datos Mongo
                if x_embed is not None:
                    x_embed_list = x_embed.tolist()
                    clientes_collection.update_one(
                        {"ID": client_folder},
                        {"$push": {"caracteristicas_faciales":{"vector_caracteristicas": x_embed_list}}}
                    )
                    print(f"caracteristicas insertadas en MongoDB para {client_folder}")


