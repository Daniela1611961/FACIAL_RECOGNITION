# Script para capturar datos basicos de clientes y guardarlos en la Base de datos DB_Clientes previamente creada

from facenet_pytorch import MTCNN
import torch
import numpy as np
import cv2
from PIL import Image
import os
from pymongo import MongoClient

# Iniciar una conexión a MongoDB
client = MongoClient("localhost", 27017)
db = client["clientes_db"]
clientes_collection = db["clientes"]

device = 'cpu'

print('Running on device: {}'.format(device))
# crear instancia del detector de rostros MTCNN
mtcnn = MTCNN(keep_all=True, device=device)

# se crea el objeto camara que captura imagenes en tiempo real '0' es el indice de la camara
cam = cv2.VideoCapture(0)
# directorio donde se encuentran alohadas las carpetas de imagenes de los clientes
destino = 'clientes/'

# Dimensionar las imagenes
new_dim = (300, 300)
continuar = True

# preguntar al usuario por el numero de fotos
fotos_por_cliente = 0
while fotos_por_cliente <= 3:
    fotos_por_cliente = int(input("Número de fotos por cliente? Debe ser un numero mayor o igual a 3: "))

# se crea el directorio para cada cliente
while continuar:
    k = 0
    cliente = input("ID del cliente: ")
    path_cliente = destino + cliente  # se crea la ruta con el numero de ID
    if not os.path.exists(path_cliente):
        print("Creando el directorio para: '{}' ".format(path_cliente))
        os.makedirs(path_cliente)
    cliente_data = {
        "ID": cliente,
        "Nombre": input("Escriba nombre Completo: "),
        "edad": input("Edad: "),
        "direccion": input("Dirección: "),
        "telefono": input("Número de Teléfono: "),
        "correo_electronico": input("Correo Electrónico: "),
        "caracteristicas_faciales": []
    }
    while k < fotos_por_cliente:
        retval, frame = cam.read()
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        boxes, confidence = mtcnn.detect(frame_pil)

        if np.ndim(boxes) != 0:
            box = boxes[0]
            c = confidence[0]
            box = box.astype(int)
            x, y, w, h = box

            if x > 0 and y > 0 and c > 0.95:
                cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
                cv2.imshow('frame', frame)
                print(f"Probabilidad de rostro: {c}")
                f_region = frame_pil.crop(box)
                f_region = f_region.resize(new_dim)
                new_name = path_cliente + '/img_' + str(k) + '.png'
                k = k + 1
                f_region.save(new_name)

                # Agregar información de la imagen a la lista de características faciales del cliente
                cliente_data["caracteristicas_faciales"].append({
                    "vector_caracteristicas": []
                })

        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cv2.destroyAllWindows()

    # Insertar los datos del cliente en la base de datos
    clientes_collection.insert_one(cliente_data)

    cont = input("¿Desea registrar otro cliente? (S/N): ")
    if cont.upper() == 'N':
        continuar = False

print('\nDone')
