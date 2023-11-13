import numpy as np
from sklearn.linear_model import LogisticRegression
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import cv2
from PIL import Image
from pymongo import MongoClient

# Conectar a la base de datos MongoDB
client = MongoClient("localhost", 27017)
db = client["clientes_db"]
clientes_collection = db["clientes"]

# Inicializar el detector de rostros MTCNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)

# Inicializar el clasificador (en este caso, regresión logística)
clf = LogisticRegression(solver='lbfgs', max_iter=1000)

# Inicializar el modelo InceptionResnetV1
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Preparar datos de entrenamiento
X_train = []
y_train = []

# Recuperar datos de características faciales y etiquetas desde MongoDB
for cliente_data in clientes_collection.find():
    id_cliente = cliente_data["ID"]
    caracteristicas_faciales = cliente_data["caracteristicas_faciales"]

    for caracteristicas in caracteristicas_faciales:
        vector_caracteristicas = caracteristicas["vector_caracteristicas"]
        X_train.append(vector_caracteristicas)
        y_train.append(id_cliente)

# Entrenar el clasificador
clf.fit(X_train, y_train)

#Definir funcion que toma los datos del cliente y lo formatea en cadena de texto
def format_data(client_data):
    client_id = client_data["ID"]
    name = client_data["Nombre"]
    age = client_data["edad"]
    address = client_data["direccion"]
    phone = client_data["telefono"]
    email = client_data["correo_electronico"]

    formatted_data = f"ID: {client_id}\nNombre: {name}\nEdad: {age}\nDirección: {address}\nTeléfono: {phone}\nCorreo Electrónico: {email}"
    return formatted_data

# Definir la función para extraer características
def extract_features(frame, model):
    x_aligned, prob = mtcnn(frame, return_prob=True)
    x_embed = None  # Inicializar x_embed con None
    if x_aligned is not None and len(x_aligned) > 0:
        print('Rostro(s) detectado(s) con probabilidad: {:0.8f}'.format(float(prob)))
        x_embed_list = []

        for x in x_aligned:
            x = x.unsqueeze(0).to(device)  # Agregar dimensión de lote (batch dimension)
            x_embed = model(x).detach().cpu().numpy()
            x_embed_list.append(x_embed)

        if len(x_embed_list) > 0:
            x_embed = np.vstack(x_embed_list)  # Apilar todos los vectores de características
        else:
            x_embed = None

    # Comprobar si x_embed no es None antes de llamar a .ravel()
    if x_embed is not None:
        return x_embed.ravel()
    else:
        return None

# Capturar video desde la cámara web
cap = cv2.VideoCapture(0)  # Usar la cámara web predeterminada (cambia el número si tienes varias cámaras)

# ...
# ...

# ...

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detectar rostros en el cuadro de video
    boxes, _ = mtcnn.detect(frame)

    if ((boxes is not None)and(len(boxes) >= 1)):
        for box in boxes:
            box = box.astype(int)
            x, y, w, h = box

            # Extraer características del rostro
            face_region = frame[y:h, x:w]
            x_embed_captura = extract_features(face_region, model)

            if x_embed_captura is not None:
                # Realizar la predicción del nombre del rostro
                nombre_predicho = clf.predict([x_embed_captura])

                if nombre_predicho is None:
                    nombre_predicho = "Desconocido"
                else:
                    nombre_predicho = str(nombre_predicho[0])  # Asegúrate de que sea str

                    # Obtener los datos del cliente y formatearlos
                    cliente_data = clientes_collection.find_one({"ID": nombre_predicho})
                    if cliente_data:
                        formatted_data = format_data(cliente_data)

                        # Dividir la cadena en líneas separadas
                        lines = formatted_data.split('\n')

                        # Dibujar el cuadro de la mini tabla con los datos del cliente
                        for i, line in enumerate(lines):
                            cv2.putText(frame, line, (x + 130, y + 10 + i * 20), cv2.FONT_ITALIC, 0.4, (255, 0, 255), 2)

                # Dibujar un cuadro alrededor del rostro y mostrar el nombre predicho
                cv2.rectangle(frame, (x, y), (w, h), (255, 0, 255), 2)
    # Mostrar el cuadro de video con la predicción
    cv2.imshow('Reconocimiento Facial', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# Liberar la cámara y cerrar la ventana
cap.release()
cv2.destroyAllWindows()
