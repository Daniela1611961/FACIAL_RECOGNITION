from pymongo import MongoClient

#crear una conexion a Mongo DB
client = MongoClient("localhost", 27017)

#crear base de datos llamada clientes_db

db = client["clientes_db"]

#crear una coleccion llamada clientes

clientes_collection = db["clientes"]
# Datos de un cliente
cliente = {
    "nombre": "Daniela Parra Gonzalez",
    "edad": 26,
    "direccion": "Cra 94 No 1 O 03",
    "telefono": "3115607619",
    "identificacion": "1112489490",
    "correo_electronico": "daniela.parra03@usc.edu.co",
    "caracteristicas_faciales": [
        {
            "imagen_url": "URL de la imagen",
            "fecha_captura": "2023-11-05",
            "otras_caracteristicas": "Descripción de las características faciales"
        }
    ]
}

# Insertar el cliente en la colección
resultado = clientes_collection.insert_one(cliente)