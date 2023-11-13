from pymongo import MongoClient

#crear una conexion a Mongo DB
client = MongoClient("localhost", 27017)

#crear base de datos llamada clientes_db

db = client["clientes_db"]

#crear una coleccion llamada clientes

clientes_collection = db["clientes"]

# Datos de un cliente
cliente = {
    "nombre": "Nombre del Cliente",
    "edad": 30,
    "direccion": "Direccion del Cliente",
    "telefono": "Numero de Teléfono",
    "identificacion": "Numero de Identificacion",
    "correo_electronico": "correo@cliente.com",
    "caracteristicas_faciales": [

    ]
}

# Insertar el cliente en la colección para que se cree la base de datos# este cliente se puede eliminar despues
resultado = clientes_collection.insert_one(cliente)

