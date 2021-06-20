"""
En este módulo se implementan las peticiones necesarias para consumir de cada
una de las API auxiliares de modelos que componen este proyecto.

Deep Learning Model API
-----------------------
"""

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from data_manager import DataManager
from configparser import ConfigParser
import api.utils as api_utils
from deep_learning_model.mind_helper_all import MindHelperAllIterator
from deep_learning_model.mind_helper import MindHelper
from numba import cuda 
import argparse


# Argumentos que se pasan como parámetros al ejecutar el modelo.
parser = argparse.ArgumentParser(
    description='Deploy a Deep Learning model (NRMS, NAML or LSTUR).')
parser.add_argument("-m", "--model", required=True,
                    help="Name of the model.")
args = parser.parse_args()

# Leemos el nombre del modelo pasado como parámetro.
model_name = args.model.lower()


# Leemos la configuración del archivo correspondiente.
config = ConfigParser()
config.read("api/app.conf")

api_config = config["API_CONFIG"]
data_config = config["DATA_CONFIG"]
model_config = config["DL_MODELS_CONFIG"]

model_config = {k:v.format(model_name) for k,v in model_config.items()}

# Arrancamos la API.
app = FastAPI(docs_url='/',
              title=f"{model_name} Model",
              description=f"{model_name} Model API")
print(f"Desplegando modelo {model_name}.")

# Configuramos la API para recibir llamadas desde el navegador.
origins = [
    "http://localhost:5000",
    "http://127.0.0.1:5000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Cargamos el dataset utilizando el DataManager
dataset = DataManager(mind_type=data_config['MIND_type'],
                      dataset_path=data_config['dataset_path'])


# Cargamos el modelo en función del iterador que necesita.
if model_name != 'naml':
    model = MindHelper(model_config['userencoder'],
                       model_config['newsencoder'],
                       model_config['iterator'],
                       model_config['news_vector'],
                       dataset,
                       model=model_name)

else:
    model = MindHelperAllIterator(model_config['userencoder'],
                                  model_config['newsencoder'],
                                  model_config['iterator'],
                                  model_config['news_vector'],
                                  dataset,
                                  model=model_name)


# Leemos el diccionario con la caché de predicciones
recommender_dict = api_utils.read_dict(model_config['cache'])

@app.on_event("shutdown")
def shutdown_event():
    """
    Esta función se dispara al terminar la ejecución. Borra la RAM de la GPU
    en la que se esta ejecutando el modelo utilizando CUDA y guarda el
    diccionario caché de predicciones en disco.
    """

    device = cuda.get_current_device()
    device.reset()
    print("Saving Prediction Dict... Size: ", len(recommender_dict.keys()))
    api_utils.save_dict(recommender_dict, model_config['cache'])
    pass


@app.get("/recommendation/")
async def get_recommendation(user_id:str, date:str):
    """
    Esta función devuelve una predicción para un usuario en un momento concreto.

    Args:
        user_id (:obj: `str`): Identificador del usuario.
        date (:obj: `str`): Timestamp del momento en el que accede.
    Returns:
        response: Lista con las predicciones o None en caso de error.
    """
    return api_utils.predict_with_cache(recommender_dict, model, user_id, date)


if __name__ == "__main__":
    uvicorn.run(app, host=api_config['host'], port=int(api_config[f'{model_name}_port']))
