"""
En este módulo se implementan las peticiones de la API principal, tanto para
hacer predicciones como para administrar los datos del dataset.

Main API
--------
"""

import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from starlette import status
from data_manager import DataManager
from configparser import ConfigParser
import api.utils as api_utils
import os
from classic_model.utils import split_clicks, split_logs
import json
from numba import cuda 
from api.api_requests import predict_external_model


# Leemos la configuración del archivo correspondiente.
config = ConfigParser()
config.read("api/app.conf")

api_config = config["API_CONFIG"]
data_config = config["DATA_CONFIG"]
model_config = config["MODEL_CONFIG"]

# Arrancamos la API.
app = FastAPI(docs_url='/',
              title=api_config['title'],
              description=api_config['description'])

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

# Use token based authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# Ensure the request is authenticated
def auth_request(token: str = Depends(oauth2_scheme)) -> bool:
    authenticated = token == os.getenv("API_KEY", "TEST_API_KEY")
    return authenticated


# Cargamos el dataset utilizando el DataManager       
dataset = DataManager(mind_type=data_config['MIND_type'],
                      dataset_path=data_config['dataset_path'])

# Cargamos los modelos basados en algoritmos clásicos
classic_models = {
    'count': api_utils.read_model(model_config['count_model_file']),
    'tfidf': api_utils.read_model(model_config['tfidf_model_file'])
}

def predict(user_id, date, model):
    if model in classic_models.keys():
        return classic_models[model].predict(user_id, date)

    return predict_external_model(user_id, date, model)

# Leemos el diccionario con la caché de URLs de imágenes
image_dict = api_utils.read_dict(data_config['image_dict_file'])

@app.on_event("shutdown")
def shutdown_event():
    """
    Esta función se dispara al terminar la ejecución. Guarda el diccionario
    caché con las URLs de imágenes en disco.
    """
    print("Saving Images Dict... Size: ", len(image_dict.keys()))
    api_utils.save_dict(image_dict, data_config['image_dict_file'])
    pass



@app.get("/recommendation/")
async def get_recommendation(user_id:str,
                             date:str,
                             algoritmo1:str='nrms',
                             algoritmo2:str='naml',
                             train:bool=False,
                             authenticated: bool = Depends(auth_request)):
    """
    Esta función devuelve una predicción para un usuario en un momento concreto,
    utilizando los algoritmos que se pasan como parámetro.

    Args:
        user_id (:obj: `str`): Identificador del usuario.
        date (:obj: `str`): Timestamp del momento en el que accede.
        algoritmo1 (:obj: `str`, optional): Primer algoritmo a utilizar.
        algoritmo2 (:obj: `str`, optional): Primer algoritmo a utilizar.
        train (:obj: `bool`, optional):
            Señaliza si queremos que la muestra utilizada sea de la partición
            de train o de la partición de test.

    Returns:
        response: Lista con las noticias presentadas, incluyendo la imagen
        asociada y la predicción de cada uno de los algoritmos.
    """

    # Check for authentication like so
    if not authenticated:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated")
    
    user_behaviors = api_utils.get_behaviors_df(dataset, train).query('user_id == @user_id')
    date = user_behaviors.timestamp.tolist()[int(date)]
    
    ids = user_behaviors.query('timestamp == @date').imp_log.tolist()[0]

    ids, true_values = split_logs(ids)

    available_news = api_utils.get_news_df(dataset, train).query('new_id in @ids')[['new_id', 'title', 'category', 'abstract', 'url']]
    
    available_news['alg1'] = predict(user_id, date, algoritmo1)
    available_news['alg2'] = predict(user_id, date, algoritmo2)

    available_news['image_url'] = available_news.apply((lambda row: api_utils.get_image_link(row['new_id'], row['title'], image_dict)), axis=1)

    available_news['true_value'] = true_values

    return json.dumps(available_news.to_dict('records'))


@app.get("/available_news/")
async def get_available_news(user_id:str, date:str, train:bool=False):
    """
    Esta función envía las noticias mostradas (candidatas) a un usuario en
    un timestamp (date) concreto.

    Args:
        user_id (:obj: `str`): Identificador del usuario.
        date (:obj: `str`): Timestamp del momento en el que accede.
        train (:obj: `bool`, optional):
            Señaliza si queremos que la muestra sea de la partición de train o
            de la partición de test.
    Returns:
        response: Lista con las noticias.
    """
    return api_utils.get_behaviors_df(dataset, train).query('user_id == @user_id and timestamp == @date').imp_log.tolist()


@app.get("/clicked_news/")
async def get_clicked_news(user_id:str, abs_limit:int=None, train:bool=False, details:bool=False):
    """
    Esta función envía las noticias que ha clickado en el pasado un usuario.

    Args:
        user_id (:obj: `str`): Identificador del usuario.
        date (:obj: `str`): Timestamp del momento en el que accede.
        train (:obj: `bool`, optional):
            Señaliza si queremos que la muestra sea de la partición de train o
            de la partición de test.
        details (:obj: `bool`, optional):
            Indica si se desea que las noticias incluyan toda la información
            asociada o solamente los identificadores.
    Returns:
        response: Lista con las noticias.
    """

    user_behaviors = api_utils.get_behaviors_df(dataset, train).query('user_id == @user_id')

    ids = user_behaviors.click_hist.iloc[0]

    ids = split_clicks(ids)
    if not details:
        return ids

    clicked_news = api_utils.get_news_df(dataset, train).query('new_id in @ids')[['new_id', 'title', 'category', 'abstract', 'url']]

    clicked_news['image_url'] = clicked_news.apply((lambda row: api_utils.get_image_link(row['new_id'], row['title'], image_dict)), axis=1)

    if abs_limit is not None:
        clicked_news.abstract = clicked_news.abstract.apply((lambda s: api_utils.limit_str(s, abs_limit)))
    
    clicked_news.title = clicked_news.title.apply((lambda s: api_utils.limit_str(s, 110)))

    return json.dumps(clicked_news.to_dict('records'))


@app.get("/timestamp/")
async def get_timestamps(user_id:str, train:bool=False):
    """
    Esta función envía los diferentes timestamps en los que el usuario accedió
    al sistema.

    Args:
        user_id (:obj: `str`): Identificador del usuario.
        train (:obj: `bool`, optional):
            Señaliza si queremos que la muestra sea de la partición de train o
            de la partición de test.
    Returns:
        response: Lista con los timestamps.
    """

    behaviors = api_utils.get_behaviors_df(dataset, train).query('user_id == @user_id').timestamp.tolist()
    print(behaviors)
    if len(behaviors) < 1:
        raise HTTPException(status_code=404, detail="User not found")
    return behaviors


@app.get("/users/")
async def get_users(n:int=10, train:bool=False):
    """
    Esta función envía una muestra aleatoria de identificadores de usuarios
    del dataset.

    Args:
        n (:obj: `int`, optional): Tamaño de la muestra.
        train (:obj: `bool`, optional):
            Señaliza si queremos que la muestra sea de la partición de train o
            de la partición de test.
    Returns:
        response: Lista con los identificadores.
    """
    return api_utils.get_behaviors_df(dataset, train).sample(n, random_state=42).user_id.unique().tolist()

@app.get("/news_sample/")
async def news_sample(n:int=10, train:bool=False):
    """
    Esta función envía una muestra aleatoria de noticias del dataset.

    Args:
        n (:obj: `int`, optional): Tamaño de la muestra.
        train (:obj: `bool`, optional):
            Señaliza si queremos que la muestra sea de la partición de train o
            de la partición de test.
    Returns:
        response: Lista con las noticias.
    """
    news = api_utils.get_news_df(dataset, train).sample(n, random_state=41)

    news['image_url'] = news.apply((lambda row: api_utils.get_image_link(row['new_id'], row['title'], image_dict)), axis=1)

    return json.dumps(news.to_dict('records'))


if __name__ == "__main__":
    uvicorn.run(app, host=api_config['host'], port=int(api_config['port']))
