"""
En este módulo se implementan las funciones auxiliares a las APIs utilizadas
en este proyecto.

API Utils
---------
"""

import pickle
import requests
import lxml.html
import json
from configparser import ConfigParser



def google_url(title):
    """
    Esta función implementa un crawler de imágenes, el cual busca el titular
    de una noticia en Google Imágenes y recoge el URL de la primera imágen
    encontrada.

    Args:
        title (:obj: `str`): Título /titular de la noticia.
    Returns:
        str: URL de la imágen.
    Note:
        Puede fallar si hacemos demasiadas peticiones desde la misma IP
        debido a un soft ban de Google.
    """
    query = "+".join(title.split())
    
    res = requests.get(f"https://www.google.com/search?q={query}&source=lnms&tbm=isch")

    tree = lxml.html.fromstring(res.text)

    # Buscamos la primera foto de la búsqueda (esa lista )
    for i in [3, 2, 0, 1, 4]:
        urls = tree.xpath(f'/html/body/div[{i}]/table/tr[1]/td[1]/div/div/div/div/table/tr[1]/td/a/div/img/@src')

        if len(urls) > 0:
            return urls[0]

    # Si no encontramos ninguna imagen devolvemos una imagen genérica
    return "https://www.desy.de/sites2009/site_www-desy/content/e409/Webheader_Aktuelles_665x342_ger.jpg"

def read_dict(file):
    """
    Esta función lee un diccionario que se encuentra serializado en un archivo
    pickle.

    Args:
        file (:obj: `str`): Nombre del fichero pickle.
    Returns:
        dict: Diccionario leído.
    """
    with open(file, 'rb') as f:
        return pickle.load(f)

def save_dict(dict, file):
    """
    Esta función serializa un diccionario en un archivo pickle.

    Args:
        dict (:obj: `str`): Diccionario a serializar.
        file (:obj: `str`): Nombre del fichero de salida.
    """
    with open(file, 'wb') as f:
        pickle.dump(dict, f)

def predict_with_cache(recommender_dict, model, user_id, date):
    """
    Esta función comprueba si ya se encuentra una predicción almacenada en
    caché, si existe la devuelve y sino ejecuta el modelo correspondiente
    para obtenerla.

    Args:
        recommender_dict (:obj: `dict`): Diccionario caché.
        model (:obj: `object`): Modelo utilizado.
        user_id (:obj: `str`): Identificador del usuario.
        date (:obj: `str`): Timestamp del momento de acceso al sistema.
    Returns:
        list: predicción.
    """
    key = user_id+date
    if key in recommender_dict:
        print("model cache")
        return json.dumps(recommender_dict[key])
    
    rec = model.predict(user_id, date).tolist()
    recommender_dict[key] = rec

    return json.dumps(rec)


def get_image_link(new_id, title, image_dict):
    """
    Esta función comprueba si ya se encuentra una imagen almacenada en
    caché, si existe la devuelve y sino ejecuta el crawler para obtenerla.

    Args:
        title (:obj: `str`): Titular de la noticia.
        new_id (:obj: `str`): Identificador de la noticia.
        image_dict (:obj: `dict`): Diccionario caché.
    Returns:
        str: URL de la imágen asociada a la noticia.
    See Also:
        google_url()
    """
    if new_id in image_dict:
        return image_dict[new_id]
    print("downloading...", new_id)
    url = google_url(title)
    image_dict[new_id] = url
    return url

def get_behaviors_df(dataset, train=False):
    """
    Esta función devuelve un dataframe de los comportamientos correspondientes
    a la partición solicitada.

    Args:
        dataset (:obj: `DataManager`): Dataset.
        train (:obj: `bool`, optional):
            Señaliza si queremos que la muestra utilizada sea de la partición
            de train o de la partición de test.
    Returns:
        Dataframe: Dataframe con los behaviors de la partición correspondiente.
    """
    return dataset.get_train_behaviors_df() if train else dataset.get_valid_behaviors_df()

def get_news_df(dataset, train=False):
    """
    Esta función devuelve un dataframe de las noticias correspondientes
    a la partición solicitada.

    Args:
        dataset (:obj: `DataManager`): Dataset.
        train (:obj: `bool`, optional):
            Señaliza si queremos que la muestra utilizada sea de la partición
            de train o de la partición de test.
    Returns:
        Dataframe: Dataframe con las noticias de la partición correspondiente.
    """
    return dataset.get_train_news_df() if train else dataset.get_valid_news_df()

def read_model(file_name):
    """
    Esta función lee un modelo almacenado en un fichero pickle.

    Args:
        file_name (:obj: `str`): Nombre del fichero.
    Returns:
        model: Modelo leido.
    """
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def limit_str(s, limit):
    """
    Esta función limita el tamaño de una cadena de texto, añadiendo [...] al
    final de esta en caso de ser troceada.

    Args:
        s (:obj: `str`): Cadena de texto a limitar.
        limit (:obj: `int`): Número de caracteres límite.
    Returns:
        str: Cadena de texto resultante.
    """
    if len(str(s)) < limit:
        return s

    return s[:limit]+'[...]'


