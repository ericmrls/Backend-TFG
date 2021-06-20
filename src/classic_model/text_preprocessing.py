"""
En este módulo algunas funciones para preprocesar texto antes de pasarselo a
cada uno de nuestros modelos.

Text Pre-processing
-------------------
"""

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer 


def rem_stopwords_tokenize(data): 
    """
    Esta función elimina stopwords de una frase concreta.

    Args:
        data (:obj: `list`): Frase, lista de palabras.
    Returns:
        list: Frase sin stopwords.
    """
    stop_words = set(stopwords.words('english')) 
    x=[]
    for i in data:
        word_tokens = word_tokenize(i) 
        x.append([w for w in word_tokens if not w in stop_words])
        
    return x


def lemmatize_all(data):
    """
    Esta función elimina el 'lema' de todas las palabras de una frase.

    Args:
        data (:obj: `list`): Frase, lista de palabras.
    Returns:
        list: Frase con palabras sin lema.
    """
    lemmatizer = WordNetLemmatizer() 
    a=[]
    for i in data:
        a.append([lemmatizer.lemmatize(j) for j in i])
    return a


def convert_to_string(data):
    """
    Esta función convierte una frase en forma de lista de palabras a una
    frase en forma de cadena de texto.

    Args:
        data (:obj: `list`): Frase, lista de palabras.
    Returns:
        str: Frase, cadena de texto
    """
    p=[]
    for i in data:
        # Importante eliminar comas 
        l = list(filter(lambda a: a != ',', i))
        listToStr = ' '.join(map(str, l))
        p.append(listToStr)
    return p
