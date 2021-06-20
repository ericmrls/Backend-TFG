"""
En este módulo se implementa la clase ClassicRecommender, encargada de facilitar
el acceso a los algoritmos basados en técnicas clásicas.

Classic Recommender
-------------------
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances
from transformers import BertTokenizer

from classic_model import text_preprocessing


def get_vectorizer(name):
    """
    Esta función devuelve el modelo correspondiente a un nombre dado.

    Args:
        name (:obj: `str`): Nombre del modelo.
    Returns:
        model: Modelo.
    Note:
        Por el momento solo admite TF-IDF y Count (basado en frecuencias).
    """
    if name == 'tfidf':
        return TfidfVectorizer(min_df = 0)
    elif name == 'count':
        return CountVectorizer(min_df = 0)
    else:
        raise ValueError('Vectorizer sould be tfidf or count.')

class ClassicRecommender:
    """
    Clase encargada de la gestión de los modelos de recomendación basados
    en técnicas clásicas.

    Attributes:
        metric (str): métrica utilizada.
        use_bert_tokenizer (bool): uso o no del tokenizador BERT.
        algorithm (str): nombre del algoritmo a utilizar.
        vectorizer (model): modelo.
    """
    def __init__(self,
                 vectorizer='tfidf',
                 metric='euclidean',
                 use_bert_tokenizer=False):
        """
        Constructor de la clase ClassicRecommender.

        Args:
            metric (:obj: `str`): Métrica utilizada.
            use_bert_tokenizer (:obj: `bool`): Uso o no del tokenizador BERT.
            vectorizer (:obj: `str`): Nombre del algoritmo a utilizar.
        """
        self.metric = metric
        self.use_bert_tokenizer = use_bert_tokenizer
        self.algorithm = vectorizer
        self.vectorizer = get_vectorizer(vectorizer)

    def fit(self, data, clean_data=False):
        """
        Esta función realiza el entrenamiento del modelo correspondiente. Como
        podemos ver el entrenamiento consiste solamente en la construcción del
        vocabulario necesario.

        Args:
            data (:obj: `list`):
                Lista de frases (cadenas de texto separadas por espacios).
            clean_data (:obj: `bool`): Índica si se desea limpiar el texto o no.
        Note:
            La ejecución utilizando BERT impacta de manera notable en el
            rendimiento del modelo.
        """
        if clean_data:
            data = text_preprocessing.rem_stopwords_tokenize(data)
            data = text_preprocessing.lemmatize_all(data)
            data = text_preprocessing.convert_to_string(data)

        
        if self.use_bert_tokenizer:
            data = data.tolist()
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            r = self.tokenizer(data)['input_ids']
            
            vectors_tokenized = [" ".join([str(s) for s in i]) for i in r]
            data = vectors_tokenized
            
        self.vectorizer.fit_transform(data)
    

    def predict(self, document_list_A, document_list_B):
        """
        Esta función calcula una predicción en base a las similitudes de dos
        listas de documentos.

        Args:
            document_list_A (:obj: `list`): Lista de frases del histórico.
            document_list_B (:obj: `list`): Lista de frases de las candidatas.

        Returns:
            Suma de distancias de cada candidata a todas las historicas,
            para ver cual se parece más a las históricas. Cuanta mas distancia
            peor, por eso invertida.
        """

        if self.use_bert_tokenizer:
            document_list_A = document_list_A.tolist()
            document_list_B = document_list_B.tolist()

            document_list_A = [" ".join([str(s) for s in i]) for i in self.tokenizer(document_list_A)['input_ids']]
            document_list_B = [" ".join([str(s) for s in i]) for i in self.tokenizer(document_list_B)['input_ids']]


        doc_translate_A = self.vectorizer.transform(document_list_A)
        doc_translate_B = self.vectorizer.transform(document_list_B)

        

        dists = pairwise_distances(doc_translate_A,
                                   doc_translate_B,
                                   metric=self.metric)

        return 1/dists.sum(axis=0)
