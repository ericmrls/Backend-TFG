"""
En este módulo se implementan las funciones auxiliares para utilizar en nuestro
proyecto los modelos desarrollados por Microsoft en el repositorio recommenders.
En este caso se trata del modulo básico, que utiliza el iterador MINDIterator,
que solo usa el título de la noticia para predecir.

Mind Helper
-----------
"""

import pickle
import sys
sys.path.append('../recommenders/')

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR') # only show error messages


# Imports relacionados con repositorio Recommenders
from reco_utils.recommender.newsrec.models.nrms import NRMSModel
from reco_utils.recommender.newsrec.models.lstur import LSTURModel
from reco_utils.recommender.newsrec.io.mind_iterator import MINDIterator
from reco_utils.recommender.newsrec.newsrec_utils import prepare_hparams

class MindHelper:
    """
    Clase utilizada para utilizar en nuestro proyecto los modelos desarrollados
    por Microsoft en el repositorio recommenders. En este caso se trata del
    modulo básico, que utiliza el iterador MINDIterator, que solo usa el título
    de la noticia para predecir.

    Attributes:
        model (newsrec.models): Modelo a utilizar.
        news_vecs (np.array): Array con los vectores de las noticias
        codificadas.
    """
    def __init__(self,
                 userencoder_file,
                 newsencoder_file, 
                 iterator_file,
                 news_vec_file,
                 dataset,
                 model,
                 batch_size=32,
                 seed=None):
        hparams = prepare_hparams(dataset.yaml_file.format(model), 
                          wordEmb_file=dataset.wordEmb_file,
                          wordDict_file=dataset.wordDict_file, 
                          userDict_file=dataset.userDict_file,
                          batch_size=batch_size,
                          vertDict_file=dataset.vertDict_file, 
                          subvertDict_file=dataset.subvertDict_file,
                          body_size=50,
                          epochs=1, # Una sola epoca ya que vamos a usarlo preentrenado
                          show_step=10,
                          metrics=['group_auc', 'mean_mrr', 'ndcg@5;10'])
        """
        Constructor de la clase MindHelper.

        Args:
            userencoder_file (:obj: `str`):
                Nombre del fichero con los pesos correspondientes al encoder de
                usuarios.
            newsencoder_file (:obj: `str`):
                Nombre del fichero con los pesos correspondientes al encoder de
                noticias.
            iterator_file (:obj: `str`):
                Nombre del fichero con el iterador serializado en un pickle.
            news_vec_file (:obj: `str`):
                Nombre del fichero con el vector de noticias codificadas
                serializado en un pickle.
            dataset (DataManager): Dataset.
            model (:obj: `str`): Modelo a utilizar.
            batch_size (:obj: `str`): Tamaño de batch.
            seed (:obj: `str`): Semilla aleatoria a utilizar.
        """

        iterator = MINDIterator
        if model == 'nrms':
            self.model = NRMSModel(hparams, iterator, seed=seed)
        elif model == 'lstur':
            self.model = LSTURModel(hparams, iterator, seed=seed)

        with open(iterator_file, "rb") as f:
            self.model.test_iterator = pickle.load(f)
        
        # Cargamos los pesos del modelo preentrenado
        self.model.userencoder.load_weights(userencoder_file)
        self.model.newsencoder.load_weights(newsencoder_file)

        # Leemos los embedings de noticias, son fijos
        with open(news_vec_file, "rb") as f:
            self.news_vecs = pickle.load(f)
        

    def __get_index(self, user_id, timestamp):
        """
        Esta función lee el índice en el que se encuentra un usuario con en
        un timestamp concreto en el vector de usuarios del iterador.

        Args:
            user_id (:obj: `str`): Identificador del usuario
            timestamp (:obj: `str`): Timestamp del momento a predecir.
        
        Returns:
            Índice en el que se encuentra.
        """
        return np.where((np.asarray(self.model.test_iterator.uindexes) == self.model.test_iterator.uid2index[user_id]) & \
                (np.asarray(self.model.test_iterator.times) == timestamp))[0][0]

    def __get_user_data(self, index):
        """
        Esta función lee la información correspondiente al usuario que se 
        se encuentra en el índice pasado como parámetro y la codifica en forma
        de vector.

        Args:
            index (:obj: `int`): Índice del usuario.
        
        Returns:
            Vector del usuario.
        """
        click_title_index = self.model.test_iterator.news_title_index[self.model.test_iterator.histories[index]]
        user_index = self.model.test_iterator.uindexes[index]
        impr_index = self.model.test_iterator.impr_indexes[index]

        batch_user = self.model.test_iterator._convert_user_data(
                    [user_index], [impr_index], [click_title_index],
                )

        _, user_vec = self.model.user(batch_user)

        return user_vec

    def __get_news_data(self, index):
        """
        Esta función lee la información correspondiente a la noticia que se 
        se encuentra en el índice pasado como parámetro y la codifica en forma
        de vector.

        Args:
            index (:obj: `int`): Índice del comportamiento.
        
        Returns:
            tuple(Índices de las noticias, vectores de noticias, etiquetas)
        """
        news_index = np.array(self.model.test_iterator.imprs[index], dtype="int32")
        label = np.array(self.model.test_iterator.imprs[index], dtype="int32")

        

        return news_index, self.news_vecs, label

    def __get_pred(self, news_index, user_vec):
        """
        Esta función calcula la predicción haciendo el producto escalar del
        vector del usuario con los vectores de cada noticia.

        Args:
            news_index (:obj: `int`): Índice del comportamiento.
            user_vec (:obj: `int`): Vector del usuario.
        
        Returns:
            List: predicciones con un score para cada noticia.
        """
        return np.dot(
                        np.stack([self.news_vecs[i] for i in news_index], axis=0),
                        user_vec[0],
                    )

    def predict(self, user_id, timestamp):
        """
        Función principal de la clase, llama a todas las funciones auxiliares
        anteriores para finalmente devolver la predicción del modelo.

        Args:
            user_id (:obj: `str`): Identificador del usuario
            timestamp (:obj: `str`): Timestamp del momento a predecir.
        
        Returns:
            List: predicciones con un score para cada noticia.
        """
        index = self.__get_index(user_id, timestamp)
        user_vec = self.__get_user_data(index)

        news_index, _, _ = self.__get_news_data(index)

        return self.__get_pred(news_index, user_vec)