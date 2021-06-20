"""
En este módulo se implementan las funciones auxiliares para hacer uso del
dataset MIND a lo largo de nuestro proyecto. Este se lee en DataFrame de
pandas y si no se encuentra en disco se descarga directamente desde los
servidores de Azure.

Data Manager
------------
"""

import sys
import os
import pandas as pd

sys.path.append('../../recommenders/')
from reco_utils.recommender.deeprec.deeprec_utils import download_deeprec_resources
from reco_utils.recommender.newsrec.newsrec_utils import get_mind_data_set


class DataManager:
    """
    Clase utilizada para hacer uso del dataset MIND a lo largo de nuestro
    proyecto. Este se lee en DataFrame de pandas y si no se encuentra en disco
    se descarga directamente desde los servidores de Azure.
    """
    def __init__(self, dataset_path='dataset', mind_type='demo'):
        """
        Constructor de la clase MindHelperAllIterator.

        Args:
            dataset_path (:obj: `str`):
                Fichero raiz en el que se encuentra el dataset, o en el que se
                va a descargar.
            mind_type (:obj: `str`):
                Versión del dataset deseada (demo, small o large).
        """
        data_path = os.path.join(dataset_path, mind_type)

        self.train_news_file = os.path.join(data_path, 'train', r'news.tsv')
        self.train_behaviors_file = os.path.join(data_path, 'train', r'behaviors.tsv')
        self.valid_news_file = os.path.join(data_path, 'valid', r'news.tsv')
        self.valid_behaviors_file = os.path.join(data_path, 'valid', r'behaviors.tsv')
        self.wordEmb_file = os.path.join(data_path, "utils", "embedding.npy")
        self.userDict_file = os.path.join(data_path, "utils", "uid2index.pkl")
        self.wordDict_file = os.path.join(data_path, "utils", "word_dict.pkl")
        self.vertDict_file = os.path.join(data_path, "utils", "vert_dict.pkl")
        self.subvertDict_file = os.path.join(data_path, "utils", "subvert_dict.pkl")
        self.yaml_file = os.path.join(data_path, "utils", r'{}.yaml')

        self.train_news_df = None
        self.valid_news_df = None
        self.train_behaviors_df = None
        self.valid_behaviors_df = None


        mind_url, mind_train_dataset, mind_dev_dataset, mind_utils = get_mind_data_set(mind_type)

        if not os.path.exists(self.train_news_file):
            download_deeprec_resources(mind_url,
                                       os.path.join(data_path, 'train'),
                                       mind_train_dataset)
            
        if not os.path.exists(self.valid_news_file):
            download_deeprec_resources(mind_url,
                                       os.path.join(data_path, 'valid'),
                                       mind_dev_dataset)
        if not os.path.exists(self.yaml_file.format('nrms')):
            URL=r'https://recodatasets.z20.web.core.windows.net/newsrec/'
            download_deeprec_resources(URL,
                                       os.path.join(data_path, 'utils'),
                                       mind_utils)
    
    def get_train_news_df(self):
        """
        Esta función comprueba si las noticias de train se encuentran en ram
        y las devuelve, cargándolas si no estuvieran.
        
        Returns:
            Noticias de train.
        """
        if self.train_news_df is None:
            self.train_news_df = pd.read_csv(self.train_news_file, header=None, sep='\t')
            self.train_news_df.columns=["new_id",
                                        "category",
                                        "subcategory",
                                        "title",
                                        "abstract",
                                        "url",
                                        "title_entities",
                                        "abstract_entities"]
        return self.train_news_df

    def get_valid_news_df(self):
        """
        Esta función comprueba si las noticias de test se encuentran en ram
        y las devuelve, cargándolas si no estuvieran.
        
        Returns:
            Noticias de test.
        """
        if self.valid_news_df is None:
            self.valid_news_df = pd.read_csv(self.valid_news_file, header=None, sep='\t')
            self.valid_news_df.columns=["new_id",
                                        "category",
                                        "subcategory",
                                        "title",
                                        "abstract",
                                        "url",
                                        "title_entities",
                                        "abstract_entities"]
        return self.valid_news_df


    def get_train_behaviors_df(self):
        """
        Esta función comprueba si los comportamientos de train se encuentran en
        RAM y los devuelve, cargándolos si no estuvieran.
        
        Returns:
            Comportamientos de train.
        """
        if self.train_behaviors_df is None:
            self.train_behaviors_df = pd.read_csv(self.train_behaviors_file, header=None, sep='\t')
            self.train_behaviors_df = self.train_behaviors_df.drop(columns=[0])

            self.train_behaviors_df.columns = ["user_id",
                                               "timestamp",
                                               "click_hist",
                                               "imp_log"]

            self.train_behaviors_df = self.train_behaviors_df.dropna()

        return self.train_behaviors_df

    def get_valid_behaviors_df(self):
        """
        Esta función comprueba si los comportamientos de test se encuentran en
        RAM y los devuelve, cargándolos si no estuvieran.
        
        Returns:
            Comportamientos de test.
        """
        if self.valid_behaviors_df is None:
            self.valid_behaviors_df = pd.read_csv(self.valid_behaviors_file, header=None, sep='\t')
            self.valid_behaviors_df = self.valid_behaviors_df.drop(columns=[0])

            self.valid_behaviors_df.columns = ["user_id",
                                               "timestamp",
                                               "click_hist",
                                               "imp_log"]

            self.valid_behaviors_df = self.valid_behaviors_df.dropna()

        return self.valid_behaviors_df