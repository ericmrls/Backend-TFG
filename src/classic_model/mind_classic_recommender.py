"""
En este módulo se implementa la clase MindClassicRecommender, encargada de
facilitar el uso de los algoritmos ClassicRecommender sobre el dataset MIND.

MIND Classic Recommender
------------------------
"""

from classic_model.classic_recommender import ClassicRecommender
from classic_model import utils

class MindClassicRecommender:
    """
    Clase encargada de facilitar el uso de los algoritmos ClassicRecommender
    sobre el dataset MIND.

    Attributes:
        dataset (DataManager): dataset MIND a utilizar.
        model (model): nombre del algoritmo a utilizar.
        field (str): campo de texto de la noticia a utilizar para comparar.
    """
    def __init__(self,
                 dataset,
                 model='count',
                 metric='manhattan',
                 field='title',
                 use_bert_tokenizer=False):
        """
        Constructor de la clase ClassicRecommender.

        Args:
            dataset (DataManager): Dataset MIND a utilizar.
            metric (:obj: `str`): Métrica utilizada.
            use_bert_tokenizer (:obj: `bool`): Uso o no del tokenizador BERT.
            model (:obj: `str`): Nombre del algoritmo a utilizar.
            field (:obj: `str`):
                Campo de texto de la noticia a utilizar para comparar noticias
    """
    
        self.model = ClassicRecommender(vectorizer=model,
                                        metric=metric,
                                        use_bert_tokenizer=use_bert_tokenizer)

        self.dataset = dataset
        self.field = field

    def fit(self):
        """
        Esta función realiza el entrenamiento del modelo correspondiente. Como
        podemos ver el entrenamiento consiste solamente en la construcción del
        vocabulario necesario.
        """
        self.model.fit(self.dataset.get_train_news_df().title.values)

    def predict(self, user_id, timestamp):
        """
        Esta función calcula una predicción de un usuario en un momento
        concreto.

        Args:
            user_id (:obj: `str`): Identificador del usuario
            timestamp (:obj: `str`): Timestamp del momento a predecir.

        Returns:
            Predicciones. Suma invertidade distancias de cada noticia candidata
            a todas las noticias historicas.
        """
        user = self.dataset.get_valid_behaviors_df().query(
            'user_id == @user_id and timestamp == @timestamp'
            ).iloc[0]

        click_hist_ids = utils.split_clicks(user.click_hist)
        imp_log_ids = utils.split_logs(user.imp_log)[0]

        click_hist = self.dataset.get_valid_news_df().query('new_id in @click_hist_ids')[self.field].values
        imp_log = self.dataset.get_valid_news_df().query('new_id in @imp_log_ids')[self.field].values

        return self.model.predict(click_hist, imp_log)