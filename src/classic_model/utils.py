"""
En este módulo se implementan las funciones auxiliares a los modelos basados
en técnicas clásicas.

Classic Utils
-------------
"""

from classic_model.classic_recommender import ClassicRecommender
from tqdm import tqdm

from joblib import Parallel, delayed

def split_clicks(string):
    """
    Esta función lee los identificadores de noticias separados por espacio de
    una cadena de texto y los devuelve en forma de lista.

    Args:
        string (:obj: `str`): Cadena con los identificadores.
    Returns:
        list: Lista con los identificadores.
    """
    return string.split()

def split_logs(string):
    """
    Esta función lee los identificadores de noticias separados por espacio de
    una cadena de texto y los devuelve en forma de lista, además separa el 1 o
    0 concatenado, que indica si la noticia fue clickada o no.

    Args:
        string (:obj: `str`): Cadena con los identificadores.
    Returns:
        tuple: (Lista con los identificadores, Lista de bool si fue click o no).
    See Also: split_clicks()
    """
    news = []
    click = []
    for i in split_clicks(string):
        tmp = i.split('-')
        news.append(tmp[0])
        click.append(int(tmp[1]))

    return news, click


def run_eval(dataset,
             vectorizer,
             metric,
             sample_size,
             seed=42,
             use_bert_tokenizer=True,
             debug=False):
    """
    Esta función ejecuta la evaluación de un modelo de la clase
    ClassicRecommender. Es decir calcula los valores que este predice para una
    serie de situaciones. De manera ITERATIVA.

    Args:
        dataset (:obj: `DataManager`): Dataset.
        vectorizer (:obj: `str`): Modelo a utilizar.
        metric (:obj: `str`): Métrica a utilizar.
        sample_size (:obj: `int`): Tamaño de la muestra a evaluar.
        seed (:obj: `int`, optional): Semilla de la muestra a evaluar.
        use_bert_tokenizer (:obj: `bool`, optional): Utilizamos BERT o no.
        debug (:obj: `bool`, optional): Se muestran lineas de progreso o no.
    Returns:
        tuple: (resultados predichos, Resultados reales).
    """
    rec = ClassicRecommender(vectorizer=vectorizer, metric=metric, use_bert_tokenizer=use_bert_tokenizer)
    results = []
    true_values = []

    rec.fit(dataset.get_train_news_df().title.values)

    valid_behaviors_df = dataset.get_valid_behaviors_df().sample(sample_size, random_state=seed)

    it = tqdm(valid_behaviors_df.iterrows(), total=valid_behaviors_df.shape[0]) if debug else valid_behaviors_df.iterrows()
    for i, user in it:
        clicked_news = split_clicks(user.click_hist)
        future_news, future_clicks = split_logs(user.imp_log)
        
        clicked_news_titles = dataset.get_valid_news_df().query('new_id in @clicked_news').title.values
        future_news_titles = dataset.get_valid_news_df().query('new_id in @future_news').title.values
        
        results.append(rec.predict(clicked_news_titles, future_news_titles))
        true_values.append(future_clicks)
    return results, true_values



def run_eval_parallel(dataset, vectorizer, metric, sample_size, seed=42, n_jobs=-1, use_bert_tokenizer=True, debug=False):
    """
    Esta función ejecuta la evaluación de un modelo de la clase
    ClassicRecommender. Es decir calcula los valores que este predice para una
    serie de situaciones. En PARALELO

    Args:
        dataset (:obj: `DataManager`): Dataset.
        vectorizer (:obj: `str`): Modelo a utilizar.
        metric (:obj: `str`): Métrica a utilizar.
        sample_size (:obj: `int`): Tamaño de la muestra a evaluar.
        seed (:obj: `int`, optional): Semilla de la muestra a evaluar.
        use_bert_tokenizer (:obj: `bool`, optional): Utilizamos BERT o no.
        debug (:obj: `bool`, optional): Se muestran lineas de progreso o no.
        n_jobs (:obj: `int`, optional): Número de procesadores entre los que
        paralizar el proceso.
    Returns:
        tuple: (resultados predichos, Resultados reales).
    """
    def helper(user, news_df):
        rec = ClassicRecommender(vectorizer=vectorizer, metric=metric, use_bert_tokenizer=use_bert_tokenizer)
        rec.fit(dataset.get_train_news_df().title.values)

        clicked_news = split_clicks(user.click_hist)
        future_news, future_clicks = split_logs(user.imp_log)
        
        clicked_news_titles = news_df.query('new_id in @clicked_news').title.values
        future_news_titles = news_df.query('new_id in @future_news').title.values
        
        return rec.predict(clicked_news_titles, future_news_titles)
    
    def true_clicks(user):
        return split_logs(user.imp_log)[1]



    valid_behaviors_df = dataset.get_valid_behaviors_df().sample(sample_size, random_states=seed)

    it = tqdm(valid_behaviors_df.iterrows(), total=valid_behaviors_df.shape[0]) if debug else dataset.get_valid_behaviors_df().iterrows()
    results = Parallel(n_jobs=n_jobs)(delayed(helper)(user, dataset.get_valid_news_df()) for _, user in it)
    
    it = tqdm(valid_behaviors_df.iterrows(), total=valid_behaviors_df.shape[0]) if debug else dataset.get_valid_behaviors_df().iterrows()
    true_values = Parallel(n_jobs=n_jobs)(delayed(true_clicks)(user) for _, user in it)
    
    return results, true_values