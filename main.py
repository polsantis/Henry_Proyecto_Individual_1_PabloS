from fastapi import FastAPI
import pandas as pd
import numpy as np

# C:\Users\Pablo\H\Projects\PI_ML_OPS\venv/Scripts/Activate.ps1
# uvicorn main:app --reload
# run app
# uvicorn main:app --reload
# STOP:
# ctrl + c

# Configuración de la API
app = FastAPI(title='PROYECTO INDIVIDUAL Nº1 - Machine Learning Operations (MLOps) - Pablo Alfonso Santisteban DataFT13',
              description='Esta API proporciona datos y predicciones de precios para de STEAM video games')


# Página de inicio de la API
@app.get('/')
async def index():
    return {'¡Bienvenido a tu API de recomendación! Para comenzar, dirígete a /docs'}

# Página About de la API
@app.get('/about/')
async def about():
    return {'PROYECTO INDIVIDUAL Nº1 -Machine Learning Operations (MLOps)'}


# Upload data
df_search = pd.read_json('search_data.json')



# Convertimos la columna 'release_date' al formato de fecha y la columna 'metascore' a valores numéricos
df_search['release_date'] = pd.to_datetime(df_search["release_date"], errors='coerce')
df_search['metascore'] = pd.to_numeric(df_search['metascore'], errors='coerce')




# Definimos la función find_year, que será utilizada por otras funciones
def find_year(anio):
    """Esta función sirve de soporte para otras funciones. Al recibir un año (int),
       devuelve un DataFrame que contiene únicamente los datos correspondientes a ese año."""
    df_anio = df_search[df_search['release_date'].dt.year == anio]
    return df_anio


# FUNCIONES

@app.get('/genre/({year})')
def genre(year:str):
    """Al recibir un año (int) devuelve una lista con los 5 géneros
       más vendidos en el orden requerido."""

    try:
        anio = int(year)
    except (ValueError, KeyError, TypeError):
        return "El dato ingresado no es correcto"

    df_genre = find_year(anio)
    df_genre = df_genre.explode("genres")
    lista_genre = df_genre['genres'].value_counts().head().index.to_list()
    return {'Año' : anio, 'Generos' : lista_genre}



@app.get('/games/({year})')
def games(year):
    """Al recibir un año (int) devuelve una lista con los 5 géneros
       más vendidos en el orden requerido."""

    try:
        anio = int(year)
    except (ValueError, KeyError, TypeError):
        return "El dato ingresado no es correcto"

    df_games = find_year(anio)
    lista_games = df_games.title.to_list()
    
    return {'Año' : anio, 'Juegos' : lista_games}



@app.get('/specs/({year})')
def specs(year):
    """Al recibir un año (int) devuelve una lista con las 5 specs
       más repetidas en el orden correspondiente."""

    try:
        anio = int(year)
    except (ValueError, KeyError, TypeError):
        return "El dato ingresado no es correcto"

    df_specs = find_year(anio)
    df_specs = df_specs.explode("specs")
    lista_specs = df_specs['specs'].value_counts().head().index.to_list()
    return {'Año' : anio, 'Specs' : lista_specs}



@app.get('/earlyacces/({year})')
def earlyacces(year):
    """Al recibir un año (int) devuelve la cantidad de juegos lanzados en ese año con early access."""

    try:
        anio = int(year)
    except (ValueError, KeyError, TypeError):
        return "El dato ingresado no es correcto"

    df_early = find_year(anio)
    early = str(df_early['early_access'].sum())

    return {'Año' : anio, 'Early acces' : early}



@app.get('/sentiment/({year})')
def sentiment(year):
    """Al recibir un año (int) devuelve una lista con la cantidad de registros que
       se encuentren categorizados con un análisis de sentimiento ese año."""

    try:
        anio = int(year)
    except (ValueError, KeyError, TypeError):
        return "El dato ingresado no es correcto"

    df_sentiment = find_year(anio)
    sent_on = (df_sentiment["sentiment"] == 'Overwhelmingly Negative').sum()
    sent_vn = (df_sentiment["sentiment"] == 'Very Negative').sum()
    sent_n  = (df_sentiment["sentiment"] == 'Negative').sum()
    sent_mn = (df_sentiment["sentiment"] == 'Mostly Negative').sum()
    sent_m  = (df_sentiment["sentiment"] == 'Mixed').sum()
    sent_mp = (df_sentiment["sentiment"] == 'Mostly Positive').sum()
    sent_p  = (df_sentiment["sentiment"] == 'Positive').sum()
    sent_vp = (df_sentiment["sentiment"] == 'Very Positive').sum()
    sent_op = (df_sentiment["sentiment"] == 'Overwhelmingly Positive').sum()

    sent_on_str = f"Overwhelmingly Negative: {sent_on}"
    sent_vn_str = f"Very Negative: {sent_vn}"
    sent_n_str  = f"Negative: {sent_n}"
    sent_mn_str = f"Mostly Negative: {sent_mn}"
    sent_m_str  = f"Mixed: {sent_m}"
    sent_mp_str = f"Mostly Positive: {sent_mp}"
    sent_p_str  = f"Positive: {sent_p}"
    sent_vp_str = f"Very Positive: {sent_vp}"
    sent_op_str = f"Overwhelmingly Positive: {sent_op}"

    lista = [[sent_on, sent_on_str], [sent_vn, sent_vn_str], [sent_n, sent_n_str], [sent_mn, sent_mn_str], [sent_m, sent_m_str],
             [sent_mp, sent_mp_str], [sent_p, sent_p_str], [sent_vp, sent_vp_str], [sent_op, sent_op_str]]

    lista_final = []

    for sent in lista:
        if sent[0] > 0:
            lista_final.append(sent[1])

    return {'Año' : anio, 'Sentiments' : lista_final}



@app.get('/metascore/({year})')
def metascore(year):
    """Al recibir un año (int) devuelve  el top 5 juegos con mayor metascore."""

    try:
        anio = int(year)
    except (ValueError, KeyError, TypeError):
        return "El dato ingresado no es correcto"

    df_meta = find_year(anio)
    df_meta = df_meta[['title', 'metascore']].sort_values('metascore', axis=0, ascending=False).head()

    lista_name_score = []

    for i in range(df_meta.shape[0]):
        name = df_meta.iloc[i:i+1, 0:1].values[0][0]
        score = df_meta.iloc[i:i+1, 1:2].values[0][0]
        name_score = f"{name}: {score}"
        lista_name_score.append(name_score)

    return {'Año' : anio, 'Títulos' : lista_name_score}
