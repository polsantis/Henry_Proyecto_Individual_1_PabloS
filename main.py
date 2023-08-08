from fastapi import FastAPI, Query
import pandas as pd
import numpy as np
import pickle
from starlette.responses import RedirectResponse
from sklearn.metrics import mean_squared_error



# C:\Users\Pablo\H\Projects\PI_ML_OPS\venv/Scripts/Activate.ps1
# uvicorn main:app --reload
# run app
# uvicorn main:app --reload
# STOP:
# ctrl + c

# Configuración inicial de la API
app = FastAPI(title='PROYECTO INDIVIDUAL Nº1 - Machine Learning Operations (MLOps) - Pablo Alfonso Santisteban DataFT13',
              description='Esta API proporciona datos y predicciones de precios para de STEAM video games')



# Redireccionamiento a /docs
@app.get('/')
def redirect_to_docs():
    return RedirectResponse(url='/docs')


# Conjunto de datos para búsquedas.
df_search = pd.read_json('search_data.json')


# Realizamos ajustes en algunas columnas para su procesamiento.
df_search['release_date'] = pd.to_datetime(df_search["release_date"], errors='coerce')
df_search['metascore'] = pd.to_numeric(df_search['metascore'], errors='coerce')



# Definición de la función find_year, que será empleada en todas las demás funciones.
def find_year(anio):
    """Esta función sirve de soporte para otras funciones. Al recibir un año (int),
       devuelve un DataFrame que contiene únicamente los datos correspondientes a ese año."""
    df_anio = df_search[df_search['release_date'].dt.year == anio]
    return df_anio


# FUNCIONES:

@app.get('/genero/({year})')
def genero(year:str):
    """Al recibir un año (int) devuelve una lista con los 5 géneros más vendidos en el orden requerido."""

    try:
        anio = int(year)
    except (ValueError, KeyError, TypeError):
        return "El dato ingresado es incorrecto"

    df_genero = find_year(anio)
    df_genero = df_genero.explode("genres")
    lista_generos = df_genero['genres'].value_counts().head().index.to_list()
    return {'Año' : anio, 'Generos' : lista_generos}



@app.get('/juegos/({year})')
def juegos(year):
    """Al recibir un año (int) devuelve una lista con los juegos lanzados en el año."""

    try:
        anio = int(year)
    except (ValueError, KeyError, TypeError):
        return "El dato ingresado es incorrecto"

    df_juegos = find_year(anio)
    lista_juegos = df_juegos.title.to_list()
    
    return {'Año' : anio, 'Juegos' : lista_juegos}



@app.get('/specs/({year})')
def specs(year):
    """Al recibir un año (int) devuelve una lista con las 5 specs más repetidas en el orden correspondiente."""

    try:
        anio = int(year)
    except (ValueError, KeyError, TypeError):
        return "El dato ingresado es incorrecto"

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
        return "El dato ingresado es incorrecto"

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
        return "El dato ingresado es incorrecto"

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
        return "El dato ingresado es incorrecto"

    df_meta = find_year(anio)
    df_meta = df_meta[['title', 'metascore']].sort_values('metascore', axis=0, ascending=False).head()

    lista_name_score = []

    for i in range(df_meta.shape[0]):
        name = df_meta.iloc[i:i+1, 0:1].values[0][0]
        score = df_meta.iloc[i:i+1, 1:2].values[0][0]
        name_score = f"{name}: {score}"
        lista_name_score.append(name_score)

    return {'Año' : anio, 'Títulos' : lista_name_score}



@app.get("/predecir_precio")
async def predecir_precio(
    early_access: bool,
    year:        int =   Query(2018, description='Año'),
    genres:      list =  Query(['action','adventure'], description='Genres'),
    specs:       list =  Query(['single-player', 'multi-player'], description='Specs'),
    precio_real: float = Query(0.0, description='Precio real')):


    """Esta función acepta todas las variables necesarias como entrada 
    y devuelve la predicción del precio. Si se proporciona el precio real, 
    calculará el RMSE (Root Mean Square Error) en relación con la predicción. 
    En caso contrario, devuelve el RMSE general más óptimo obtenido por el modelo 
    mediante la técnica de Cross Validation."""
    
    # Modelo
    with open('modelo_elastic.pkl', 'rb') as modelo:
        modelo_elastic = pd.read_pickle(modelo)

    # Tabla vacía para predicción
    with open('x_prediccion.pkl', 'rb') as x_prediccion:
        x_pred = pd.read_pickle(x_prediccion)


    # Early_access True o False
    x_pred['early_access'] = early_access


    x_pred['year'] = year


    # Lista de specs, tags y genres:
    lista_features = x_pred.columns.tolist()
    lista_features = lista_features[2:]

    # Completamos con 1 las columnas correspondientes
    for genre in genres:
        if genre.lower() in lista_features:
            x_pred[genre.lower()] = 1
    for spec in specs:
        if spec.lower() in lista_features:
            x_pred[spec.lower()] = 1


    # Predicción
    prediccion = modelo_elastic.predict(x_pred)

    # Realización de este paso adicional debido a que el método 
    # mean_squared_error demanda un valor con formato de array, 
    # al margen de la prolijidad.

    if precio_real == 0:
        rmse = f"RMSE del modelo: 9.901"
        pred_str = f"Precio predicho: ${round(prediccion[0], 2)}"
        return {'prediccion': pred_str,
                'RMSE' : rmse}
    else:
        rmse = f"RMSE: {np.sqrt(mean_squared_error(pd.Series({'precio':precio_real}), prediccion))}"
        pred_str = f"Precio predicho: ${round(prediccion[0], 2)}"
        precio_real_str = f"Precio real: ${precio_real}"
        return {'prediccion': pred_str,
                'precio_real': precio_real_str,
                'RMSE' : rmse}