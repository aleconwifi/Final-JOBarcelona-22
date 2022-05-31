import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime , timedelta
#import matplotlib.pyplot as plt
from PIL import Image
import streamlit.components.v1 as components
import plotly.express as px
import plotly.express as px
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report


#librerias del jupyter 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

from featuring import FeatureEngineeringDf


df =pd.read_csv('../data/train.csv')

data = FeatureEngineeringDf(df)
data.add_goals_value()
data.add_assists_value()
data.add_nations_value()
data.add_champions_score()

#Dataset_fallecidos = data[data['Atención'] == 'Fallecido']

###---------Titulos principales -------------------

image = Image.open('hack.png')
st.image(image)
st.write("""
# Proyecto Reto JOBarcelona '22
*Telefónica Team: Arnau Puche vila, Jordi Segura, Alejandro Marcano y Pol *
""")
st.sidebar.header('Navegación')

nav_link = st.sidebar.radio(" ", ("Inicio y Datasets Utilizados 🧑🏼‍💻", "Análisis Exploratorio 📊", "Modelo Predictivo 🔥", "Conclusiones ⚽️"))

if nav_link == "Inicio y Datasets Utilizados 🧑🏼‍💻":
    st.write('No ha sido desapercibido que el contexto actual es muy delicado en el mundo del fútbol. La audiencia y la falta de un 2020 sin la venta de entradas ha hecho que muchos clubes tengan una postura económica muy delicada. Para ello se nos encomienda una tarea de suma responsabilidad, crear un sistema que permita asignar precios a jugadores de forma objetiva, evitando la especulación. Los datos asociados a este enunciado son estos. Pensando en un aplicativo para solucionar este problema, cada equipo tiene que proporcionar una solución innovadora dentro de su modalidad. El dataset es el siguiente')

    st.write(''' ## Data Science
    El objetivo es poder predecir los precios de los futbolistas de una manera precisa.). Algunas Recomendaciones
    * Utilizar desde el principio Github o el que más os guste
    * Generar unas tareas básicas
    * Llevar preparado todo la idea para que el día de la final podáis programar allí y sobresalir
    * Llevar algunas dudas ready para el día de la final así posibles problemas que salgan se pueden resolver
    ''')
   
    st.header("🧑🏼‍💻 Dataset Utilizados")

    df2 =pd.read_csv('../data/Clubs-Ranking.csv')

    df3 =pd.read_csv('../data/Countries-Continents.csv')


    st.write('### Dataset de la competición')
    st.write(df)
    st.write('### Dataset de Football/Soccer Clubs Ranking')
    st.write('Esta es una lista de listas de clubes de fútbol de asociación de todo el mundo. Cada uno de los artículos enlazados desde aquí incluye listas de clubes que juegan al más alto nivel en cada país.')
    st.write("Link del dataset [link](https://www.kaggle.com/datasets/ramjasmaurya/footballsoccer-clubs-ranking)")
    st.write(df2)
    st.write('### Dataset de Países por Continentes')
    st.write('Esto lo usamos para diferenciar competiciones como la Champions Asiática de la Champions Europa, de la Copa Libertadores.')
    st.write("Link del dataset [link](https://github.com/dbouquin/IS_608/blob/master/NanosatDB_munging/Countries-Continents.csv)")
    st.write(df3) 

elif nav_link == "Análisis Exploratorio 📊":
    st.header("📊 Análisis Exploratorio de Datos")

    st.write('### Análisis General de los Datos')
    st.write(df.describe())

    st.write("""
    Podemos ver:
    *   El jugador mayor tiene 46 años, la media de edad es 26 años. 
    *   El jugador con más goles en la Selección tiene 86 goles.
    *   El precio medio es 11.6897 y el máximo es 423.0000.

    """)


    st.write('A continuación se muestran una serie de gráficos que despliegan información acerca de los datos:')
  
    st.write('### Media de las ligas con precios más altos')

    df['price'] = df['price'].astype('int')
    df_by_selections_nation =df.groupby("league", as_index=False)["price"].mean()
    df_by_selections_nation =df_by_selections_nation.sort_values(by='price', ascending=False)
    fig0 = px.bar(df_by_selections_nation.head(20), x='league', y='price')
    st.plotly_chart(fig0, use_container_width=True)


    st.write('### Media de Edad de los jugadores más costosos')
    df_by_age =df.groupby("age", as_index=False)["price"].mean()
    df_by_age =df_by_age.sort_values(by='price', ascending=False)
    fig0 = px.bar(df_by_age, x='age', y='price')
    st.plotly_chart(fig0, use_container_width=True)

    st.write('### Media de Costo de Jugadores por Posición')
    df['price'] = df['price'].astype('int32')
    df_by_position =df.groupby("position", as_index=False)["price"].mean()
    df_by_position =df_by_position.sort_values(by='price', ascending=False)
    fig00 = px.bar(df_by_position, x='position', y='price')
    st.plotly_chart(fig00, use_container_width=True)


    p = open("../data/data_profile.html")

    components.html(p.read(),  height= 1600, width=1600)

    #st_profile_report(p)
    
    #pr = df.profile_report()

    #st_profile_report(pr)

    st.write('### Correlación del dataset sin filtrar por posición')

    fig = px.imshow(data.df.corr(), text_auto='.2f', aspect="auto",  color_continuous_scale='BuPu', width=1200, height=800)
    st.plotly_chart(fig, use_container_width=True)


    st.write('### Correlación solo de Porteros')
    fig2 = px.imshow(data.filter_by_position(0).corr(), text_auto='.2f', aspect="auto",  color_continuous_scale='BuPu', width=1200, height=800)
    st.plotly_chart(fig2, use_container_width=True)

    st.write('### Correlación solo de Defensas')
    fig3 = px.imshow(data.filter_by_position(1).corr(), text_auto='.2f', aspect="auto",  color_continuous_scale='BuPu', width=1200, height=800)
    st.plotly_chart(fig3, use_container_width=True)

    st.write('### Correlación solo de Mediocampistas')
    fig4 = px.imshow(data.filter_by_position(2).corr(), text_auto='.2f', aspect="auto",  color_continuous_scale='BuPu', width=1200, height=800)
    st.plotly_chart(fig4, use_container_width=True)

    st.write('### Correlación solo de Delanteros')
    fig5 = px.imshow(data.filter_by_position(3).corr(), text_auto='.2f', aspect="auto",  color_continuous_scale='BuPu', width=1200, height=800)
    st.plotly_chart(fig5, use_container_width=True)

    st.write('### Delanteros en las ligas top')
    fig6 = px.imshow(data.filter_top_leagues().corr(), text_auto='.2f', aspect="auto",  color_continuous_scale='BuPu', width=1200, height=800)
    st.plotly_chart(fig6, use_container_width=True)

    st.write("""## Feature Engineering
                    * Según la liga del jugador:
                        - Multiplicar los goles o las asistencias de los jugadores por una score que puntua la importáncia de las ligas.

                    * Según la nacionalidad:
                        - Multiplicar el valor de veces seleccionado por una score que puntua la importáncia de las selecciones.
                        - El paso anterior con los goles también.
            """)
    
    st.write(''' 
    
        * ATTENTION
        - With that data, defenders will be valued poorly. We can't see her performance, because we've got their goals but not how many goals scored the teams against them. Same with Goalkeepers.

    * MISSING INFO
        - Goals /game --> how many minutes to score a goal?
        - Goals in top matches? Finals...
        - Experience in big matches
        - Historical injuries
        - Player mentality
        - Player position during the game
    
    * CURIOUS FACTS TO ADD INFO
        - Goal value increases if the football tendency is to score less goals in a match.
        - Global economy. If more people whats football and moves more money, players value will increase.
    
    
    ''')

elif nav_link == "Modelo Predictivo 🔥":
    st.write(
    """## Modelos de Predicción 
    """)


elif nav_link == "Conclusiones ⚽️":
    st.write("""
    ## Conclusiones
    *   Con información más precisa, el modelo será más preciso
        - Con los datos del dataset train.py queríamos añadir una nueva columna en dónde se calcularía un score para los goles de champions, es decir, és más difícil marcar en la Champions que en el torneo continental de Oceania.
        - Información extra:
            + Lesiones historicas
            + Posición del jugador durante el partido. 
            + Experiència en grandese partidos
            + Goles/minutos
        - Falta información para clasificar los porteros
        - Sería bueno ver la evolución de los goles/temporada en el mundo del futbol para ajustar el valor del gol en el momento actual.
        - Ver como evoluciona la economía en el mundo del fútbol, hace 20 años los jugadores costaban menos.


    *   Conclusiones del modelo

    """)