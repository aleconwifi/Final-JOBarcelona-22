import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime , timedelta
import matplotlib.pyplot as plt
from PIL import Image
import streamlit.components.v1 as components
@st.cache
def load_data():
    raw = pd.read_csv('train.csv')
    return raw


data = load_data()
#Dataset_fallecidos = data[data['Atención'] == 'Fallecido']

###---------Titulos principales -------------------

image = Image.open('hack.png')
st.image(image)
st.write("""
# Proyecto Reto JOBarcelona '22
*Telefónica Team: Arnau Puche vila, Jordi Segura, Alejandro Marcano y Pol *
""")
st.sidebar.header('Navegacion')

nav_link = st.sidebar.radio(" ", ("Inicio", "Análisis Exploratorio", "Modelo Predictivo", "Conclusiones"))

if nav_link == "Inicio":
    st.write('No ha sido desapercibido que el contexto actual es muy delicado en el mundo del fútbol. La audiencia y la falta de un 2020 sin la venta de entradas ha hecho que muchos clubes tengan una postura económica muy delicada. Para ello se nos encomienda una tarea de suma responsabilidad, crear un sistema que permita asignar precios a jugadores de forma objetiva, evitando la especulación. Los datos asociados a este enunciado son estos. Pensando en un aplicativo para solucionar este problema, cada equipo tiene que proporcionar una solución innovadora dentro de su modalidad. El dataset es el siguiente')

    st.write(data) 
    st.write(''' ## Data Science
    El objetivo es poder predecir los precios de los futbolistas de una manera precisa.). Algunas Recomendaciones
    * Utilizar desde el principio Github o el que más os guste
    * Generar unas tareas básicas
    * Llevar preparado todo la idea para que el día de la final podáis programar allí y sobresalir
    * Llevar algunas dudas ready para el día de la final así posibles problemas que salgan se pueden resolver
    ''')
   

elif nav_link == "Análisis Exploratorio":
    st.header("Análisis Exploratorio de Datos")
    st.write('A continuación se muestran una serie de gráficos que despliegan información acerca de los datos:')
  
    p = open("data_profile.html")

    components.html(p.read())


    # st.write('**Estado de los infectados por sexo**')
    
    # st.write('Se puede observar que la mayor cantidad de pacientes se encuentran recuperados, lo que es una buena señal sobre la letalidad del virus, ya que la mayor cantidad de los pacientes sobrevive el virus. Se observa también, que hay ligeramente más mujeres recuperadas, pero más hombres tanto fallecidos como en hospital')

    # st.write('**Distribución Infectados**')
  
    # st.write('Luego realizamos una distribución de cantidad de infectados por sexo, se observa que la distribución en las cinco ciudades es completamente equitativa entre hombres y mujeres, aunque se sabe que por literatura los hombres son un poco mas susceptibles a ser infectados')
    
    
    # # Ciudad de ubicación de pacientes
    # st.write('**Distribución de Infectados por Ciudad**')
    
    # st.write('Se revisa la distribución de infectados por ciudad, se encuentra una distribución acorde al tamaño poblacional de las ciudades, aunque Barranquilla esta bastante cerca de Cali, teniendo la primera la mitad de población que la segunda. ')

    # # Visualización de categorias por edad
    # st.write('**Visualización de categorias por edad**')
  
    # st.write('Se revisa la cantidad de infectados por rango de edad, se observa que la mayor cantidad de infectados se encuentra en la categoria de adulto, entre los rangos de edad de 26 a 59, esto puede asociarse a que a medida que incrementa la edad las personas son mas susceptibles al virus, ademas de que esta categoria es la que posee un mayor rango respecto a las demás')

    # #Casos importados vs contraidos en territorio nacional
    # st.write('**Casos importados vs contraidos en territorio nacional**')
  
    # st.write('Por último se visualiza para los casos importados, que países son los que más casos han aportado a Colombia, se observa que España y Estados Unido son los paises que mas aportan por una gran diferencia respecto a los demás países. ')

elif nav_link == "Modelo Predictivo":
    st.write(
    """## Modelos de Predicción 
    """)


elif nav_link == "Conclusiones":
    st.write("""
    ## Conclusiones
    *   Conclusion 1
    *   Conclusion 2
    """)
