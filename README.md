# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy import stats
import scipy.stats as st
from scipy.stats import levene
import seaborn as sns


# In[2]:


df = pd.read_csv('/datasets/games.csv')
df


# # Títulos y filas nulas o duplicadas

# In[3]:


df = df.rename(columns=str.lower)
df.info()


# In[4]:


print('El número de filas duplicadas es', df.duplicated().sum())
print('El número de filas nulas es', df.isnull().all(axis=1).sum())
print('El número de filas ausentes es', df.isna().all(axis=1).sum())


# # Revisar valores usentes y duplicados en la columna name

# In[5]:


print('La cantidad de videojuegos con nombres nulos es de', df['name'].isnull().sum(), 'juegos')
print()
print('La cantidad de videojuegos con nombres ausentes es de', df['name'].isna().sum(), 'juegos' )
print()
print('La cantidad de videojuegos con nombres duplicados es de', df['name'].duplicated().sum(), 'juegos')


# In[6]:


df.loc[[659, 14244], 'name'] = 'NaN información incompleta'


# In[7]:


print(df.loc[[659, 14244]])


# Estas filas no tienen nombre del videojuego, el año de lanzamiento y la plataforma son los mismos datos, ambos no tienen valores
# en las columnas genere, critic_score, user_score y rating. La fila 14244 solo tiene ventas en Japón, la fila 659 tiene ventas en 
# NA, EU y en Otros.
# Debido a que los datos son muy pocos en ambos casos, y solo son dos, podemos prescindir de estas filas.
# 
# En el caso de los nombres duplicados, es normal puesto que el mismo juego está disponible para varias plataformas.


# # Revisar valores usentes y duplicados en la columna platform

# In[8]:


print('El número de filas con valores nulos es', df['platform'].isnull().sum())
print()
print('El número de filas con valores ausentes es', df['platform'].isna().sum())


# In[9]:


print('Los datos de la columna platform son del tipo', df['platform'].dtype)


# La columna platform no tiene valores ausentes o nulos, y los valores duplicados se deben a que hay plataformas que comparten
# el mismo juego.

# # Revisar valores usentes y duplicados en la columna year_of_release

# In[10]:


print('El número de filas con valores nulos es de', df['year_of_release'].isnull().sum())
print()
print('El número de filas con valores ausentes es de', df['year_of_release'].isna().sum())


# In[11]:


porcentaje_yor_null = (df['year_of_release'].isnull().sum() / df.shape[0] ) * 100
print('El porcentaje de datos nulos es del', porcentaje_yor_null,'%')


# In[12]:


def ext_year(nombre):
    match = re.search(r'\b\d{4}\b', nombre)
    if match:
        return match.group()
    else:
        return None

df['temp_year'] = df['name'].apply(ext_year)

df['year_of_release'].fillna(df['temp_year'], inplace=True)

df


# In[13]:


porcentaje_yor_null = (df['year_of_release'].isnull().sum() / df.shape[0] ) * 100
print('El porcentaje de datos nulos es del', porcentaje_yor_null,'%')


# In[14]:


print('El número de filas con valores nulos es de', df['year_of_release'].isnull().sum())
print('Se ha reducido', (269 - 252), 'valores nulos.')


# In[15]:


df.drop(columns=['temp_year'], inplace=True)
df['year_of_release'].fillna(0, inplace=True)
df['year_of_release'] = df['year_of_release'].astype(int)

df


# In[16]:


df[df['year_of_release'] == 0]


# En la columna year_of_release había 269 valores nulos (1.5% del total). El título del videojuego incluye, en algunas ocasiones,
# el año, y aunque no siempre es de lanzamiento, puesto que normalmente el lanzamiento es un año antes del incluído en el título,
# sí puede servir para darnos una idea del promedio de ventas anuales.
# 
# Para el resto de filas con valores nulos, he incluído un 0 para poder que todos los datos sean int y así poder hacer los 
# cálculos necesarios.


# # Revisar valores usentes y duplicados en la columna genre

# In[17]:


print('Hay ', df['genre'].isnull().sum(), 'filas con valores nulos.')
print('Hay ', df['genre'].isna().sum(), 'filas con valores ausentes.')


# In[18]:


print('Esta columna tiene datos del tipo', df['genre'].dtype)


# In[19]:


df['genre'].unique()


# La columna genre no tiene valores ausentes, y los valores repetidos se deben a que varios videojuegos tienen el mismo género.

# # Revisar valores usentes y duplicados en las columnas na_sales, eu_sales, jp_sales y other_sales

# In[20]:


print('Para la columna na_sales:')
print('Hay ', df['na_sales'].isnull().sum(), 'filas con valores nulos.')
print('Hay ', df['na_sales'].isna().sum(), 'filas con valores ausentes.')
print()
print('Para la columna eu_sales:')
print('Hay ', df['eu_sales'].isnull().sum(), 'filas con valores nulos.')
print('Hay ', df['eu_sales'].isna().sum(), 'filas con valores ausentes.')
print()
print('Para la columna jp_sales:')
print('Hay ', df['jp_sales'].isnull().sum(), 'filas con valores nulos.')
print('Hay ', df['jp_sales'].isna().sum(), 'filas con valores ausentes.')
print()
print('Para la columna other_sales:')
print('Hay ', df['other_sales'].isnull().sum(), 'filas con valores nulos.')
print('Hay ', df['other_sales'].isna().sum(), 'filas con valores ausentes.')


# In[21]:


print('Para la columna na_sales:')
print('Hay ', df['na_sales'].duplicated().sum(), 'filas duplicadas.')
print()
print('Para la columna eu_sales:')
print('Hay ', df['eu_sales'].duplicated().sum(), 'filas duplicadas.')
print()
print('Para la columna jp_sales:')
print('Hay ', df['jp_sales'].duplicated().sum(), 'filas duplicadas.')
print()
print('Para la columna other_sales:')
print('Hay ', df['other_sales'].duplicated().sum(), 'filas duplicadas.')


# Ninguna de las cuatro columnas tiene valores ausentes ni nulos. Los valores repetidos (y debido al cantidad), se debe a que
# tienen similares cantidades de ventas.

# # Revisar valores usentes y duplicados en la columna critic_score

# In[22]:


print('Hay', df['critic_score'].isnull().sum(), 'filas con valores nulos.')
print()
print('Hay', df['critic_score'].isna().sum(), 'filas con valores ausentes.')
print()
print('Hay', df['critic_score'].duplicated().sum(), 'filas con valores duplicados.')


# In[23]:


print(df['critic_score'].unique())


# In[24]:


mean_cs_name = df.groupby('name')['critic_score'].mean()
median_cs_name = df.groupby('name')['critic_score'].median()

print(mean_cs_name.sort_values(ascending=False))
print()
print(median_cs_name.sort_values(ascending=False))


# In[25]:


mean_cs_name.describe()


# In[26]:


videogames_names = mean_cs_name.shape[0]
videogames_names_cs = mean_cs_name.shape[0] - mean_cs_name.isna().sum()

print(videogames_names)
print(videogames_names_cs)


# In[27]:


porcentaje_cs_name = (videogames_names_cs / videogames_names )*100
print('El',  porcentaje_cs_name,'% de videojuegos tienen una calificación de la crítica definida.')


# In[28]:


mean_cs_name.hist()
plt.title('Histograma de la media de critic_score por videojuego')
plt.xlabel('Valor medio de critic_score')
plt.ylabel('Frecuencia')
plt.show()


# In[29]:


median_cs_name.hist()
plt.title('Histograma de la mediana de critic_score por videojuego')
plt.xlabel('Valor mediano de critic_score')
plt.ylabel('Frecuencia')
plt.show()


# Del total de videojuegos (11559) solo el 44% (5085 videojuegos) cuenta con calificación de la crítica, lo que es menos de la
# mitad de los videojuegos incluídos en el dataset.
# 
# Sobre este 44% de los videojuegos, todos están entre 16.0 (la menor) y 98.0 (la mayor) y el promedio es de 68.32. 
# La media (promedio) y la mediana son la misma, debido a que la distribucion es simétrica.

# # Revisar valores usentes y duplicados en la columna user_score

# In[30]:


print('Hay', df['user_score'].isnull().sum(), 'filas con valores nulos.')
print()
print('Hay', df['user_score'].isna().sum(), 'filas con valores ausentes.')
print()
print('Hay', df['user_score'].duplicated().sum(), 'filas con valores duplicados.')


# In[31]:


df['user_score'].unique()


# In[32]:


print((df['user_score']=='tbd').sum())
print(df['user_score'].isnull().sum())


# In[33]:


df['user_score'] = df['user_score'].replace('tbd', np.nan)
print((df['user_score']=='tbd').sum())
print(df['user_score'].isnull().sum())


# In[34]:


df['user_score'] = pd.to_numeric(df['user_score'], errors='coerce')


# In[35]:


mean_us_name = df.groupby('name')['user_score'].mean()
median_us_name = df.groupby('name')['user_score'].median()
print(mean_us_name.sort_values(ascending=False))
print()
print(median_us_name.sort_values(ascending=False))


# In[36]:


mean_us_name.describe()


# In[37]:


vg_names = mean_us_name.shape[0]
vg_names_us = mean_us_name.shape[0] - mean_us_name.isna().sum()

print(vg_names)
print(vg_names_us)


# In[38]:


porcentaje_us_name = (vg_names_us / vg_names )*100
print('El',  porcentaje_us_name,'% de videojuegos tienen una calificación de los usuarios definida.')


# In[39]:


mean_us_name.hist()
plt.title('Histograma de la media de user_score por videojuego')
plt.xlabel('Valor medio de user_score')
plt.ylabel('Frecuencia')
plt.show()


# In[40]:


median_us_name.hist()
plt.title('Histograma de la mediana de user_score por videojuego')
plt.xlabel('Valor mediano de user_score')
plt.ylabel('Frecuencia')
plt.show()


# El dataset cuenta con 11559 títulos de videojuegos, de los cuales solo el 40.6% (4694) cuentan con una calificación por parte de
# los usuarios, menos de la mitad.
# 
# Sobre esta cantidad de videojuegos, la calificación de los usuarios está entre 0.0 y 9.7, teniendo un promedio de 7.22.
# La media (promedio) y la mediana son iguales debido a la simetría de la distribución de los datos.

# # Revisar valores usentes y duplicados en la columna rating

# In[41]:


print('Hay', df['rating'].isnull().sum(), 'filas con valores nulos.')
print()
print('Hay', df['rating'].isna().sum(), 'filas con valores ausentes.')
print()
print('Hay', df['rating'].duplicated().sum(), 'filas con valores duplicados.')


# In[42]:


df['rating'].unique()


# In[43]:


mode_rating_name = df.groupby('name')['rating'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
mode_rating_name


# In[44]:


for name, mode_rating in mode_rating_name.items():
    df.loc[df['name'] == name, 'rating'] = df.loc[df['name'] == name, 'rating'].fillna(mode_rating_name)

print('Hay', df['rating'].isnull().sum(), 'filas con valores nulos.')


# In[45]:


total_rating = len(df['rating'])
total_null_rating =  df['rating'].isnull().sum()
porcentaje_rating_null = (total_null_rating / total_rating)*100
print('El',porcentaje_rating_null,'% de los valores es nulo.' )


# In[46]:


df['rating'].fillna('not rated', inplace=True)
print('Hay', df['rating'].isnull().sum(), 'filas con valores nulos.')


# En la columna rating hay valores duplicados puesto que varios videojuegos comparten la misma clasificación.
# Respecto a los valores nulos, he hecho el cambio a 'not rated', pues representan el 40.48% del total de valores de la columna.

# # Revisión de ventas

# In[47]:


df['total_sales'] = df['na_sales'] + df['eu_sales'] + df['jp_sales'] + df['other_sales']
df



# # Juegos lanzados por año

# In[48]:


df_2011 = df[df['year_of_release'] >= 2011]

games_plat_launched_per_year = df_2011.groupby(['year_of_release', 'platform'])['name'].count()

games_plat_launched_per_year.head(20)


# In[49]:


games_launched_per_year = df_2011.groupby(['year_of_release'])['name'].count()
print(games_launched_per_year)
print()
games_launched_per_year.plot(title='Cantidad de videojuegos lanzados por año', kind='bar', 
                             xlabel='Año de lanzamiento', ylabel='Cantidad de videojuegos', figsize=[10,10])
plt.show()


# A partir del año 2011 las ventas han ido a la baja, y aunque hubo una ligera alza en las ventas, la tendencia ha sido a 
# la baja.

# In[50]:


platform_sales = df_2011.groupby('platform')['total_sales'].sum().nlargest(8)
top_platforms = platform_sales.index.tolist()


# In[51]:


df_top_platforms = df_2011[df_2011['platform'].isin(top_platforms)]


# In[52]:


yearly_sales = df_top_platforms.groupby(['year_of_release', 'platform'])['total_sales'].sum().unstack()


# In[53]:


yearly_sales.plot(kind='bar', stacked=True, figsize=(15, 10))
plt.title('Distribución de ventas totales por año para las plataformas de videojuegos con mayores ventas')
plt.xlabel('Año')
plt.ylabel('Ventas totales')
plt.legend(title='Plataforma')
plt.xticks(rotation=45)
plt.show()


# In[54]:


sales_per_platform = df.groupby(['platform'])['total_sales'].sum()
sales_per_platform = sales_per_platform.sort_values(ascending=False)
print(sales_per_platform)
print()
sales_per_platform.plot(title='Total de ventas por plataforma', kind='bar', 
                             xlabel='Plataforma', ylabel='Ventas totales', figsize=[20,15])
plt.show()
print()
print('Las plataformas líderes en ventas son PS2, X360, PS3, Wii, DS y PS.')


# In[55]:


df_release = df[df['year_of_release'] >= 2000]

platforms_trendings = df_release.groupby(['year_of_release', 'platform'])['total_sales'].sum().unstack()

platforms_trendings = platforms_trendings.fillna(0)

platforms_trendings


# In[56]:


platforms_trendings.plot(kind='line', stacked=True, figsize=(50, 50))
plt.title('Tendencia de ventas por plataforma')
plt.xlabel('Año')
plt.ylabel('Ventas por año')
plt.legend(title='Plataforma')
plt.xticks(rotation=45)
plt.show()


# In[57]:


plt.figure(figsize=(10, 50))
sns.boxplot(sales_per_platform)
plt.title('Diagrama de Caja de la Ventas totales por plataforma')
plt.show()



# In[64]:


sales_per_platform = pd.DataFrame(sales_per_platform)

print("Columnas del DataFrame:", sales_per_platform.columns)


# In[66]:


sales_per_platform.reset_index()


# In[69]:


sales_per_platform_regular = sales_per_platform.drop(sales_per_platform[sales_per_platform['total_sales'] > 400].index)

sales_per_platform_regular= sales_per_platform_regular.reset_index()

sales_per_platform_regular


# In[72]:


print(sales_per_platform_regular['total_sales'].describe())
print()
print('La mediana de las ventas totales es', sales_per_platform_regular['total_sales'].median())


# In[73]:


sales_per_platform_regular['cuartil'] = pd.qcut(sales_per_platform_regular['total_sales'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
amount_per_q = sales_per_platform_regular['cuartil'].value_counts()
amount_per_q


# In[74]:


Q1 = sales_per_platform_regular['total_sales'].quantile(0.25)
Q3 = sales_per_platform_regular['total_sales'].quantile(0.75)
IQR = Q3 - Q1

upper_limit = Q3 + 1.5 * IQR

outliers_count = (sales_per_platform_regular['total_sales'] > upper_limit).sum()

print("Cantidad de valores atípicos por encima del límite superior:", outliers_count)


# In[75]:


plt.figure(figsize=(10, 6))
sns.barplot(x='platform', y='total_sales', data=sales_per_platform_regular)
plt.title('Ventas totales por plataforma')
plt.xlabel('Platform')
plt.ylabel('Total Sales')
plt.show()


# Aún cuando la distribución de valores es más o menos igual, existen muchos valores atípicos (1889) por encima del valor máximo, 
# lo que indica que hay un sesgo hacia arriba. La gráfica de caja lo muestra, todos los valores atípicos están por encima del 
# valor máximo. 

# Las plataformas PS4 y XOne tienen 4 años y, según la tendencia, tienen todavía 6 años de ventas antes de que sus ventas 
# lleguen a 0. 
# La plataforma 3DS lleva 6 años, quedando 4 de vigencia, sin embargo, las ventas ya van a la baja, y en 2016 ha disminuido en un
# 54.5% respecto a las ventas del 2015.
# 
# Sin embargo, sin tomar en cuenta los valores atípicos, tenemos que las plataformas con más ventas han sido GBA, PS4 y PSP.

# In[76]:


corr_scores_sales = df.groupby('platform').agg({'critic_score':'mean', 'user_score':'mean', 'total_sales': 'sum'}).reset_index()

corr_scores_sales


# In[77]:


plt.figure(figsize=(8, 8))
plt.scatter(corr_scores_sales['critic_score'], corr_scores_sales['total_sales'], color='blue', label='Calificación de críticos Vs Ventas totales')
plt.scatter(corr_scores_sales['user_score'], corr_scores_sales['total_sales'], color='red', label='Calificación de usuarios Vs Ventas totales')

plt.xlabel('Puntaje promedio')
plt.ylabel('Ventas totales')
plt.legend()
plt.grid(True)
plt.show()


# Basado en la gráfica de dispersión, puedo observar que la opinión, tanto de los críticos como de los usuarios, no interfiere 
# con las ventas. Hay plataformas que pueden tener calificaciones bajas y tener altas ventas, así como plataformas con
# calificaciones altas y ventas bajas.

# In[78]:


better_genres = df.groupby('genre')['total_sales'].sum()

print(better_genres.sort_values(ascending=False))


# In[79]:


better_games = df.groupby(['genre', 'name'])['total_sales'].sum()

print(better_games.sort_values(ascending=False))


# In[80]:


top_three_genres = better_genres.sort_values(ascending=False).head(5)

print('Los géneros más rentables son:')
print(top_three_genres)


# Los 5 géneros más rentagles son Action, Sports, Shooter, Role-playing y Platform.
# Los 5 videojuegos más vendidos son Wii Sports, Grand Theft Auto V  , Super Mario Bros., Tetris y Mario Kart Wii.
# 
# Wii Sports y Grand Theft Auto V (los juegos que más ventas han generado) pertenecen a los géneros más rentables:
#     Action - Grand Theft Auto V
#     Sports - Wii Sports

# In[81]:


top_platforms = df.groupby('platform')[['na_sales', 'eu_sales', 'jp_sales', 'other_sales', 'total_sales']].sum().reset_index()
top_five_platforms = top_platforms.nlargest(5, 'total_sales')

print('Las cinco plataformas más rentables son:')
print(top_five_platforms.sort_values(by='total_sales', ascending=False))


# In[82]:


top_five_platforms.plot(title='Las 5 plataformas más rentables por región', x='platform', style = 'o-', xlabel='Plataforma', ylabel='Ventas',
                        figsize=[20,20])
plt.show()


# En Norteamérica se da la mayor cantidad de ventas en las cinco plataformas principales, seguida de Europa.
# Las ventas en other_sales superan a las de Japón, región en donde se da la menor cantidad de ventas. Japón supera a Other sólo
# en las ventas de DS.

# In[83]:


top_genres = df.groupby('genre')[('na_sales', 'eu_sales', 'jp_sales', 'other_sales', 'total_sales')].sum().reset_index()
top_five_genres = top_genres.nlargest(5, 'total_sales')

print('Los cinco géneros más rentables son:')
print(top_five_genres.sort_values(by='total_sales', ascending=False))


# In[84]:


top_five_genres.plot(title='Los 5 géneros más rentables por región', x='genre', style = 'o-', xlabel='Género', ylabel='Ventas',
                        figsize=[20,20])
plt.show()


# La región de Norteamérica encabeza las ventas en 4 de los cinco géneros con más ventas. Sólo en el género Role-Playing es
# superado por Japón.
# Europa le sigue a Norteamérica, siendo superado por Japón solo en el género de Role-Playing.
# Other_sales supera a Japón en dos de cinco géneros.

# In[85]:


top_rating = df.groupby('rating')[('na_sales', 'eu_sales', 'jp_sales', 'other_sales', 'total_sales')].sum().reset_index()
top_five_rating = top_rating.nlargest(5, 'total_sales')

print('Las cinco clasificaciones más rentables son:')
print(top_five_rating.sort_values(by='total_sales', ascending=False))


# In[86]:


top_five_rating.plot(title='Las 5 clasificaciones más rentables por región', x='rating', style = 'o-', xlabel='Clasificación', 
                     ylabel='Ventas', figsize=[20,20])
plt.show()



# Los videojuegos con clasificación E (para todas las edades) son los preferidos en todas las regiones.
# 
# La clasificación T (para adolescentes) es la segunda preferida en Norteamérica, Europa y Japón. En el resto de las regiones,
# la segunda clasificación preferida es la M (para adolescentes de 17 años y mayores).

# # Prueba de Hipótesis

# In[87]:


xone = df[df['platform'] == 'XOne']
pc = df[df['platform'] == 'PC']


# In[88]:


xone_mean = xone['user_score'].mean()
xone_median = xone['user_score'].median()
print(xone_mean)
print(xone_median)


# In[89]:


pc_mean = pc['user_score'].mean()
pc_median = pc['user_score'].median()
print(pc_mean)
print(pc_median)


# In[90]:


xone['user_score'] = xone['user_score'].fillna(xone_mean)
xone


# In[91]:


print(xone['user_score'].isnull().sum())
print(xone['user_score'].mean())


# In[92]:


pc['user_score'] = pc['user_score'].fillna(pc_mean)
pc


# In[93]:


print(pc['user_score'].isnull().sum())
print(pc['user_score'].mean())


# In[94]:


action_df = df[df['genre'] == 'Action']


# In[95]:


action_mean = action_df['user_score'].mean()
action_median = action_df['user_score'].median()
print(action_mean)
print(action_median)


# In[96]:


action_df['user_score'] = action_df['user_score'].fillna(action_mean)


# In[97]:


print(action_df['user_score'].isnull().sum())
print(action_df['user_score'].mean())


# In[98]:


sports_df = df[df['genre'] == 'Sports']
sports_df


# In[99]:


sports_mean = sports_df['user_score'].mean()
sports_median = sports_df['user_score'].median()
print(sports_mean)
print(sports_median)


# In[100]:


sports_df['user_score'] = sports_df['user_score'].fillna(sports_mean)


# In[101]:


print(sports_df['user_score'].isnull().sum())
print(sports_df['user_score'].mean())


# ## Hipótesis nula: Las calificaciones promedio de los usuarios para las plataformas Xbox One y PC son las mismas.
#     
# ## Hipótesis alternativa: La clasificación promedio de los usuarios para las plataformas Xbox One y PC no son iguales.

# In[102]:


xone_score = xone['user_score']
pc_score = pc['user_score']

statistic, p_value = levene(xone_score, pc_score)

print("Estadístico de la prueba de Levene:", statistic)
print("Valor p:", p_value)

alfa = 0.05
if p_value < alfa:
    print("Hay evidencia para rechazar la hipótesis nula de igualdad de varianzas.")
else:
    print("No hay suficiente evidencia para rechazar la hipótesis nula de igualdad de varianzas.")


# In[103]:


alpha =.05 

results = st.ttest_ind(xone_score, pc_score, equal_var= False)

print('valor p:', results.pvalue)

if (results.pvalue < alpha):
    print("Rechazamos la hipótesis nula")
else:
    print("No podemos rechazar la hipótesis nula")


# Aun cuando las muestras no son iguales, los usuarios prefieren los videojuegos de PC sobre los de Xbox One.

# ## Hipótesis nula: Las calificaciones promedio de los usuarios para los géneros de Acción y Deportes son diferentes.
# 
# ## Hipótesis alternativa: Las calificaciones promedio de los usuarios de los géneros de Acción y Deportes son iguales.

# In[104]:


action = action_df['user_score']
sports = sports_df['user_score']

statistic, p_value = levene(action, sports)

print("Estadístico de la prueba de Levene:", statistic)
print("Valor p:", p_value)

alfa = 0.05
if p_value < alfa:
    print("Hay evidencia para rechazar la hipótesis nula de igualdad de varianzas.")
else:
    print("No hay suficiente evidencia para rechazar la hipótesis nula de igualdad de varianzas.")


# In[105]:


alpha =.05 

results = st.ttest_ind(action, sports, equal_var= True)

print('valor p:', results.pvalue)

if (results.pvalue < alpha):
    print("Rechazamos la hipótesis nula")
else:
    print("No podemos rechazar la hipótesis nula")


# Aun cuando las muestras no son iguales, los usuarios prefieren los videojuegos de acción sobre los de deportes.

# He aplicado la prueba de igualdad de varianzas antes de hacer la prueba de t de Student, para determinar si el valor de 
# equal_val debe ser True o False.
# Dados los resultados, he determinado que equal_var=False. 

# # CONCLUSIÓN

# In[106]:


top_games = df[(df['platform'].isin(['XOne', 'PS4'])) & 
               (df['genre'].isin(['Action', 'Sports'])) & 
               (df['rating'].isin(['E', 'T', 'M']))]

top_games.sort_values(by='total_sales', ascending = False)


# Las plataformas PS4 y XOne tienen 4 años y, según la tendencia, tienen todavía 6 años de ventas antes de que sus ventas lleguen a 0. La plataforma 3DS lleva 6 años, quedando 4 de vigencia, sin embargo, las ventas ya van a la baja, y en 2016 ha disminuido en un 54.5% respecto a las ventas del 2015.
# 
# Sin embargo, la plataforma PS4 tiene más ventas que la plataforma XOne.
# 
# Los géneros que más ventas generan son los de Action y Sports, además de aquellos que tienen la clasificación E, M y T.
# 
# El videojuego que cumple con esas características es 'Grand Theft Auto V', en la plataforma PS4, género: Action,
# clasificación: M y con más ventas.

