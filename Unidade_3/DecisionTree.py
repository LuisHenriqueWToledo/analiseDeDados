# Standard operational package imports
import numpy as np
import pandas as pd

# Important imports for modeling and evaluation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import sklearn.metrics as metrics

# Visualization package imports
import matplotlib.pyplot as plt
import seaborn as sns


# código para carregar o dataset
# df --> data frame
df_original = pd.read_csv("ArquivoCriadoViaPrompt.csv")


# visualizar as primeiras 10 linhas. Sempre importante chamar essa linha de código para monitorar o que foi feito.
df_original.head(n = 10)


# verificar os tipos de cada dado para cada coluna.
df_original.dtypes


# verificar as categorias de uma das colunas preditoras para possibilidade de inferencias significativas.
df_original["Coluna_Selecionada"].unique()


# verificar a coluna "Target" para saber a contagem de cada valor:
df_original['Target'].value_counts(dropna = False)


#verificar valores nulos no data frame e sua soma
df_original.isnull().sum()


# verificar como está o data frame, quantas linhas e quantas colunas possuem.
df_original.shape


# em uma nova variavel para o data frame, receber o valor do data frame anterior retirados os valores nulos (Nan - not a number)
df_subset = df_original.dropna(axis=0).reset_index(drop = True)


# check novamente por valores nulos:
df_subset.isna().sum()

# check novamente quantas linhas e colunas permaneceram.
df_subset.shape

#Transformar valores tipo object ou string em numeros:
# é possivel usar o label encoder, o One Hot Key ou fazer manualmente como abaixo. o codigo irá substituir cada categoria em um numero para poder ser utilizado o modelo de IA.
df_subset['Coluna_Selecionada'] = df_subset['Coluna_Selecionada'].map({"Categoria_3": 3, "Categoria_2": 2, "Categoria_1": 1}) 


#Transformar tambem o data set em valores numericos:
df_subset['Target'] = df_subset['Target'].map({"Exemplo_EmpregoDecente": 1, "Exemplo_Emprego_Nao_Decente": 0})


#Verificar se ainda existem colunas categóricas no data set
df_subset = pd.get_dummies(df_subset, drop_first = True)


#check novamente pelos tipos
df_subset.dtypes

















