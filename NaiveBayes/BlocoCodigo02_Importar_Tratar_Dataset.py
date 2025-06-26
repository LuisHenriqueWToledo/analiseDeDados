# Importar o Dataset
# Carregue o arquivo (.csv) criado anteriormente para o Dataframe chamado: 
# OBS: o arquivo_criado.csv deve estar na sua pasta de trabalho ou indicar o caminho de onde estiver dentro das aspas.

extracted_data = pd.read_csv('arquivo_criado.csv')


# Mostrar as primeiras 10 linhas de dados.

extracted_data.head(10)


# Defina y (minusculo) como a variável resultante (target).

y = extracted_data['Target']


# Defina X (maiúsculo) como as variaveis preditoras (predictor).

X = extracted_data.drop('Target', axis = 1)


# Mostrar as primeiras 10 linhas de dados do alvo ('Target').

y.head(10)


# Mostrar as primeiras 10 linhas de dados da variaveis preditoras.

X.head(10)
