# Importar o Dataset
# Carregue o arquivo (.csv) criado anteriormente para o Dataframe chamado: 
# OBS: o arquivo_criado.csv deve estar na sua pasta de trabalho ou indicar o caminho de onde estiver dentro das aspas.

df_extracted_data = pd.read_csv('arquivo_criado.csv')

# Exibir informações de alto nivel dos dados.

df_extracted_data.info()


# Criar um novo data frame sem as colunas indesejadas. Exemplo: Raca, Numero_cpf, sexo (caso seja motivo de aumento do viés).

extracted_data = df_extracted_data.drop(['Raca', 'Numero_cpf'], 
                            axis=1)

# Dependendo do seu data set, pode-se criar uma nova coluna com base na união de duas outras. Depende da sua interpretação do que pode ser importante.
# Criar variavel "Lealdade":
extracted_data = extracted_data['Lealdade'] = extracted_data['Tempo_cliente'] / churn_df['Idade']

# Exibir valores unicos de uma localidade, por exemplo. Quer saber quais localidades existem.
# No caso hipotetico abaixo, apos o comando, resulta em 3 localidades.

extracted_data['Localidade'].unique()
#>>> array (['Gama', 'Taguatinga', 'Ceilandia'], dtype=objetc)

# a função Dummy codifica exatamente essas variaveis categoricas, formando novas colunas no data set.
# pode ajustar o drop_first = False para nao descartar a primeira categoria.

extracted_data = pd.get_dummies(extracted_data, drop_first=True)


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
