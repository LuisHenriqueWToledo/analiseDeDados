# Define a variavel y (Target)

y = extracted_data['Target']


# Define a variavel X (preditoras)

X = extracted_data.copy()
X = X.drop('Exited', axis=1)


#Split em treino e teste (75% para treino, podendo ser 80%)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, \ stratify=y, random_state=42)


# Finalmente podemos usar o modelo

gnb = GaussianNB() gnb.fit(X_train, y_train)


# Alcance as predicoes nos dados de teste. 

y_preds = gnb.predict(X_test)


# agora podemos verificar a performance do nosso modelo usando as metricas importadas

print('Accuracy:', '%.3f' % accuracy_score(y_test, y_preds))
print('Precision:', '%.3f' % precision_score(y_test, y_preds))
print('Recall:', '%.3f' % recall_score(y_test, y_preds))
print('F1 Score:', '%.3f' % f1_score(y_test, y_preds))


# validações avançadas caso alguma métrica for zero, ou seja, o numerador ou o denominador das formulas das métricas é zero.
# é necessario checar se não há algum vetor [0,1] na predição "y_preds", chamando a função np.unique()

np.unique(y_preds)

#>>> array([0])
# retornando zero, confirma que algo pode estar errado no modelo, mas é mais provavel que mais um ajuste nos dados resolva o problema.
# verifique as estatisticas
X.describe()

# uma possivel solução é usar o min e max scaler para ordenar os valores dentro de um intervalo.
# Importe a escala
from sklearn.preprocessing import MinMaxScaler

# Instancie a escala
scaler = MinMaxScaler()

# Fit a escala para os dados de treinamento
scaler.fit(X_train)

# Escale os dados de treinamento
X_train = scaler.transform(X_train)

# Escale os dados de teste
X_test = scaler.transform(X_test)

# Agora repita o processo do modelo de IA
# Fit do modelo
gnb_scaled = GaussianNB()
gnb_scaled.fit(X_train, y_train)

# Alcance as predições nos dados de teste.
scaled_preds = gnb_scaled.predict(X_test)

#exibir os resultados escalados.
print('Accuracy:', '%.3f' % accuracy_score(y_test, scaled_preds))
print('Precision:', '%.3f' % precision_score(y_test,scaled_preds))
print('Recall:', '%.3f' % recall_score(y_test, scaled_preds))
print('F1 Score:', '%.3f' % f1_score(y_test, scaled_preds))


#Por fim, fazer uma matriz confusão para validação dos resultados
def conf_matrix_plot(model, x_data, y_data):
  ''' Accepts as argument model object, X data (test or validate), and y data␣ →(test or validate). Return a plot of confusion matrix for predictions on y data. '''
  model_pred = model.predict(x_data)
  cm = confusion_matrix(y_data, model_pred, labels=model.classes_)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_, ) disp.plot(values_format='')
  
  # obs: `values_format=''` retira a notação científica.
  
  plt.show()

#chame a função:
conf_matrix_plot(gnb_scaled, X_test, y_test)
