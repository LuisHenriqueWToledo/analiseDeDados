# Para o aprendizado de maquina, separar o data set em dois conjuntos: treino (X_train, y_train) e teste (X_test, y_test)

y = df_subset["Target"]

X = df_subset.copy()
X = X.drop("Target", axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# Crie uma instância de árvore de decisão chamada decision_tree e passe 0 para o parâmetro random_state.
# Isso serve apenas para que, se outros profissionais de dados executarem este código, obtenham os mesmos resultados.
# Ajuste o modelo ao conjunto de treinamento, use a função predict() no conjunto de teste e atribua essas previsões à variável dt_pred.

decision_tree = DecisionTreeClassifier(random_state=0)

decision_tree.fit(X_train, y_train)

dt_pred = decision_tree.predict(X_test)


# Mostre os valores alcançados pelo modelo nos quesitos: accuracy, precision, recall, and F1 score.

print("Decision Tree")
print("Accuracy:", "%.6f" % metrics.accuracy_score(y_test, dt_pred))
print("Precision:", "%.6f" % metrics.precision_score(y_test, dt_pred))
print("Recall:", "%.6f" % metrics.recall_score(y_test, dt_pred))
print("F1 Score:", "%.6f" % metrics.f1_score(y_test, dt_pred))


# Produza uma Matriz Confusao para ajudar a melhorar a interpretação dos erros obtidos pelo algoritmo.

cm = metrics.confusion_matrix(y_test, dt_pred, labels = decision_tree.classes_)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix = cm,display_labels = decision_tree.classes_)
disp.plot()


# Avalie a sua arvore de decisão usando a função plot_tree() para visualizar a arvore criada pelo modelo.

plt.figure(figsize=(20,12))
plot_tree(decision_tree, max_depth=2, fontsize=14, feature_names=X.columns);


# É interessante tambem verificar a distribuição dos dados pela importancia que eles possuem para a decisao
# Usando o atributo feature_importances_ para buscar as importâncias relativas de cada recurso, você pode então plotar os resultados.

importances = decision_tree.feature_importances_

forest_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)

fig, ax = plt.subplots()
forest_importances.plot.bar(ax=ax);


