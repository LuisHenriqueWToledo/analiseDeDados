# Define a variavel y (Target)

y = extracted_data['Target']


# Define a variavel X (preditoras)

X = extracted_data.copy()
X = X.drop('Exited', axis=1)


#Split em treino e teste (75% para treino, podendo ser 80%)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, \ stratify=y, random_state=42)

