import pandas as pd
import numpy as np
import random
import os

# --- Configurações do Dataset ---
NUM_LINHAS = 10000

# Possíveis categorias para as colunas
idades = np.random.randint(18, 65, NUM_LINHAS)
niveis_educacao = [
	'Médio Incompleto', 'Médio Completo', 'Ensino Técnico',
	'Superior Incompleto', 'Superior Completo', 'Pós-Graduação'
]
experiencias = np.random.randint(0, 40, NUM_LINHAS)
rendas = np.random.randint(1000, 15000, NUM_LINHAS) # Renda em BRL
setores_atividade = [
	'Comércio', 'Tecnologia', 'Serviços', 'Construção', 'Saúde',
	'Educação', 'Alimentos', 'Finanças', 'Indústria', 'Varejo',
	'Transporte', 'Beleza', 'Engenharia', 'Agrícola', 'Logística',
	'Pesquisa', 'Turismo', 'Comunicação', 'Segurança', 'Manutenção'
]
distancias_residencia = np.random.randint(1, 50, NUM_LINHAS) # Distância em KM

# --- Criação do DataFrame Inicial ---
df = pd.DataFrame({
	'Idade': idades,
	'Nivel_Educacao': np.random.choice(niveis_educacao, NUM_LINHAS),
	'Experiencia_Anos': experiencias,
	'Renda_Mensal_BRL': rendas,
	'Setor_Atividade': np.random.choice(setores_atividade, NUM_LINHAS),
	'Distancia_Residencia_KM': distancias_residencia
})

# --- Lógica para Definir o 'Target' ---
def definir_target(row):
	# Condições que indicam "Trabalho Indigno"
	if row['Renda_Mensal_BRL'] < 1500 and row['Nivel_Educacao'] == 'Médio Incompleto':
    	return 'Trabalho Indigno (Informal)'
	elif row['Renda_Mensal_BRL'] < 1800 and row['Setor_Atividade'] == 'Agrícola' and row['Distancia_Residencia_KM'] > 10:
    	return 'Trabalho Indigno (Condições Precárias)'
	elif row['Distancia_Residencia_KM'] > 25 and row['Renda_Mensal_BRL'] < 2500:
    	# Longo deslocamento com baixa renda, sugerindo horas excessivas de trajeto
    	return 'Trabalho Indigno (Horas Excessivas)'
	elif row['Experiencia_Anos'] > 15 and row['Renda_Mensal_BRL'] < 2000:
    	return 'Trabalho Indigno (Baixa Remuneração Crônica)'
	elif row['Setor_Atividade'] in ['Construção', 'Segurança'] and row['Renda_Mensal_BRL'] < 2800:
    	return 'Trabalho Decente (Risco)' # Pode ser decente mas com risco elevado

	# Condições para "Trabalho Decente" com nuances
	elif row['Renda_Mensal_BRL'] < 2200 and row['Nivel_Educacao'] in ['Médio Completo', 'Ensino Técnico']:
    	return 'Trabalho Decente (Renda Baixa)'
	elif row['Renda_Mensal_BRL'] >= 5000 and row['Nivel_Educacao'] in ['Superior Completo', 'Pós-Graduação'] and row['Experiencia_Anos'] >= 5:
    	return 'Trabalho Decente'
	elif row['Renda_Mensal_BRL'] >= 3000 and row['Distancia_Residencia_KM'] <= 10:
    	return 'Trabalho Decente' # Boa renda e deslocamento razoável

	# Caso não se encaixe nas condições acima, atribui um valor padrão ou aleatório com base na renda
	if row['Renda_Mensal_BRL'] < 2500:
    	return random.choice(['Trabalho Decente (Renda Baixa)', 'Trabalho Indigno (Informal)'])
	else:
    	return random.choice(['Trabalho Decente', 'Trabalho Decente (Risco)'])

# Aplicando a função para criar a coluna 'Target'
df['Target'] = df.apply(definir_target, axis=1)

# --- Exportação para CSV ---
nome_arquivo = 'dataset_trabalho_decente_10000_linhas.csv'
caminho_completo_arquivo = os.path.join(os.getcwd(), nome_arquivo)

df.to_csv(caminho_completo_arquivo, index=False)

print(f"Dataset com {NUM_LINHAS} linhas gerado com sucesso!")
print(f"Arquivo '{nome_arquivo}' salvo em: {caminho_completo_arquivo}")
