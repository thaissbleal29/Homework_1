#HW1 - Análise exploratória e PCA do dataset Wine Quality

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import colors


#=== Função do PCA ===
def pca(x, n_componentes=2):
    #Padronizando os dados
    x_std = x.apply(lambda a: (a - a.mean()) / a.std())

    #Matriz de Covariância dos dados padronizados
    x_covm = x_std.cov()

    #Autovalores e autovetores da Matriz de Covariância
    autovalores, autovetores = np.linalg.eigh(x_covm)

    #Ordenando (os autovetores estão nas colunas, então ordena as colunas)
    i_ordenados = np.argsort(autovalores)[::-1]
    autovalores_ordenados = autovalores[i_ordenados]
    autovetores_ordenados = autovetores[:, i_ordenados]

    #Feature vector
    pca_components = autovetores_ordenados[:, :n_componentes]

    #Dados Finais
    projected_data = np.dot(x_std, pca_components)

    return projected_data, autovalores_ordenados, autovetores_ordenados, x_std


#=== Carregando os dados ===
df = pd.read_csv('wine+quality/winequality-white.csv', sep=';')

os.makedirs('figuras', exist_ok=True)

#info() - mostra informações sobre o DataFramepi
df.info()

#Estatísticas do DataFrame ajudam a identificar possíveis anomalias nos dados
print(df.describe())

#Checando por valores nulos no DataFrame (não há valores nulos)
print("Número de valores faltantes por variável: \n", df.isnull().sum())


#=== Características do dataset ===
#Separando preditores (X) e alvo (y)
x = df.drop('quality', axis=1)
y = df['quality']

x.info()

#N, D e L
N, D = x.shape
L = y.nunique()
print(f"Número de amostras (N) = {N},\n Número de atributos (D) = {D},\n Número de classes L = {L}")

#Class-distribution
print(y.value_counts().sort_index())


#=== Matriz de correlação ===
#Matriz de correlação dos preditores (dados originais)
x_corr = x.corr()
print("Correlações")
print(x_corr)

#Colormap customizado: azul (negativa), branco (zero) e vermelho (positiva)
colors_list = ['#2166AC', '#FFFFFF', '#B2182B']
cmap = colors.LinearSegmentedColormap.from_list('corr', colors_list)

#Plotando o heatmap com anotações
plt.figure(figsize=(9, 7))
plt.imshow(x_corr, cmap=cmap, vmin=-1, vmax=1)
for i in range(D):
    for j in range(D):
        plt.annotate(str(round(x_corr.values[i][j], 2)),
                     xy=(j, i),
                     ha='center', va='center', color='black', fontsize=7)

#Colorbar
cbar = plt.colorbar(ticks=[-1, -0.5, 0, 0.5, 1])
cbar.set_label('Correlação de Pearson')

#Título e rótulos dos eixos
plt.title("Matriz de correlação dos preditores")
plt.xlabel("Preditores")
plt.ylabel("Preditores")

#Rótulos dos ticks
plt.xticks(range(len(x_corr.columns)), x_corr.columns, rotation=90)
plt.yticks(range(len(x_corr.columns)), x_corr.columns)

plt.tight_layout()
plt.savefig('figuras/matriz_correlacao.png', dpi=300)
#plt.show()


#=== PCA ===
projected_data, autovalores_ordenados, autovetores_ordenados, x_std = pca(x, n_componentes=2)

print("Dados padronizados:")
print(x_std)

#Todo: Fazer boxplot dos preditores antes e depois da std

print("Covariancias")
print(x_std.cov())

print(f"Autovalores: \n{autovalores_ordenados}")
print(f"Autovetores: \n{autovetores_ordenados}")

print(f"Dados projetados: \n{projected_data}")
#Todo: Plotar os autovetores?

#Variancias
pc1_var = autovalores_ordenados[0]/autovalores_ordenados.sum()
pc2_var = autovalores_ordenados[1]/autovalores_ordenados.sum()

print("Variância PC1:", pc1_var*100, "%")
print("Variância PC2:", pc2_var*100, "%")

#Gráfico dos 2 primeiros componentes principais, por classe
plt.figure(figsize=(7, 6))
for classe in sorted(y.unique()):
    mask = (y == classe).values
    plt.scatter(projected_data[mask, 0], projected_data[mask, 1], s=15, alpha=0.6, label=str(classe))
plt.legend(title='Qualidade')
plt.title('Projeção nos dois primeiros componentes principais')
plt.xlabel(f"PC1 ({pc1_var*100:.1f}% da variância)")
plt.ylabel(f"PC2 ({pc2_var*100:.1f}% da variância)")
plt.savefig('figuras/pca_2d.png', dpi=300)
#plt.show()