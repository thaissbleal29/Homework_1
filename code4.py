# HW1 - Análise Multivariada Incondicional via PCA (Implementação Manual)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. Carregamento dos Dados ===
caminho_dataset = 'wine+quality/winequality-white.csv'
df = pd.read_csv(caminho_dataset, sep=';')

# Criar diretório para salvar a figura
os.makedirs('figuras', exist_ok=True)

# Separar Preditores (X) e Alvo (y)
coluna_classe = 'quality'
X = df.drop(coluna_classe, axis=1)
y = df[coluna_classe]

# === 2. Pré-processamento: Padronização Z-score ===
# A padronização é essencial para o PCA, garantindo média = 0 e variância = 1
# evitando que variáveis com escalas maiores dominem os componentes.
X_std = (X - X.mean()) / X.std()

# === 3. Implementação Manual do PCA (Álgebra Linear) ===
# a) Matriz de Covariância dos dados padronizados
cov_matrix = X_std.cov()

# b) Decomposição em Autovalores e Autovetores
autovalores, autovetores = np.linalg.eigh(cov_matrix)

# c) Ordenação decrescente (maior variância primeiro)
indices_ordenados = np.argsort(autovalores)[::-1]
autovalores_ordenados = autovalores[indices_ordenados]
autovetores_ordenados = autovetores[:, indices_ordenados]

# d) Seleção dos 2 primeiros Componentes Principais
n_componentes = 2
matriz_projecao = autovetores_ordenados[:, :n_componentes]

# e) Projeção dos dados padronizados no espaço 2D dos PCs
X_projected = np.dot(X_std.values, matriz_projecao)

# f) Cálculo da variância explicada
var_explicada = autovalores_ordenados / np.sum(autovalores_ordenados)
pc1_var = var_explicada[0] * 100
pc2_var = var_explicada[1] * 100

print(f"Variância explicada por PC1: {pc1_var:.2f}%")
print(f"Variância explicada por PC2: {pc2_var:.2f}%")
print(f"Variância acumulada dos 2 PCs: {pc1_var + pc2_var:.2f}%")

# === 4. Visualização Gráfica com Cores e Símbolos por Classe ===
classes_unicas = sorted(y.unique())
simbolos = ['D', 'o', 's', '^', 'p', 'X', '*'] # Marcadores distintos
paleta_cores = sns.color_palette("Set1", len(classes_unicas))

plt.figure(figsize=(10, 7))
sns.set_theme(style="ticks")

for idx, classe in enumerate(classes_unicas):
    mask = (y == classe).values
    plt.scatter(
        X_projected[mask, 0],
        X_projected[mask, 1],
        label=f'Qualidade {classe}',
        color=paleta_cores[idx],
        marker=simbolos[idx % len(simbolos)],
        s=25,
        alpha=0.6,
        edgecolor='w',
        linewidth=0.3
    )

plt.title('Projeção PCA 2D dos Preditores (Cores e Símbolos por Classe)', fontsize=13, pad=12)
plt.xlabel(f"Primeiro Componente Principal - PC1 ({pc1_var:.1f}% da variância)")
plt.ylabel(f"Segundo Componente Principal - PC2 ({pc2_var:.1f}% da variância)")
plt.legend(title='Classe (Qualidade)', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

caminho_figura = 'figuras/pca_2d_classes.png'
plt.savefig(caminho_figura, dpi=300, bbox_inches='tight')
print(f"Gráfico do PCA salvo com sucesso em: {caminho_figura}")

plt.show()