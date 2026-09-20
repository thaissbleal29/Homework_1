# HW1 - Análise Bivariada: Matriz de Correlação em Imagem e Pairplot por Classe

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. Carregamento dos Dados ===
caminho_dataset = 'wine+quality/winequality-white.csv'
df = pd.read_csv(caminho_dataset, sep=';')

# Garantir que a pasta 'figuras' existe
os.makedirs('figuras', exist_ok=True)

# === 2. Identificação dos Preditores e Variável Alvo ===
coluna_classe = 'quality'
preditores = [col for col in df.columns if col != coluna_classe]
x = df[preditores]

print("=" * 80)
print("1. GERANDO IMAGEM DA MATRIZ DE CORRELAÇÃO DE PEARSON (ρ)")
print("=" * 80)

# === 3. Cálculo e Plotagem da Matriz de Correlação como Imagem ===
rho_matrix = x.corr(method='pearson')

plt.figure(figsize=(10, 8))

# Criar máscara para exibir apenas o triângulo inferior no Heatmap (opcional, deixa mais limpo)
mask = np.triu(np.ones_like(rho_matrix, dtype=bool))

sns.heatmap(
    rho_matrix,
    mask=mask,                           # Oculta o triângulo superior duplicado
    annot=True,                          # Escreve o valor do coeficiente em cada célula
    fmt=".2f",                           # Arredonda para 2 casas decimais
    cmap='coolwarm',                     # Azul (negativo), Branco (zero), Vermelho (positivo)
    vmin=-1, vmax=1,                     # Escala fixa da correlação
    square=True,                         # Mantém células quadradas
    linewidths=0.5,                      # Linhas separadoras entre células
    cbar_kws={"label": "Coeficiente de Correlação de Pearson (ρ)"}
)

plt.title('Matriz de Correlação de Pearson (ρ) entre Preditores', fontsize=14, pad=15)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

caminho_heatmap = 'figuras/matriz_correlacao_heatmap.png'
plt.savefig(caminho_heatmap, dpi=300, bbox_inches='tight')
print(f"-> Imagem da Matriz de Correlação salva com sucesso em: {caminho_heatmap}\n")
plt.close() # Fecha a figura atual para liberar memória antes do pairplot

print("=" * 80)
print("2. GERANDO MATRIZ VISUAL DE SCATTER PLOTS (PAIRPLOT)")
print("=" * 80)

# === 4. Configuração Visual dos Símbolos por Classe ===
classes_unicas = sorted(df[coluna_classe].unique())
simbolos = ['D', 'o', 's', '^', 'p', 'X', '*']
dicionario_simbolos = {
    classe: simbolos[idx % len(simbolos)] 
    for idx, classe in enumerate(classes_unicas)
}

sns.set_theme(style="ticks")

# === 5. Gerar o Pairplot Bivariado ===
g = sns.pairplot(
    df,
    vars=preditores,                   # Exibe apenas os preditores nos eixos
    hue=coluna_classe,                 # Agrupamento por classe de qualidade (cores)
    markers=dicionario_simbolos,       # Marcadores/símbolos distintos
    palette='Set1',                    # Paleta de cores com bom contraste
    corner=True,                       # Mantém apenas o triângulo inferior
    diag_kind='kde',                   # Curvas de densidade na diagonal principal
    plot_kws={'alpha': 0.6, 's': 15},  # Transparência e tamanho dos pontos
    diag_kws={'fill': True, 'alpha': 0.3}
)

# Ajustes na legenda e título
g.fig.suptitle('Análise Bivariada de Preditores por Classe de Qualidade', y=1.01, fontsize=14)
sns.move_legend(g, "center right", bbox_to_anchor=(0.98, 0.6), title='Qualidade')

# === 6. Salvar e Exibir o Pairplot ===
caminho_pairplot = 'figuras/scatter_matrix_preditores_classes.png'
plt.savefig(caminho_pairplot, dpi=300, bbox_inches='tight')
print(f"-> Imagem do Pairplot salva com sucesso em: {caminho_pairplot}\n")

plt.show()