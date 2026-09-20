
import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from google.colab import files

#lendo arquivo e definindo separador ; do dataset
df = pd.read_csv('wine+quality/winequality-white.csv', sep=';')

coluna_classe = 'quality'

#colunas com valor numerico diferenciadas da de classe
colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
if coluna_classe in colunas_numericas:
  colunas_numericas.remove(coluna_classe)

#estatisticas gerais
estatisticas_geral = pd.DataFrame({
    'Média (μ)': df[colunas_numericas].mean(),
    'Desvio Padrão (σ)': df[colunas_numericas].std(),
    'Assimetria (γ)': df[colunas_numericas].skew(),
})

#INCONDICIONAIS

print('Estatísticas Gerais')
display(estatisticas_geral.round(3))

for col in colunas_numericas:
  fig, axes = plt.subplots(1, 2, figsize=(12, 4))

  #histograma incondicional
  sns.histplot(
      data=df, x=col, color='#4a148c', kde=True, ax=axes[0], element='step'
  )
  axes[0].set_title(f'Distribuição Incondicional de {col}')

  #boxplot incondicional
  sns.boxplot(data=df, x=col, color='#7b1fa2', ax=axes[1])
  axes[1].set_title(f'Boxplot Incondicional de {col}')

  plt.tight_layout()
  plt.savefig(f'grafico_incondicional_{col}.png', dpi=300)
  plt.show()

#CONDICIONAIS
print('\n Estatísticas condicionais por classe')
estatisticas_classe = (
    df.groupby(coluna_classe)[colunas_numericas]
    .agg(['mean', 'std', 'skew'])
    .T
)
display(estatisticas_classe.round(3))

for col in colunas_numericas:
  fig, axes = plt.subplots(1, 2, figsize=(12, 4))

  #histogramas condicionais
  sns.histplot(
      data=df,
      x=col,
      hue=coluna_classe,
      palette='plasma',
      kde=True,
      ax=axes[0],
      element='step',
  )
  axes[0].set_title(f'Distribuição condicional de {col}')

  # boxplots condicionais
  sns.boxplot(data=df, x=coluna_classe, y=col, ax=axes[1])
  axes[1].set_title(f'Boxplot condicional de {col}')

  plt.tight_layout()
  plt.savefig(f'grafico_{col}.png', dpi=300)
  plt.show()
