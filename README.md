# Análise exploratória da qualidade de vinhos brancos

Trabalho da disciplina TI0175 - Inteligência Computacional Aplicada. O projeto analisa o conjunto **Wine Quality (vinho branco)**, do UCI Machine Learning Repository, relacionando propriedades físico-químicas à classe de qualidade atribuída ao vinho.

## Dados e análises

O conjunto possui **4.898 observações**, **11 preditores numéricos** e **7 classes** de qualidade (3 a 9). O trabalho inclui:

- descrição do conjunto e distribuição das classes;
- análises univariadas incondicional e condicionada à classe, com histogramas, boxplots, médias, desvios-padrão e assimetrias;
- análise bivariada com diagramas de dispersão e matriz de correlação de Pearson;
- PCA implementada manualmente, sem funções prontas, e projeção nos dois primeiros componentes principais.

Os gráficos e as tabelas produzidos ficam no diretório `figuras/`.

## Como executar

Requisitos: Python 3.9 ou superior e as bibliotecas `pandas`, `numpy`, `matplotlib` e `seaborn`. No Google Colab, a biblioteca `google-colab` já está disponível.

```bash
pip install pandas numpy matplotlib seaborn
python codigo_completo.py
```

O arquivo `codigo_completo.py` executa, em sequência, os quatro scripts da análise. O dataset deve permanecer em `wine+quality/winequality-white.csv`.

## Arquivos principais

- `codigo_completo.py`: execução geral.

## Contribuições dos autores

Preencher antes da entrega:

| Autor | Contribuição |
|---|---|
| André |  implementação do PCA, seção métodos |
| Lorena | análise estatística monovariada, geração de gráficos, seção introdução e resultados monovariados. |
| Thaís |  análise bivariada, seção resultados bivariados, multivariados e conclusão |

## Uso de inteligência artificial

A IA foi utilizada durante o projeto a fim de facilitar a organização do documento e para formatação correta do overleaf, mostrando-se extremamente útil para o preenchimento de tabelas com os dados apurados pelo código e para ajudar a manter o formato IEEE.
