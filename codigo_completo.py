"""HW1 - Análise exploratória completa do Wine Quality"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ARQUIVO_DADOS = Path("wine+quality/winequality-white.csv")
PASTA_FIGURAS = Path("figuras")
COLUNA_CLASSE = "quality"


def carregar_dados() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Carrega o dataset e separa preditores e classe."""
    # === Carregamento dos dados ===
    # Lendo o arquivo e definindo ";" como separador do dataset.
    df = pd.read_csv(ARQUIVO_DADOS, sep=";")

    # === Identificação dos preditores e da variável-alvo ===
    # Separando preditores (X) e alvo (y).
    x = df.drop(columns=COLUNA_CLASSE)
    y = df[COLUNA_CLASSE]
    return df, x, y


def analise_inicial(df: pd.DataFrame, x: pd.DataFrame, y: pd.Series) -> None:
    """Exibe dimensões, tipos, estatísticas e dados faltantes."""
    print("\n" + "=" * 80)
    print("1. ANÁLISE INICIAL")
    print("=" * 80)
    # info() mostra informações sobre as colunas do DataFrame.
    df.info()

    # As estatísticas do DataFrame ajudam a identificar possíveis anomalias.
    print("\nEstatísticas descritivas:\n", df.describe().round(3))

    # Checando valores nulos no DataFrame (não há valores nulos neste dataset).
    print("\nValores faltantes:\n", df.isnull().sum())

    # N, D e L: amostras, atributos e classes, respectivamente.
    n_amostras, n_atributos = x.shape
    n_classes = y.nunique()
    print(f"\nNúmero de amostras (N): {n_amostras}")
    print(f"Número de atributos (D): {n_atributos}")
    print(f"Número de classes (L): {n_classes}")
    # Class distribution.
    print("\nDistribuição das classes:\n", y.value_counts().sort_index())


def analise_univariada(df: pd.DataFrame, preditores: list[str]) -> None:
    """Produz estatísticas e gráficos uni variados, com e sem classe."""
    print("\n" + "=" * 80)
    print("2. ANÁLISE UNIVARIADA")
    print("=" * 80)

    pasta = PASTA_FIGURAS / "univariada"
    pasta.mkdir(parents=True, exist_ok=True)

    # === Análise incondicional ===
    # Estatísticas gerais dos preditores, sem separação pela classe.
    estatisticas_gerais = pd.DataFrame(
        {
            "media": df[preditores].mean(),
            "desvio_padrao": df[preditores].std(),
            "assimetria": df[preditores].skew(),
        }
    )
    estatisticas_gerais.to_csv(pasta / "estatisticas_incondicionais.csv")
    print("\nEstatísticas incondicionais:\n", estatisticas_gerais.round(3))

    # === Análise condicional ===
    # Estatísticas dos preditores agrupadas pela classe de qualidade.
    estatisticas_classe = (
        df.groupby(COLUNA_CLASSE, observed=False)[preditores]
        .agg(["mean", "std", "skew"])
        .T
    )
    estatisticas_classe.to_csv(pasta / "estatisticas_condicionais.csv")
    print("\nEstatísticas condicionais por classe:\n", estatisticas_classe.round(3))

    for coluna in preditores:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))

        # Histograma incondicional.
        sns.histplot(data=df, x=coluna, kde=True, color="#4a148c", ax=axes[0, 0])
        axes[0, 0].set_title(f"Distribuição incondicional de {coluna}")

        # Boxplot incondicional.
        sns.boxplot(data=df, x=coluna, color="#7b1fa2", ax=axes[0, 1])
        axes[0, 1].set_title(f"Boxplot incondicional de {coluna}")

        # Histograma condicional: uma distribuição para cada classe.
        sns.histplot(
            data=df,
            x=coluna,
            hue=COLUNA_CLASSE,
            palette="plasma",
            kde=True,
            element="step",
            ax=axes[1, 0],
        )
        axes[1, 0].set_title(f"Distribuição de {coluna} por classe")

        # Boxplot condicional por classe de qualidade.
        sns.boxplot(data=df, x=COLUNA_CLASSE, y=coluna, ax=axes[1, 1])
        axes[1, 1].set_title(f"Boxplot de {coluna} por classe")

        fig.tight_layout()
        fig.savefig(pasta / f"univariada_{coluna}.png", dpi=300)
        plt.close(fig)


def analise_bivariada(df: pd.DataFrame, x: pd.DataFrame) -> None:
    """Analisa as relações entre pares de preditores."""
    print("\n" + "=" * 80)
    print("3. ANÁLISE BIVARIADA")
    print("=" * 80)

    pasta = PASTA_FIGURAS / "bivariada"
    pasta.mkdir(parents=True, exist_ok=True)
    # === Cálculo da matriz de correlação ===
    # Matriz de correlação dos preditores usando o coeficiente de Pearson.
    correlacao = x.corr(method="pearson")
    correlacao.to_csv(pasta / "matriz_correlacao_pearson.csv")
    print("\nMatriz de correlação de Pearson:\n", correlacao.round(3))

    # Criar máscara para exibir somente o triângulo inferior do heatmap,
    # evitando a repetição dos mesmos coeficientes.
    mascara = np.triu(np.ones_like(correlacao, dtype=bool))
    plt.figure(figsize=(11, 9))
    sns.heatmap(
        correlacao,
        mask=mascara,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Correlação de Pearson"},
    )
    plt.title("Matriz de correlação entre os preditores")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(pasta / "matriz_correlacao_heatmap.png", dpi=300)
    plt.close()

    # === Configuração visual dos símbolos por classe ===
    classes = sorted(df[COLUNA_CLASSE].unique())
    marcadores = ["D", "o", "s", "^", "p", "X", "*"]
    mapa_marcadores = {
        classe: marcadores[i % len(marcadores)] for i, classe in enumerate(classes)
    }

    # === Matriz visual de scatter plots (pairplot) ===
    # Exibe apenas os preditores nos eixos, usando cores e marcadores para
    # distinguir as classes de qualidade.
    grade = sns.pairplot(
        df,
        vars=x.columns.tolist(),
        hue=COLUNA_CLASSE,
        markers=mapa_marcadores,
        palette="Set1",
        corner=True,
        diag_kind="hist",
        plot_kws={"alpha": 0.45, "s": 12},
        diag_kws={"alpha": 0.35},
    )
    grade.fig.suptitle("Relações bivariadas por classe de qualidade", y=1.01)
    grade.fig.savefig(
        pasta / "scatter_matrix_preditores_classes.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(grade.fig)

    # === Pares representativos para inclusão no artigo ===
    # A matriz completa é mantida como resultado complementar, enquanto esta
    # figura compacta apresenta as três relações de maior interesse.
    pares_representativos = [
        ("residual sugar", "density"),
        ("alcohol", "density"),
        ("free sulfur dioxide", "total sulfur dioxide"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (preditor_x, preditor_y) in zip(axes, pares_representativos):
        sns.scatterplot(
            data=df,
            x=preditor_x,
            y=preditor_y,
            hue=COLUNA_CLASSE,
            palette="Set1",
            alpha=0.5,
            s=15,
            legend=False,
            ax=ax,
        )
        rho = correlacao.loc[preditor_x, preditor_y]
        ax.set_title(rf"$\rho = {rho:.3f}$")

    fig.tight_layout()
    fig.savefig(
        pasta / "pares_representativos.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def calcular_pca(x: pd.DataFrame, n_componentes: int = 2):
    """Calcula manualmente o PCA a partir da matriz de covariância."""
    # === Pré-processamento: padronização Z-score ===
    x_padronizado = (x - x.mean()) / x.std()

    # === Implementação manual do PCA por álgebra linear ===
    # Matriz de covariância dos dados padronizados.
    matriz_covariancia = x_padronizado.cov()

    # Decomposição da matriz em autovalores e autovetores.
    autovalores, autovetores = np.linalg.eigh(matriz_covariancia)

    # Ordenação decrescente, destacando os componentes de maior variância primeiro.
    # Os autovetores estão nas colunas, portanto suas colunas são reordenadas.
    ordem = np.argsort(autovalores)[::-1]
    autovalores = autovalores[ordem]
    autovetores = autovetores[:, ordem]

    # Seleção dos primeiros componentes principais.
    componentes = autovetores[:, :n_componentes]

    # Projeção dos dados padronizados no espaço dos componentes.
    dados_projetados = x_padronizado.to_numpy() @ componentes

    # Cálculo da proporção de variância explicada por componente.
    variancia_explicada = autovalores / autovalores.sum()

    return (
        dados_projetados,
        autovalores,
        autovetores,
        variancia_explicada,
        x_padronizado,
    )


def analise_multivariada(x: pd.DataFrame, y: pd.Series) -> None:
    """Executa o PCA e apresenta a projeção nos dois primeiros componentes."""
    print("\n" + "=" * 80)
    print("4. ANÁLISE MULTIVARIADA INCONDICIONAL - PCA")
    print("=" * 80)

    pasta = PASTA_FIGURAS / "pca"
    pasta.mkdir(parents=True, exist_ok=True)
    # Execução do PCA manual com dois componentes principais.
    projetados, autovalores, autovetores, variancia, x_std = calcular_pca(x)

    print("\nMatriz de covariância:\n", x_std.cov().round(3))
    print("\nAutovalores:\n", autovalores)
    print("\nAutovetores:\n", autovetores)
    print(f"\nVariância explicada por PC1: {variancia[0] * 100:.2f}%")
    print(f"Variância explicada por PC2: {variancia[1] * 100:.2f}%")
    print(f"Variância acumulada: {variancia[:2].sum() * 100:.2f}%")

    pd.DataFrame(
        projetados,
        columns=["PC1", "PC2"],
        index=x.index,
    ).assign(quality=y).to_csv(pasta / "dados_projetados.csv", index=False)

    # === Visualização gráfica com cores e símbolos por classe ===
    classes = sorted(y.unique())
    marcadores = ["D", "o", "s", "^", "p", "X", "*"]
    cores = sns.color_palette("Set1", len(classes))
    plt.figure(figsize=(10, 7))

    for i, classe in enumerate(classes):
        filtro = (y == classe).to_numpy()
        plt.scatter(
            projetados[filtro, 0],
            projetados[filtro, 1],
            label=f"Qualidade {classe}",
            color=cores[i],
            marker=marcadores[i % len(marcadores)],
            s=25,
            alpha=0.6,
            edgecolor="white",
            linewidth=0.3,
        )

    plt.title("Projeção PCA dos preditores")
    plt.xlabel(f"PC1 ({variancia[0] * 100:.1f}% da variância)")
    plt.ylabel(f"PC2 ({variancia[1] * 100:.1f}% da variância)")
    plt.legend(title="Classe", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.grid(linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(pasta / "pca_2d_classes.png", dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    # Configuração visual comum a todos os gráficos.
    sns.set_theme(style="ticks")
    PASTA_FIGURAS.mkdir(exist_ok=True)

    # O dataset é carregado apenas uma vez e compartilhado por todas as etapas.
    df, x, y = carregar_dados()
    preditores = x.columns.tolist()

    # Execução sequencial das quatro partes do trabalho.
    analise_inicial(df, x, y)
    analise_univariada(df, preditores)
    analise_bivariada(df, x)
    analise_multivariada(x, y)
    print(f"\nAnálise concluída. Arquivos salvos em: {PASTA_FIGURAS.resolve()}")


if __name__ == "__main__":
    main()
