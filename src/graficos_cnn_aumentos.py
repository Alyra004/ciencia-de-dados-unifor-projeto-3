"""Gráficos do efeito do aumento de dados nas CNNs (validação; o teste não é usado).

As CNNs foram treinadas com três receitas de aumento (nenhum, leve e forte), duas resoluções e duas rodadas
(hiperparâmetros iniciais e do Optuna). Lê resultados/experimentos.csv e grava em resultados/figuras/cnn/:
  aumentos_1_f1_por_receita.png     F1 macro de cada CNN por receita, um painel por resolução e rodada
  aumentos_2_efeito_medio.png       ganho médio (pontos de F1) de cada receita sobre o aumento leve, por família
  aumentos_3_custo.png              tempo por época de cada receita

Nomes dos treinos: <CNN>-<R>_preliminar (leve), <CNN>-<R>_semaug_preliminar (nenhum), <CNN>-<R>_aug2_preliminar (forte),
e as versões _opt_preliminar para os hiperparâmetros do Optuna.

Uso (a partir da raiz do projeto): python src/graficos_cnn_aumentos.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graficos_cnn import ARQS, FONTE, estilo, fmt, rodape  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
SAIDA = RAIZ / "resultados" / "figuras" / "cnn"
RECEITAS = [("nenhum", "#2a78d6", "Sem aumento"), ("leve", "#eb6834", "Aumento leve"), ("forte", "#1baf7a", "Aumento forte")]


def carregar():
    e = pd.read_csv(RAIZ / "resultados" / "experimentos.csv")
    e = e[e["nome_execucao"].str.match(r"^(ResNet|VGG)")].copy()
    e["arq"] = e["nome_execucao"].str.extract(r"^([A-Za-z0-9]+)-")[0]
    e["res"] = e["nome_execucao"].str.extract(r"-(\d+)_")[0].astype(int)
    e["rodada"] = np.where(e["nome_execucao"].str.contains("_opt_"), "Optuna", "inicial")
    e["aug"] = e["nome_execucao"].apply(lambda n: "nenhum" if "_semaug" in n else ("forte" if "_aug2" in n else "leve"))
    e["fam"] = e["arq"].str.extract(r"^(ResNet|VGG)")[0]
    return e


def main():
    estilo()
    SAIDA.mkdir(parents=True, exist_ok=True)
    e = carregar()

    # 1. F1 por receita
    fig, eixos = plt.subplots(2, 2, figsize=(15, 8.5), sharey=True)
    for ax, (res, rod) in zip(eixos.ravel(), [(128, "inicial"), (128, "Optuna"), (256, "inicial"), (256, "Optuna")]):
        x, larg = np.arange(len(ARQS)), 0.27
        for k, (aug, cor, rot) in enumerate(RECEITAS):
            d = e[(e.res == res) & (e.rodada == rod) & (e.aug == aug)].set_index("arq")["f1_macro_validacao"]
            vals = [d.get(a, np.nan) for a in ARQS]
            ax.bar(x + (k - 1) * larg, vals, larg, color=cor, label=rot)
        ax.set_xticks(x, ARQS, rotation=25, ha="right", fontsize=8)
        ax.set_ylim(0.8, 1.0)
        ax.set_title(f"{res} × {res}, hiperparâmetros {rod.lower()}", fontsize=10)
    eixos[0, 0].legend(frameon=False, fontsize=8, loc="lower right")
    for ax in eixos[:, 0]:
        ax.set_ylabel("F1 macro na validação")
    fig.suptitle("F1 macro de validação das 8 CNNs por receita de aumento de dados", fontweight="bold")
    rodape(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(SAIDA / "aumentos_1_f1_por_receita.png", dpi=150)
    plt.close(fig)

    # 2. efeito médio sobre o aumento leve
    ref = e[e.aug == "leve"].set_index(["arq", "res", "rodada"])["f1_macro_validacao"]
    linhas = []
    for aug in ("nenhum", "forte"):
        x_ = e[e.aug == aug].set_index(["arq", "res", "rodada"])["f1_macro_validacao"]
        d = (x_ - ref.reindex(x_.index)).dropna().rename("dif").reset_index()
        d["aug"], d["fam"] = aug, d["arq"].str.extract(r"^(ResNet|VGG)")[0]
        linhas.append(d)
    if linhas:
        d = pd.concat(linhas)
        grupos = [(f, r, rod) for f in ("ResNet", "VGG") for r in (128, 256) for rod in ("inicial", "Optuna")]
        fig, ax = plt.subplots(figsize=(13, 4.8))
        x, larg = np.arange(len(grupos)), 0.38
        for k, (aug, cor, rot) in enumerate([("nenhum", "#2a78d6", "Sem aumento − leve"), ("forte", "#1baf7a", "Aumento forte − leve")]):
            vals = [100 * d[(d.aug == aug) & (d.fam == f) & (d.res == r) & (d.rodada == rod)]["dif"].mean() for f, r, rod in grupos]
            ax.bar(x + (k - 0.5) * larg, vals, larg, color=cor, label=rot)
            for xi, v in zip(x + (k - 0.5) * larg, vals):
                if not np.isnan(v):
                    ax.text(xi, v + (0.15 if v >= 0 else -0.35), f"{v:+.1f}".replace(".", ","), ha="center", fontsize=8)
        ax.axhline(0, color="#555", lw=1)
        ax.set_xticks(x, [f"{f}\n{r} × {r}\n{rod.lower()}" for f, r, rod in grupos], fontsize=8)
        ax.set_ylabel("Diferença de F1 macro (pontos)")
        ax.set_title("Efeito médio do aumento de dados sobre o aumento leve (média das 4 CNNs de cada família)", fontweight="bold", fontsize=11)
        ax.legend(frameon=False)
        rodape(fig)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(SAIDA / "aumentos_2_efeito_medio.png", dpi=150)
        plt.close(fig)

    # 3. custo
    fig, eixos = plt.subplots(1, 2, figsize=(13, 4.6), sharey=False)
    for ax, res in zip(eixos, (128, 256)):
        x, larg = np.arange(len(ARQS)), 0.27
        for k, (aug, cor, rot) in enumerate(RECEITAS):
            d = e[(e.res == res) & (e.rodada == "inicial") & (e.aug == aug)].set_index("arq")["tempo_medio_epoca_s"]
            ax.bar(x + (k - 1) * larg, [d.get(a, np.nan) for a in ARQS], larg, color=cor, label=rot)
        ax.set_xticks(x, ARQS, rotation=25, ha="right", fontsize=8)
        ax.set_title(f"{res} × {res}, hiperparâmetros iniciais", fontsize=10)
        ax.set_ylabel("Segundos por época (A100)")
    eixos[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Tempo por época de cada receita de aumento (medido na A100)", fontweight="bold")
    rodape(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(SAIDA / "aumentos_3_custo.png", dpi=150)
    plt.close(fig)
    print("figuras em", SAIDA)


if __name__ == "__main__":
    main()
