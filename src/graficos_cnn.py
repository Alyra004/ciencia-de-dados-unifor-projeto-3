"""Gráficos de resumo das CNNs (validação; o teste não é usado).

Lê resultados/experimentos.csv, resultados/historicos/, resultados/matrizes_confusao/ e
resultados/metricas_por_classe/ (gerados por notebooks/03_CNNs.ipynb e src/avaliar_cnn_validacao.py)
e grava em resultados/figuras/cnn/.

Uso (a partir da raiz do projeto): python src/graficos_cnn.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

RAIZ = Path(__file__).resolve().parent.parent
RES = RAIZ / "resultados"
SAIDA = RES / "figuras" / "cnn"
CLASSES = ("airport", "beach", "circular_farmland", "dense_residential", "desert",
           "forest", "harbor", "mountain", "river", "storage_tank")
ARQS = ["ResNet18", "ResNet18SE", "ResNet18CBAM", "ResNet18SECBAM", "VGG16", "VGG16SE", "VGG16CBAM", "VGG16SECBAM"]
FONTE = "Fonte: elaborado pelos autores (NWPU-RESISC45, 10 classes). Dados de validação; o teste não foi usado."
COR = {"ResNet18": "#2a78d6", "VGG16": "#eb6834"}
virgula = FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ","))


def fmt(v, casas=3):
    return f"{v:.{casas}f}".replace(".", ",")


def estilo():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "figure.facecolor": "white"})


def familia(arq):
    return "ResNet18" if arq.startswith("ResNet") else "VGG16"


def rodape(fig):
    fig.text(0.01, 0.005, FONTE, fontsize=8, color="#666")


def carregar():
    exp = pd.read_csv(RES / "experimentos.csv")
    exp = exp[exp["nome_execucao"].str.match(r"^(ResNet|VGG)")].copy()
    exp["arq"] = exp["nome_execucao"].str.extract(r"^([A-Za-z0-9]+)-")[0]
    exp["res"] = exp["nome_execucao"].str.extract(r"-(\d+)_")[0].astype(int)
    exp["rodada"] = np.where(exp["nome_execucao"].str.contains("_opt_"), "Optuna", "Inicial")
    return exp.set_index("nome_execucao")


def nome(arq, res, rodada):
    return f"{arq}-{res}_opt_preliminar" if rodada == "Optuna" else f"{arq}-{res}_preliminar"


def grafico_f1_antes_depois(exp, res):
    fig, ax = plt.subplots(figsize=(12, 5.2))
    x, larg = np.arange(len(ARQS)), 0.38
    for k, (rodada, cor, alfa) in enumerate([("Inicial", "#b9b8b3", 1), ("Optuna", None, 1)]):
        vals = [exp.loc[nome(a, res, rodada), "f1_macro_validacao"] if nome(a, res, rodada) in exp.index else np.nan for a in ARQS]
        cores = [cor or COR[familia(a)] for a in ARQS]
        b = ax.bar(x + (k - 0.5) * larg, vals, larg, color=cores, label=rodada)
        for xi, v in zip(x + (k - 0.5) * larg, vals):
            if not np.isnan(v):
                ax.text(xi, v + 0.004, fmt(v), ha="center", fontsize=8)
    ax.set_xticks(x, ARQS, rotation=20)
    ax.set_ylim(0.6, 1.0)
    ax.yaxis.set_major_formatter(virgula)
    ax.set_ylabel("F1 macro na validação")
    ax.set_title(f"F1 macro de validação das 8 CNNs, {res} × {res}: hiperparâmetros iniciais (cinza) × Optuna (cor)", fontweight="bold", fontsize=11)
    rodape(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(SAIDA / f"resumo_1_f1_antes_depois_{res}.png", dpi=150)
    plt.close(fig)


def grafico_custo(exp, res):
    fig, eixos = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, rodada in zip(eixos, ["Inicial", "Optuna"]):
        for a in ARQS:
            n = nome(a, res, rodada)
            if n not in exp.index:
                continue
            r = exp.loc[n]
            ax.scatter(r["tempo_total_s"], r["f1_macro_validacao"], s=r["parametros"] / 1e5, color=COR[familia(a)], alpha=0.75, edgecolor="white")
            ax.annotate(a.replace("ResNet18", "R18").replace("VGG16", "V16"), (r["tempo_total_s"], r["f1_macro_validacao"]),
                        textcoords="offset points", xytext=(6, 5), fontsize=8)
        ax.set_title(f"Hiperparâmetros: {rodada.lower()}", fontsize=10)
        ax.set_xlabel("Tempo total de treino (s)")
        ax.yaxis.set_major_formatter(virgula)
    eixos[0].set_ylabel("F1 macro na validação")
    fig.suptitle(f"Custo × desempenho, {res} × {res} (área = nº de parâmetros; azul = ResNet-18, laranja = VGG-16)", fontweight="bold", fontsize=11)
    rodape(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(SAIDA / f"resumo_2_custo_desempenho_{res}.png", dpi=150)
    plt.close(fig)


def f1_por_classe(n):
    p = RES / "metricas_por_classe" / f"{n}.csv"
    if not p.is_file():
        return None
    return pd.read_csv(p, index_col=0).loc[list(CLASSES), "f1-score"]


def grafico_f1_classe(exp, res):
    linhas = {}
    for rodada in ("Inicial", "Optuna"):
        for a in ARQS:
            n = nome(a, res, rodada)
            s = f1_por_classe(n)
            if s is not None:
                linhas[f"{a} ({rodada})"] = s
    if not linhas:
        return
    m = pd.DataFrame(linhas).T
    fig, ax = plt.subplots(figsize=(12, 0.42 * len(m) + 2))
    im = ax.imshow(m.values, cmap="viridis", vmin=0.6, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=35, ha="right")
    ax.set_yticks(range(len(m)), m.index)
    ax.grid(False)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, fmt(m.values[i, j], 2), ha="center", va="center", fontsize=7.5, color="white" if m.values[i, j] < 0.85 else "black")
    fig.colorbar(im, ax=ax, label="F1 da classe")
    ax.set_title(f"F1 por classe na validação, {res} × {res} (checkpoint de cada treino)", fontweight="bold", fontsize=11)
    rodape(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(SAIDA / f"resumo_3_f1_por_classe_{res}.png", dpi=150)
    plt.close(fig)


def grafico_curvas(exp, res):
    fig, eixos = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
    for i, fam in enumerate(["ResNet18", "VGG16"]):
        for j, rodada in enumerate(["Inicial", "Optuna"]):
            ax = eixos[i, j]
            for k, a in enumerate([x for x in ARQS if familia(x) == fam]):
                n = nome(a, res, rodada)
                p = RES / "historicos" / f"{n}_historico.csv"
                if not p.is_file():
                    continue
                h = pd.read_csv(p)
                cor = plt.cm.tab10(k)
                ax.plot(h["epoca"], h["f1_macro_validacao"], color=cor, lw=1.6, label=a)
                ck = h[h["checkpoint"]].iloc[0]
                ax.scatter([ck["epoca"]], [ck["f1_macro_validacao"]], color=cor, zorder=3, s=36)
            ax.set_title(f"{fam}, hiperparâmetros {rodada.lower()}", fontsize=10)
            ax.yaxis.set_major_formatter(virgula)
            ax.set_xlabel("Época")
            if j == 0:
                ax.set_ylabel("F1 macro na validação")
            ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(f"F1 macro de validação por época, {res} × {res} (ponto = checkpoint)", fontweight="bold", fontsize=11)
    rodape(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(SAIDA / f"resumo_4_curvas_validacao_{res}.png", dpi=150)
    plt.close(fig)


def carregar_cm(n):
    p = RES / "matrizes_confusao" / f"{n}_cm.csv"
    return pd.read_csv(p, index_col=0).values if p.is_file() else None


def desenhar_cm(ax, cm, normalizar, titulo, anotar=True, rotulos=True):
    m = cm / cm.sum(1, keepdims=True) if normalizar else cm
    ax.imshow(m, cmap="Blues", vmin=0, vmax=1 if normalizar else None)
    ax.grid(False)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(CLASSES)), CLASSES, fontsize=7)
    if rotulos:
        ax.set_xlabel("Classe prevista", fontsize=8)
        ax.set_ylabel("Classe real", fontsize=8)
    ax.set_title(titulo, fontsize=9)
    if anotar:
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                v = m[i, j]
                txt = fmt(v, 2) if normalizar else f"{int(v)}"
                if (normalizar and v < 0.005) or (not normalizar and v == 0):
                    continue
                ax.text(j, i, txt, ha="center", va="center", fontsize=6.5, color="white" if v > (0.5 if normalizar else m.max() / 2) else "black")


def graficos_matrizes(exp):
    (SAIDA / "matrizes").mkdir(parents=True, exist_ok=True)
    for n in exp.index:
        cm = carregar_cm(n)
        if cm is None:
            continue
        r = exp.loc[n]
        fig, eixos = plt.subplots(1, 2, figsize=(13, 6.2))
        desenhar_cm(eixos[0], cm, False, "Contagens")
        desenhar_cm(eixos[1], cm, True, "Normalizada pela classe real (linhas somam 1)")
        fig.suptitle(f"{n} — matriz de confusão na validação (reais nas linhas, previstas nas colunas)\n"
                     f"F1 macro {fmt(r['f1_macro_validacao'])} | acurácia {fmt(r['acuracia_validacao'])} | checkpoint na época {int(r['epoca_checkpoint'])}",
                     fontweight="bold", fontsize=10)
        rodape(fig)
        fig.tight_layout(rect=(0, 0.03, 1, 0.93))
        fig.savefig(SAIDA / "matrizes" / f"{n}_confusao.png", dpi=140)
        plt.close(fig)


def grafico_grade_matrizes(exp, res):
    for rodada in ("Inicial", "Optuna"):
        nomes = [nome(a, res, rodada) for a in ARQS if carregar_cm(nome(a, res, rodada)) is not None]
        if not nomes:
            continue
        fig, eixos = plt.subplots(2, 4, figsize=(20, 10.5))
        for ax in eixos.ravel():
            ax.axis("off")
        for ax, n in zip(eixos.ravel(), nomes):
            ax.axis("on")
            desenhar_cm(ax, carregar_cm(n), True, f"{n.split('_')[0]}  (F1 {fmt(exp.loc[n, 'f1_macro_validacao'])})", anotar=False, rotulos=False)
        fig.suptitle(f"Matrizes de confusão normalizadas na validação, {res} × {res}, hiperparâmetros {rodada.lower()} (reais nas linhas, previstas nas colunas)", fontweight="bold")
        rodape(fig)
        fig.tight_layout(rect=(0, 0.03, 1, 0.96))
        fig.savefig(SAIDA / f"resumo_5_matrizes_{rodada.lower()}_{res}.png", dpi=130)
        plt.close(fig)


def pares_confundidos(exp, res, topo=8):
    """Barras com os pares (real → prevista) mais confundidos do modelo de maior F1 macro de validação."""
    cand = [n for n in exp.index if exp.loc[n, "res"] == res and carregar_cm(n) is not None]
    if not cand:
        return None
    melhor = max(cand, key=lambda n: exp.loc[n, "f1_macro_validacao"])
    cm = carregar_cm(melhor)
    pares = [(CLASSES[i], CLASSES[j], int(cm[i, j]), cm[i, j] / cm[i].sum()) for i in range(10) for j in range(10) if i != j and cm[i, j] > 0]
    pares.sort(key=lambda t: -t[2])
    pares = pares[:topo]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    rot = [f"{r} → {p}" for r, p, _, _ in pares][::-1]
    ax.barh(rot, [c for _, _, c, _ in pares][::-1], color="#2a78d6")
    for y_, (_, _, c, f) in enumerate(pares[::-1]):
        ax.text(c + 0.1, y_, f"{c} ({fmt(100 * f, 1)}% da classe real)", va="center", fontsize=8)
    ax.set_xlabel("Imagens de validação confundidas")
    ax.set_title(f"Pares de classes mais confundidos — {melhor} (maior F1 macro de validação em {res} × {res})", fontweight="bold", fontsize=10)
    ax.set_xlim(0, max(c for _, _, c, _ in pares) * 1.35)
    rodape(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(SAIDA / f"resumo_6_pares_confundidos_{res}.png", dpi=150)
    plt.close(fig)
    return melhor, pares


def main():
    estilo()
    SAIDA.mkdir(parents=True, exist_ok=True)
    exp = carregar()
    for res in sorted(exp["res"].unique()):
        grafico_f1_antes_depois(exp, res)
        grafico_custo(exp, res)
        grafico_f1_classe(exp, res)
        grafico_curvas(exp, res)
        grafico_grade_matrizes(exp, res)
        r = pares_confundidos(exp, res)
        if r:
            print("modelo de maior F1 de validação:", r[0])
            for p in r[1]:
                print("  ", p)
    graficos_matrizes(exp)
    print("figuras em", SAIDA)


if __name__ == "__main__":
    main()
