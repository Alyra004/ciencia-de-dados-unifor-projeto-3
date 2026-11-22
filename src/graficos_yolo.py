"""Gráficos do bônus YOLO (validação; o teste não é usado).

Lê resultados/yolo/ (gerado por notebooks/03_CNNs.ipynb, seção 7, e src/avaliar_yolo_validacao.py) e a melhor CNN
em resultados/metricas_por_classe/, e grava em resultados/figuras/yolo/:
  yolo_1_matriz_confusao.png    contagens e normalização pela classe real
  yolo_2_curvas.png             perda e acurácia de validação por época, com a melhor época marcada
  yolo_3_f1_por_classe.png      F1 por classe: YOLO contra a melhor CNN de validação

Uso (a partir da raiz do projeto): python src/graficos_yolo.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graficos_cnn import CLASSES, FONTE, desenhar_cm, estilo, fmt, rodape  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
YOLO = RAIZ / "resultados" / "yolo"
SAIDA = RAIZ / "resultados" / "figuras" / "yolo"
CNN_REF = "VGG16SE-256_opt_preliminar"      # CNN de maior F1 macro de validação


def main():
    estilo()
    SAIDA.mkdir(parents=True, exist_ok=True)
    cm = pd.read_csv(YOLO / "yolo_cls_256_matriz_confusao.csv", index_col=0).values
    res = pd.read_csv(YOLO / "yolo_cls_256" / "results.csv")
    res.columns = [c.strip() for c in res.columns]
    resumo = pd.read_json(YOLO / "yolo_cls_256_resumo.json", typ="series")

    # 1. matriz de confusão
    fig, eixos = plt.subplots(1, 2, figsize=(13, 6.2))
    desenhar_cm(eixos[0], cm, False, "Contagens")
    desenhar_cm(eixos[1], cm, True, "Normalizada pela classe real (linhas somam 1)")
    fig.suptitle("YOLO26n-cls (256 × 256) — matriz de confusão na validação (reais nas linhas, previstas nas colunas)\n"
                 f"F1 macro {fmt(resumo['f1_macro_validacao'], 4)} | acurácia {fmt(resumo['acuracia_validacao'], 4)} | "
                 f"melhor época {int(resumo['epoca_melhor_acuracia'])}", fontweight="bold", fontsize=10)
    rodape(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(SAIDA / "yolo_1_matriz_confusao.png", dpi=140)
    plt.close(fig)

    # 2. curvas
    ep = int(resumo["epoca_melhor_acuracia"])
    fig, eixos = plt.subplots(1, 2, figsize=(12, 4.4))
    eixos[0].plot(res["epoch"], res["train/loss"], color="#2a78b5", lw=2, label="Treino")
    eixos[0].plot(res["epoch"], res["val/loss"], color="#d9762b", lw=2, label="Validação")
    eixos[0].set_title("Perda")
    eixos[1].plot(res["epoch"], res["metrics/accuracy_top1"], color="#d9762b", lw=2, label="Validação")
    eixos[1].scatter([ep], [res.loc[res["epoch"] == ep, "metrics/accuracy_top1"].iloc[0]], color="#d9762b", zorder=3, s=40)
    eixos[1].set_title("Acurácia (top-1)")
    for ax in eixos:
        ax.axvline(ep, color="#555", ls="--", lw=1)
        ax.set_xlabel("Época")
        ax.legend(frameon=False)
    fig.suptitle(f"YOLO26n-cls — curvas de aprendizagem (melhor época {ep}; parada antecipada na {int(res['epoch'].max())})",
                 fontweight="bold")
    rodape(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(SAIDA / "yolo_2_curvas.png", dpi=150)
    plt.close(fig)

    # 3. F1 por classe contra a melhor CNN
    f_yolo = pd.read_csv(YOLO / "yolo_cls_256_metricas_por_classe.csv", index_col=0).loc[list(CLASSES), "f1-score"]
    f_cnn = pd.read_csv(RAIZ / "resultados" / "metricas_por_classe" / f"{CNN_REF}.csv", index_col=0).loc[list(CLASSES), "f1-score"]
    x, larg = np.arange(len(CLASSES)), 0.38
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.bar(x - larg / 2, f_cnn.values, larg, color="#2a78d6", label=f"{CNN_REF.split('_')[0]} (do zero)")
    ax.bar(x + larg / 2, f_yolo.values, larg, color="#eb6834", label="YOLO26n-cls (pré-treinado)")
    for xi, v in zip(x - larg / 2, f_cnn.values):
        ax.text(xi, v + 0.003, fmt(v, 2), ha="center", fontsize=7.5)
    for xi, v in zip(x + larg / 2, f_yolo.values):
        ax.text(xi, v + 0.003, fmt(v, 2), ha="center", fontsize=7.5)
    ax.set_xticks(x, CLASSES, rotation=25, ha="right")
    ax.set_ylim(0.85, 1.02)
    ax.set_ylabel("F1 da classe (validação)")
    ax.set_title("F1 por classe: YOLO26n-cls contra a CNN de maior F1 macro de validação", fontweight="bold", fontsize=11)
    ax.legend(frameon=False, loc="lower right")
    rodape(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(SAIDA / "yolo_3_f1_por_classe.png", dpi=150)
    plt.close(fig)
    print("figuras em", SAIDA)


if __name__ == "__main__":
    main()
