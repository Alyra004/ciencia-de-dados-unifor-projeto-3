"""Curvas de aprendizagem (perda e acurácia de treino/validação por época)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

COR_TREINO = "#2a78b5"
COR_VAL = "#d9762b"


def plotar_curvas(historico: pd.DataFrame, nome: str, destino: Path) -> Path:
    ep_ckpt = int(historico.loc[historico["checkpoint"], "epoca"].iloc[0])
    perto_do_fim = ep_ckpt > 0.6 * historico["epoca"].max()
    fig, eixos = plt.subplots(1, 3, figsize=(15, 4.2))

    paineis = [
        ("perda", "Perda (entropia cruzada)"),
        ("acuracia", "Acurácia"),
        ("f1_macro", "F1 macro"),
    ]
    for ax, (chave, titulo) in zip(eixos, paineis):
        ax.plot(historico["epoca"], historico[f"{chave}_treino"], color=COR_TREINO, lw=2, label="Treino")
        ax.plot(historico["epoca"], historico[f"{chave}_validacao"], color=COR_VAL, lw=2, label="Validação")
        ax.axvline(ep_ckpt, color="#555", ls="--", lw=1)
        ax.annotate(
            f"checkpoint (época {ep_ckpt})", xy=(ep_ckpt, ax.get_ylim()[1]),
            xytext=(-4, -12) if perto_do_fim else (4, -12), textcoords="offset points",
            fontsize=8, color="#555", ha="right" if perto_do_fim else "left",
        )
        ax.set_title(titulo)
        ax.set_xlabel("Época")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    eixos[0].legend(frameon=False)
    fig.suptitle(f"{nome} — curvas de aprendizagem", fontweight="bold")
    fig.text(0.01, 0.01, "Fonte: elaborado pelos autores (NWPU-RESISC45, 10 classes).", fontsize=8, color="#666")
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, dpi=150)
    plt.close(fig)
    return destino
