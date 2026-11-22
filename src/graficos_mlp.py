"""Gráficos das 12 MLPs (validação; o teste não é usado), a partir de resultados/ já gerado.

Reaproveita as funções de gráfico do notebook 02_mlp_preliminar.ipynb (mesmo estilo de antes) e acrescenta as
matrizes de confusão no padrão das CNNs. Grava em resultados/figuras/mlp/:
  <nome>_preliminar_curvas.png            curvas de perda, acurácia e F1 de cada um dos 12 treinos
  comparacao_1..5_*.png                   comparação dos 6 treinos com hiperparâmetros fixos (sem aumento, leve e forte)
  comparacao_6_optuna_antes_depois.png    hiperparâmetros fixos contra os do Optuna
  matrizes/<nome>_confusao.png            matriz de confusão de cada treino (contagens e normalizada)
  resumo_matrizes_<rodada>.png            grade com as 6 matrizes normalizadas de cada rodada

Uso (a partir da raiz do projeto): python src/graficos_mlp.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graficos_cnn import CLASSES, desenhar_cm, estilo, fmt, rodape  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
SAIDA = RAIZ / "resultados" / "figuras" / "mlp"
RECEITAS_OPT = [("", False, "sem aumento"), ("_aug", True, "aumento leve"), ("_aug2", "forte", "aumento forte")]
NOMES = [f"MLP-{a}-128{s}{o}_preliminar" for o in ("", "_opt") for a in "AB" for s in ("", "_aug", "_aug2")]


def celula(i):
    nb = json.loads((RAIZ / "notebooks" / "02_mlp_preliminar.ipynb").read_text(encoding="utf-8"))
    return "".join(nb["cells"][i]["source"])


def main():
    SAIDA.mkdir(parents=True, exist_ok=True)
    ns = {"RAIZ": RAIZ, "plt": plt, "pd": pd, "np": np, "RECEITAS_OPT": RECEITAS_OPT}
    cel10 = celula(10)
    exec(cel10[cel10.index("def plotar_curvas"):], ns)                      # só a função de curvas
    for nome in NOMES:                                                      # curvas dos 12 treinos
        hist = pd.read_csv(RAIZ / "resultados" / "historicos" / f"{nome}_historico.csv")
        ns["plotar_curvas"](hist, nome.replace("_preliminar", ""))
    c36 = celula(36)
    assert "tempo medido no Mac, com o chip Apple MPS" in c36
    exec(c36.replace("tempo medido no Mac, com o chip Apple MPS", "tempo medido na GPU A100 do Colab"), ns)   # comparacao_1..5
    ns["graficos_comparativos"](RAIZ, SAIDA)
    exec(celula(54), ns)                                                    # tabela antes x depois do Optuna
    exec(celula(55), ns)                                                    # comparacao_6

    # matrizes de confusão
    estilo()
    exp = pd.read_csv(RAIZ / "resultados" / "experimentos.csv").set_index("nome_execucao")
    (SAIDA / "matrizes").mkdir(exist_ok=True)

    def cm_de(nome):
        return pd.read_csv(RAIZ / "resultados" / "matrizes_confusao" / f"{nome}_cm.csv", index_col=0).values

    for nome in NOMES:
        cm, r = cm_de(nome), exp.loc[nome]
        fig, eixos = plt.subplots(1, 2, figsize=(13, 6.2))
        desenhar_cm(eixos[0], cm, False, "Contagens")
        desenhar_cm(eixos[1], cm, True, "Normalizada pela classe real (linhas somam 1)")
        fig.suptitle(f"{nome} — matriz de confusão na validação (reais nas linhas, previstas nas colunas)\n"
                     f"F1 macro {fmt(r['f1_macro_validacao'])} | acurácia {fmt(r['acuracia_validacao'])} | checkpoint na época {int(r['epoca_checkpoint'])}",
                     fontweight="bold", fontsize=10)
        rodape(fig)
        fig.tight_layout(rect=(0, 0.03, 1, 0.93))
        fig.savefig(SAIDA / "matrizes" / f"{nome}_confusao.png", dpi=140)
        plt.close(fig)
    for rodada, sufixo in (("inicial", ""), ("optuna", "_opt")):
        nomes = [f"MLP-{a}-128{s}{sufixo}_preliminar" for a in "AB" for s in ("", "_aug", "_aug2")]
        fig, eixos = plt.subplots(2, 3, figsize=(15, 10.5))
        for ax, n in zip(eixos.ravel(), nomes):
            desenhar_cm(ax, cm_de(n), True, f"{n.split('_preliminar')[0]}  (F1 {fmt(exp.loc[n, 'f1_macro_validacao'])})", anotar=False, rotulos=False)
        fig.suptitle(f"Matrizes de confusão normalizadas na validação, MLPs, hiperparâmetros {rodada} (reais nas linhas, previstas nas colunas)", fontweight="bold")
        rodape(fig)
        fig.tight_layout(rect=(0, 0.03, 1, 0.96))
        fig.savefig(SAIDA / f"resumo_matrizes_{rodada}.png", dpi=130)
        plt.close(fig)
    print("figuras em", SAIDA)


if __name__ == "__main__":
    main()
