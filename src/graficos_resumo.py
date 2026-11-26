"""Gráficos de resumo do projeto inteiro (README e acompanhamento): dados, seis execuções finais, otimização, atenção, custo.

Só usa treino e validação (o teste não é usado). Lê resultados/experimentos.csv, resultados/optuna/, resultados/metricas_por_classe/
e dados/splits/particoes.csv, e grava em resultados/figuras/resumo/.

Uso (a partir da raiz do projeto): python src/graficos_resumo.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graficos_cnn import CLASSES, estilo, fmt, rodape  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
RES = RAIZ / "resultados"
SAIDA = RES / "figuras" / "resumo"
AZUL, LARANJA, VERDE, ROXO, CINZA = "#2a78d6", "#eb6834", "#2e9d6a", "#7a5cc7", "#8a8f98"

# As seis execuções finais escolhidas (receita comum: aumento leve; MLPs com os valores do Optuna, CNNs com os valores iniciais)
FINAIS = [("MLP-A-128", "MLP-A-128_aug_opt_preliminar"), ("MLP-B-128", "MLP-B-128_aug_opt_preliminar"),
          ("ResNet18-128", "ResNet18-128_preliminar"), ("VGG16-128", "VGG16-128_preliminar"),
          ("ResNet18-256", "ResNet18-256_preliminar"), ("VGG16-256", "VGG16-256_preliminar")]


def exp():
    d = pd.read_csv(RES / "experimentos.csv")
    return d.set_index("nome_execucao")


def salvar(fig, nome, dpi=150):
    SAIDA.mkdir(parents=True, exist_ok=True)
    fig.text(0.01, -0.045, "Fonte: elaborado pelos autores (NWPU-RESISC45, 10 classes). Dados de validação; o teste não foi usado.", fontsize=8, color="#666")
    fig.savefig(SAIDA / nome, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def rotular(ax, barras, casas=3, dy=0.004, fs=8.5):
    for b in barras:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy, fmt(b.get_height(), casas), ha="center", fontsize=fs)


def dados():
    p = pd.read_csv(RAIZ / "dados" / "splits" / "particoes.csv")
    t = p.groupby(["classe", "particao"]).size().unstack().reindex(list(CLASSES))[["treino", "validacao", "teste"]]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    base = np.zeros(len(t))
    for col, cor, nome in (("treino", AZUL, "Treino (4.899)"), ("validacao", LARANJA, "Validação (1.050)"), ("teste", VERDE, "Teste (1.051)")):
        ax.bar(t.index, t[col], bottom=base, color=cor, label=nome)
        for i, v in enumerate(t[col]):
            ax.text(i, base[i] + v / 2, str(v), ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
        base += t[col].values
    ax.set_ylabel("Imagens")
    ax.set_title("Partições estratificadas (70/15/15, semente 42): 7.000 imagens em 10 classes", fontweight="bold")
    ax.set_xticks(range(len(t)), t.index, rotation=25, ha="right")
    ax.legend(frameon=False, ncol=3, loc="upper center")
    ax.set_ylim(0, 800)
    salvar(fig, "resumo_1_particoes.png")


def seis_finais():
    e = exp()
    cores = [CINZA, CINZA, AZUL, LARANJA, AZUL, LARANJA]
    fig, eixos = plt.subplots(1, 2, figsize=(13, 4.8))
    nomes = [n for n, _ in FINAIS]
    f1 = [e.loc[r, "f1_macro_validacao"] for _, r in FINAIS]
    ac = [e.loc[r, "acuracia_validacao"] for _, r in FINAIS]
    b = eixos[0].bar(nomes, f1, color=cores, edgecolor="white")
    rotular(eixos[0], b, 4, 0.01)
    eixos[0].set_ylim(0, 1.08)
    eixos[0].set_ylabel("F1 macro na validação")
    eixos[0].set_title("F1 macro das seis execuções finais", fontweight="bold")
    eixos[0].tick_params(axis="x", rotation=25)
    tempo = [e.loc[r, "tempo_total_s"] / 60 for _, r in FINAIS]
    b2 = eixos[1].bar(nomes, tempo, color=cores, edgecolor="white")
    for bb, (_, r) in zip(b2, FINAIS):
        eixos[1].text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 0.2, f"{bb.get_height():.1f} min\n{int(e.loc[r, 'epocas_executadas'])} ép.", ha="center", fontsize=8.5)
    eixos[1].set_ylim(0, max(tempo) * 1.25)
    eixos[1].set_ylabel("Tempo total de treino (min)")
    eixos[1].set_title("Custo: tempo e épocas até a parada antecipada", fontweight="bold")
    eixos[1].tick_params(axis="x", rotation=25)
    fig.suptitle("As seis execuções finais (validação; aumento leve; um treino por modelo)", fontweight="bold", y=1.02)
    fig.tight_layout()
    salvar(fig, "resumo_2_seis_finais.png")


def com_sem_otimizacao():
    e = exp()
    itens = [("MLP-A\n128", "MLP-A-128_aug_preliminar", "MLP-A-128_aug_opt_preliminar"), ("MLP-B\n128", "MLP-B-128_aug_preliminar", "MLP-B-128_aug_opt_preliminar")]
    for arq in ("ResNet18", "VGG16"):
        for r in (128, 256):
            itens.append((f"{arq}\n{r}", f"{arq}-{r}_preliminar", f"{arq}-{r}_opt_preliminar"))
    x = np.arange(len(itens))
    fig, ax = plt.subplots(figsize=(12, 4.8))
    a = [e.loc[i[1], "f1_macro_validacao"] for i in itens]
    o = [e.loc[i[2], "f1_macro_validacao"] for i in itens]
    b1 = ax.bar(x - 0.2, a, 0.4, color=CINZA, label="Sem otimização (hiperparâmetros escolhidos por nós)")
    b2 = ax.bar(x + 0.2, o, 0.4, color=VERDE, label="Com otimização (Optuna)")
    rotular(ax, b1, 3, 0.01, 8)
    rotular(ax, b2, 3, 0.01, 8)
    for xi, va, vo in zip(x, a, o):
        ax.text(xi + 0.2, vo + 0.065, (f"{(vo - va) * 100:+.1f} pt" if abs(vo - va) >= 0.0005 else "0,0 pt").replace(".", ","), ha="center", fontsize=8.5, color=VERDE if vo >= va else "#c0392b", fontweight="bold")
    ax.set_xticks(x, [i[0] for i in itens])
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("F1 macro na validação")
    ax.legend(frameon=False, loc="upper left", ncol=2)
    ax.set_title("Com e sem otimização de hiperparâmetros (aumento leve, mesma receita nas duas barras)", fontweight="bold")
    salvar(fig, "resumo_3_com_sem_otimizacao.png")


def f1_por_classe():
    m = pd.DataFrame({n: pd.read_csv(RES / "metricas_por_classe" / f"{r}.csv", index_col=0).loc[list(CLASSES), "f1-score"] for n, r in FINAIS})
    fig, ax = plt.subplots(figsize=(10, 5.2))
    im = ax.imshow(m.values, cmap="YlGn", vmin=0.2, vmax=1.0, aspect="auto")
    ax.set_xticks(range(m.shape[1]), m.columns, rotation=25, ha="right")
    ax.set_yticks(range(m.shape[0]), m.index)
    ax.grid(False)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, fmt(m.values[i, j], 2), ha="center", va="center", fontsize=8.5, color="black" if m.values[i, j] < 0.8 else "white")
    fig.colorbar(im, ax=ax, label="F1 da classe")
    ax.set_title("F1 por classe nas seis execuções finais (validação)", fontweight="bold")
    salvar(fig, "resumo_4_f1_por_classe.png")


def atencao():
    e = exp()
    arqs = ["", "SE", "CBAM", "SECBAM"]
    nomes = ["sem atenção", "SE", "CBAM", "SE + CBAM"]
    cores = [CINZA, AZUL, LARANJA, ROXO]
    fig, eixos = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, fam in zip(eixos, ("ResNet18", "VGG16")):
        x = np.arange(2)
        for k, (a, n, c) in enumerate(zip(arqs, nomes, cores)):
            v = [e.loc[f"{fam}{a}-{r}_preliminar", "f1_macro_validacao"] for r in (128, 256)]
            b = ax.bar(x + (k - 1.5) * 0.2, v, 0.19, color=c, label=n)
            rotular(ax, b, 3, 0.003, 7.5)
        ax.set_xticks(x, ["128 × 128", "256 × 256"])
        ax.set_ylim(0.84, 0.98)
        ax.set_title(fam, fontweight="bold")
    eixos[0].set_ylabel("F1 macro na validação")
    eixos[0].legend(frameon=False, ncol=2, loc="upper left")
    fig.suptitle("Mecanismos de atenção (aumento leve, hiperparâmetros iniciais): efeito irregular", fontweight="bold", y=1.02)
    fig.tight_layout()
    salvar(fig, "resumo_5_atencao.png")

    # efeito médio da atenção sobre as 3 receitas e 2 rodadas e 2 resoluções (12 pares por variante e família)
    linhas = []
    for fam in ("ResNet18", "VGG16"):
        for a, n in zip(arqs[1:], nomes[1:]):
            ds = []
            for r in (128, 256):
                for suf in ("_preliminar", "_opt_preliminar", "_semaug_preliminar", "_semaug_opt_preliminar", "_aug2_preliminar", "_aug2_opt_preliminar"):
                    ds.append(e.loc[f"{fam}{a}-{r}{suf}", "f1_macro_validacao"] - e.loc[f"{fam}-{r}{suf}", "f1_macro_validacao"])
            ds = np.array(ds)
            linhas.append((fam, n, ds.mean(), (ds > 0).sum(), len(ds), ds.min(), ds.max()))
    t = pd.DataFrame(linhas, columns=["familia", "variante", "efeito_medio", "melhora_em", "de", "pior", "melhor"])
    t.to_csv(SAIDA / "atencao_efeito.csv", index=False)
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    x = np.arange(3)
    for k, fam in enumerate(("ResNet18", "VGG16")):
        sub = t[t.familia == fam]
        b = ax.bar(x + (k - 0.5) * 0.38, sub.efeito_medio * 100, 0.36, color=AZUL if fam == "ResNet18" else LARANJA, label=fam)
        for bb, (_, r) in zip(b, sub.iterrows()):
            ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + (0.15 if bb.get_height() >= 0 else -0.55),
                    f"{r.efeito_medio * 100:+.2f} pt\n({int(r.melhora_em)}/{int(r.de)} melhor)".replace(".", ","), ha="center", fontsize=8)
    ax.axhline(0, color="#333", lw=1)
    ax.set_xticks(x, ["SE", "CBAM", "SE + CBAM"])
    ax.set_ylabel("Diferença média de F1 vs. sem atenção (pontos)")
    ax.set_ylim(-4.2, 3.2)
    ax.legend(frameon=False)
    ax.set_title("Efeito médio da atenção em 12 pares por barra (3 receitas × 2 rodadas × 2 resoluções)", fontweight="bold", fontsize=10.5)
    salvar(fig, "resumo_6_atencao_efeito_medio.png")


def optuna():
    nomes = [("MLP-A-128", "MLP-A-128"), ("MLP-B-128", "MLP-B-128"), ("ResNet18-128", "ResNet18-128"), ("ResNet18-256", "ResNet18-256"),
             ("VGG16-128", "VGG16-128"), ("VGG16-256", "VGG16-256")]
    linhas = []
    for n, f in nomes:
        d = pd.read_csv(RES / "optuna" / f"tentativas_{f}.csv")
        linhas.append((n, (d.state == "COMPLETE").sum(), (d.state == "PRUNED").sum(), d.value[d.state == "COMPLETE"].max()))
    fig, eixos = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(len(linhas))
    eixos[0].bar(x, [l[1] for l in linhas], color=VERDE, label="concluídas")
    eixos[0].bar(x, [l[2] for l in linhas], bottom=[l[1] for l in linhas], color=CINZA, label="podadas (cortadas cedo)")
    for xi, l in zip(x, linhas):
        eixos[0].text(xi, l[1] + l[2] + 0.3, f"{l[1]} + {l[2]}", ha="center", fontsize=9)
    eixos[0].set_xticks(x, [l[0] for l in linhas], rotation=25, ha="right")
    eixos[0].set_ylabel("Tentativas do Optuna")
    eixos[0].set_title("Tentativas por busca (MLP: 20; CNN: 10)", fontweight="bold")
    eixos[0].legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.92))
    eixos[0].set_ylim(0, 24)
    b = eixos[1].bar(x, [l[3] for l in linhas], color=[CINZA, CINZA, AZUL, AZUL, LARANJA, LARANJA])
    rotular(eixos[1], b, 4, 0.01)
    eixos[1].set_xticks(x, [l[0] for l in linhas], rotation=25, ha="right")
    eixos[1].set_ylim(0, 1.1)
    eixos[1].set_ylabel("Melhor F1 macro de validação da busca")
    eixos[1].set_title("Melhor tentativa de cada busca", fontweight="bold")
    fig.suptitle("Busca de hiperparâmetros (Optuna): fase exploratória, separada das seis execuções finais", fontweight="bold", y=1.02)
    fig.tight_layout()
    salvar(fig, "resumo_7_optuna_busca.png")


def custo_resolucao():
    e = exp()
    fig, eixos = plt.subplots(1, 3, figsize=(14, 4.4))
    for fam, cor in (("ResNet18", AZUL), ("VGG16", LARANJA)):
        f = [e.loc[f"{fam}-{r}_preliminar", "f1_macro_validacao"] for r in (128, 256)]
        t = [e.loc[f"{fam}-{r}_preliminar", "tempo_total_s"] for r in (128, 256)]
        te = [e.loc[f"{fam}-{r}_preliminar", "tempo_medio_epoca_s"] for r in (128, 256)]
        eixos[0].plot([128, 256], f, "-o", color=cor, lw=2.2, label=fam)
        eixos[1].plot([128, 256], te, "-o", color=cor, lw=2.2, label=fam)
        eixos[2].plot([128, 256], t, "-o", color=cor, lw=2.2, label=fam)
        for r, a, b_, c in zip((128, 256), f, te, t):
            eixos[0].text(r, a + 0.0018, fmt(a, 3), ha="center", fontsize=9, color=cor)
            eixos[1].text(r, b_ + 0.9, f"{b_:.1f} s".replace(".", ","), ha="center", fontsize=9, color=cor)
            eixos[2].text(r, c + 55, f"{c:.0f} s", ha="center", fontsize=9, color=cor)
    for ax, t in zip(eixos, ("F1 macro (validação)", "Tempo médio por época", "Tempo total até a parada")):
        ax.set_xticks([128, 256], ["128 × 128", "256 × 256"])
        ax.set_title(t, fontweight="bold")
        ax.margins(x=0.25, y=0.18)
    eixos[0].legend(frameon=False)
    fig.suptitle("Efeito da resolução nas CNNs finais: ganho de F1 contra custo (mesma A100, mesmos hiperparâmetros)", fontweight="bold", y=1.03)
    fig.tight_layout()
    salvar(fig, "resumo_8_resolucao_custo.png")


def todos_modelos():
    """F1 de validação de todas as famílias (melhor de cada) e das seis execuções finais."""
    e = exp()
    itens = [("YOLO26n-cls\n(bônus)", 0.9895, VERDE), ("VGG16SE-256\n(melhor CNN,\nforte+Optuna)", e.loc["VGG16SE-256_aug2_opt_preliminar", "f1_macro_validacao"], LARANJA),
             ("VGG16-256\n(final)", e.loc["VGG16-256_preliminar", "f1_macro_validacao"], LARANJA), ("ResNet18-256\n(final)", e.loc["ResNet18-256_preliminar", "f1_macro_validacao"], AZUL),
             ("VGG16-128\n(final)", e.loc["VGG16-128_preliminar", "f1_macro_validacao"], LARANJA), ("ResNet18-128\n(final)", e.loc["ResNet18-128_preliminar", "f1_macro_validacao"], AZUL),
             ("MLP-B-128\n(final)", e.loc["MLP-B-128_aug_opt_preliminar", "f1_macro_validacao"], CINZA), ("MLP-A-128\n(final)", e.loc["MLP-A-128_aug_opt_preliminar", "f1_macro_validacao"], CINZA)]
    fig, ax = plt.subplots(figsize=(13, 4.8))
    b = ax.bar([i[0] for i in itens], [i[1] for i in itens], color=[i[2] for i in itens])
    rotular(ax, b, 4, 0.01)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("F1 macro na validação")
    ax.set_title("Panorama: das MLPs ao YOLO (validação)", fontweight="bold")
    ax.tick_params(axis="x", labelsize=8.2)
    salvar(fig, "resumo_9_panorama.png")


PASSOS_MLP = [
    ("0", "Ponto de partida: sem aumento, hiperparâmetros escolhidos por nós", "", False),
    ("1", "+ aumento leve (espelhar e girar)", "_aug", False),
    ("2", "Alternativa: aumento forte (zoom, cor, apagamento)", "_aug2", False),
    ("3", "Optuna, sem aumento", "", True),
    ("4", "Optuna + aumento leve  (ESCOLHIDA para as finais)", "_aug", True),
    ("5", "Optuna + aumento forte", "_aug2", True),
]


def progressao_mlp():
    """Tabela (CSV) e gráfico da progressão das configurações das MLPs, do ponto de partida às boas."""
    import json
    e = exp()
    linhas = []
    for passo, desc, rec, opt in PASSOS_MLP:
        for a in "AB":
            n = f"MLP-{a}-128{rec}{'_opt' if opt else ''}_preliminar"
            c = json.loads((RES / "configs" / f"{n}.json").read_text(encoding="utf-8"))
            ex = json.loads(c["resultado"]["extras"]) if isinstance(c["resultado"].get("extras"), str) else {}
            x = e.loc[n]
            base = e.loc[f"MLP-{a}-128_preliminar", "f1_macro_validacao"]
            linhas.append(dict(passo=passo, descricao=desc, mlp=f"MLP-{a}-128", execucao=n, aumento={"": "nenhum", "_aug": "leve", "_aug2": "forte"}[rec],
                               hiperparametros="Optuna" if opt else "iniciais", lr=float(x.taxa_aprendizagem), weight_decay=float(x.weight_decay),
                               dropout=ex.get("dropout", 0.5), lote=int(x.tamanho_lote), epocas=int(x.epocas_executadas), melhor_epoca=int(x.epoca_checkpoint),
                               f1_macro_validacao=float(x.f1_macro_validacao), acuracia_validacao=float(x.acuracia_validacao),
                               ganho_vs_passo0_pontos=round((x.f1_macro_validacao - base) * 100, 2)))
    d = pd.DataFrame(linhas)
    d.to_csv(RES / "progressao_mlp.csv", index=False)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    x = np.arange(len(PASSOS_MLP))
    for k, (a, cor) in enumerate((("A", AZUL), ("B", LARANJA))):
        v = d[d.mlp == f"MLP-{a}-128"].f1_macro_validacao.values
        b = ax.bar(x + (k - 0.5) * 0.38, v, 0.36, color=cor, label=f"MLP-{a}-128")
        rotular(ax, b, 3, 0.008, 8.5)
    curtos = ["0\nponto de partida\n(sem aumento)", "1\n+ aumento leve", "2\naumento forte\n(alternativa)", "3\nOptuna\n(sem aumento)", "4\nOptuna +\naumento leve", "5\nOptuna +\naumento forte"]
    ax.set_xticks(x, curtos, fontsize=8.5)
    ax.set_ylim(0.4, 0.78)
    ax.set_ylabel("F1 macro na validação")
    ax.axvspan(3.5, 4.5, color="#2e9d6a", alpha=0.12)
    ax.text(4, 0.755, "escolhida", ha="center", color="#2e9d6a", fontweight="bold", fontsize=9)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Progressão das MLPs: do ponto de partida às configurações boas (validação)", fontweight="bold")
    salvar(fig, "resumo_10_progressao_mlp.png")


if __name__ == "__main__":
    estilo()
    dados()
    seis_finais()
    com_sem_otimizacao()
    f1_por_classe()
    atencao()
    optuna()
    custo_resolucao()
    todos_modelos()
    progressao_mlp()
    print("figuras em", SAIDA)
