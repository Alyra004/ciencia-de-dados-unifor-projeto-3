"""Diagramas das arquiteturas (MLP-A/B, ResNet18, VGG16 e os módulos de atenção SE e CBAM).

Os números (canais, tamanhos de mapa, parâmetros) vêm dos mesmos blocos de código de notebooks/02_mlp_preliminar.ipynb
e notebooks/03_CNNs.ipynb. Grava em resultados/figuras/arquiteturas/.

Uso (a partir da raiz do projeto): python src/graficos_arquiteturas.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

RAIZ = Path(__file__).resolve().parent.parent
SAIDA = RAIZ / "resultados" / "figuras" / "arquiteturas"
FONTE = "Fonte: elaborado pelos autores. Arquiteturas escritas do zero em PyTorch, sem pesos pré-treinados."

AZUL, LARANJA, VERDE, ROXO, CINZA, VERMELHO, AMARELO = "#2a78d6", "#eb6834", "#2e9d6a", "#7a5cc7", "#8a8f98", "#d64545", "#e8b23a"
CLARO = {AZUL: "#dbe9fb", LARANJA: "#fde4d8", VERDE: "#d8f0e3", ROXO: "#e6dff6", CINZA: "#ebedf0", VERMELHO: "#f8dcdc", AMARELO: "#faefc9"}


def caixa(ax, x, y, w, h, texto, cor=AZUL, fs=9, bold=False, tc="#1b2430"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=CLARO[cor], ec=cor, lw=1.6))
    ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold" if bold else "normal")


def seta(ax, p, q, cor="#444", rad=0.0, lw=1.5, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, color=cor, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def tela(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def salvar(fig, nome):
    SAIDA.mkdir(parents=True, exist_ok=True)
    fig.text(0.01, 0.005, FONTE, fontsize=7.5, color="#666")
    fig.savefig(SAIDA / nome, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ MLPs
def mlps():
    larguras = {"A": [49152, 512, 128, 10], "B": [49152, 512, 256, 128, 64, 10]}
    params = {"A": "25.234.570", "B": "25.341.386"}
    fig, eixos = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [4, 6]})
    for ax, (k, ws) in zip(eixos, larguras.items()):
        n = len(ws)
        ax.set_xlim(0, n * 2.1)
        ax.set_ylim(-1.6, 5.2)
        ax.axis("off")
        for i, w in enumerate(ws):
            altura = 0.7 + 3.0 * (min(w, 512) / 512) ** 0.6 if i else 4.2
            x = 0.3 + i * 2.1
            y = 2.0 - altura / 2
            cor = CINZA if i == 0 else (VERDE if i == n - 1 else AZUL)
            caixa(ax, x, y, 1.3, altura, "", cor)
            rotulo = "Entrada\n49.152" if i == 0 else (f"Saída\n{w}" if i == n - 1 else f"{w}")
            ax.text(x + 0.65, 2.0, rotulo, ha="center", va="center", fontsize=9.5, fontweight="bold")
            if i == 0:
                ax.text(x + 0.65, y - 0.45, "128×128×3\nem vetor", ha="center", va="top", fontsize=8, color="#555")
            elif i < n - 1:
                ax.text(x + 0.65, y - 0.45, "Linear\nBatchNorm\nReLU\nDropout", ha="center", va="top", fontsize=7.5, color="#555")
            else:
                ax.text(x + 0.65, y - 0.45, "Linear\n+ softmax\n(na perda)", ha="center", va="top", fontsize=7.5, color="#555")
            if i < n - 1:
                seta(ax, (x + 1.35, 2.0), (x + 2.05, 2.0))
        ax.set_title(f"MLP-{k}: {n - 2 if k == 'B' else 2} camadas escondidas, {params[k]} parâmetros", fontsize=11, fontweight="bold")
    fig.suptitle("MLPs (entrada 128 × 128 achatada; o tamanho das caixas é só ilustrativo)", fontsize=12.5, fontweight="bold", y=1.02)
    salvar(fig, "arq_1_mlps.png")


# ------------------------------------------------------------------------------------------------ ResNet18
def bloco_residual(ax, x0, y0, titulo, com_atalho_conv=False, atencao=None):
    """Desenha o bloco residual (duas conv 3x3, atalho e soma)."""
    w, h = 2.0, 0.55
    itens = [("Conv 3×3\nBatchNorm\nReLU", AZUL), ("Conv 3×3\nBatchNorm", AZUL)]
    if atencao:
        itens += [(atencao, ROXO)]
    y = y0
    topo = []
    for texto, cor in itens:
        caixa(ax, x0, y, w, h + 0.25, texto, cor, fs=8)
        topo.append(y)
        y += h + 0.25 + 0.4
    for a, b in zip(topo[:-1], topo[1:]):
        seta(ax, (x0 + w / 2, a + h + 0.25), (x0 + w / 2, b))
    # soma
    ys = y
    ax.add_patch(plt.Circle((x0 + w / 2, ys + 0.2), 0.22, fc="white", ec=VERDE, lw=1.8))
    ax.text(x0 + w / 2, ys + 0.2, "+", ha="center", va="center", fontsize=13, color=VERDE, fontweight="bold")
    seta(ax, (x0 + w / 2, topo[-1] + h + 0.25), (x0 + w / 2, ys - 0.02))
    # atalho
    xa = x0 + w + 0.55
    ax.plot([x0 + w / 2, x0 + w / 2], [y0 - 0.55, y0 - 0.35], color="#444", lw=1.5)
    seta(ax, (x0 + w / 2, y0 - 0.35), (x0 + w / 2, y0))
    ax.plot([x0 + w / 2, xa], [y0 - 0.45, y0 - 0.45], color=VERDE, lw=1.5)
    ax.plot([xa, xa], [y0 - 0.45, ys + 0.2], color=VERDE, lw=1.5)
    seta(ax, (xa, ys + 0.2), (x0 + w / 2 + 0.24, ys + 0.2), cor=VERDE)
    ax.text(xa + 0.08, (y0 + ys) / 2, "atalho\n" + ("(Conv 1×1 +\nBatchNorm)" if com_atalho_conv else "(identidade)"),
            fontsize=7.5, color=VERDE, va="center")
    ax.text(x0 + w / 2, ys + 0.75, "ReLU", ha="center", fontsize=8, color="#555")
    seta(ax, (x0 + w / 2, ys + 0.42), (x0 + w / 2, ys + 0.68))
    ax.text(x0 + w / 2, y0 - 0.85, titulo, ha="center", fontsize=9, fontweight="bold")


def resnet():
    fig = plt.figure(figsize=(15, 7.2))
    ax = fig.add_axes([0, 0, 0.66, 1])
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    etapas = [("Entrada", "3×128×128\n(3×256×256)", CINZA), ("Stem\nConv 7×7 /2\nBN + ReLU\nMaxPool /2", "64 canais\n32×32\n(64×64)", AZUL),
              ("Estágio 1\n2 blocos", "64 canais\n32×32\n(64×64)", AZUL), ("Estágio 2\n2 blocos\n(1º com /2)", "128 canais\n16×16\n(32×32)", AZUL),
              ("Estágio 3\n2 blocos\n(1º com /2)", "256 canais\n8×8\n(16×16)", AZUL), ("Estágio 4\n2 blocos\n(1º com /2)", "512 canais\n4×4\n(8×8)", AZUL),
              ("Média global\n(GAP)", "512", LARANJA), ("Linear\n512 → 10", "10 classes", VERDE)]
    x = 0.15
    for i, (t, s, c) in enumerate(etapas):
        caixa(ax, x, 3.5, 1.45, 1.9, t, c, fs=8.2, bold=(i in (1, 2, 3, 4, 5)))
        ax.text(x + 0.72, 3.3, s, ha="center", va="top", fontsize=7.8, color="#444")
        if i < len(etapas) - 1:
            seta(ax, (x + 1.47, 4.45), (x + 1.78, 4.45))
        x += 1.78
    ax.text(0.15, 6.55, "ResNet18 do zero: 11.181.642 parâmetros (11.270.722 com SE, 11.271.506 com CBAM, 11.360.586 com SE + CBAM)",
            fontsize=11, fontweight="bold")
    ax.text(0.15, 6.15, "Entre parênteses, o tamanho em 256 × 256. Os filtros, os blocos e a cabeça são os mesmos nas duas resoluções, e os parâmetros não mudam.",
            fontsize=9, color="#444")
    ax.text(0.15, 2.45, "Os 8 blocos residuais têm os canais (64, 64, 128, 128, 256, 256, 512, 512). O 1º bloco de cada estágio\n"
            "novo reduz o mapa pela metade (stride 2) e usa um atalho com Conv 1×1; os demais usam atalho identidade.",
            fontsize=9, color="#333", va="top")
    ax.text(0.15, 1.45, "Cabeça: média global dos mapas (agregação global, pedida no PDF) e uma camada linear.\n"
            "Sem camadas densas grandes, por isso o número de parâmetros é o mesmo em 128 e 256.", fontsize=9, color="#333", va="top")
    bx = fig.add_axes([0.66, 0, 0.34, 1])
    bx.set_xlim(0, 5.6)
    bx.set_ylim(0, 7.2)
    bx.axis("off")
    bloco_residual(bx, 1.0, 1.55, "Bloco residual (ResNet18)", com_atalho_conv=False)
    bx.text(2.8, 6.8, "Dentro de cada estágio", ha="center", fontsize=10, fontweight="bold")
    salvar(fig, "arq_2_resnet18.png")


# ------------------------------------------------------------------------------------------------ VGG16
def vgg():
    fig, ax = tela(15, 6.6)
    blocos = [("Bloco 1", "2 × Conv 3×3", "64", "64×64", "(128×128)"), ("Bloco 2", "2 × Conv 3×3", "128", "32×32", "(64×64)"),
              ("Bloco 3", "3 × Conv 3×3", "256", "16×16", "(32×32)"), ("Bloco 4", "3 × Conv 3×3", "512", "8×8", "(16×16)"),
              ("Bloco 5", "3 × Conv 3×3", "512", "4×4", "(8×8)")]
    ax.text(0.15, 6.2, "VGG16 com BatchNorm, do zero: 14.724.042 parâmetros (14.801.894 com SE, 14.802.384 com CBAM, 14.880.236 com SE + CBAM)",
            fontsize=11, fontweight="bold")
    ax.text(0.15, 5.8, "13 convoluções 3 × 3 (cada uma com BatchNorm e ReLU) em 5 blocos. As camadas densas da VGG original são trocadas por média global.",
            fontsize=9, color="#444")
    caixa(ax, 0.15, 2.7, 1.3, 1.9, "Entrada\n3×128×128\n(3×256×256)", CINZA, fs=8.2)
    seta(ax, (1.47, 3.65), (1.75, 3.65))
    x = 1.78
    alturas = [1.9, 1.9, 1.9, 1.9, 1.9]
    for i, (t, c, ch, s128, s256) in enumerate(blocos):
        caixa(ax, x, 2.7, 1.5, 1.9, f"{t}\n{c}\n{ch} canais\n+ MaxPool 2×2", AZUL, fs=8.2, bold=False)
        ax.text(x + 0.75, 2.5, f"após o pool:\n{s128} {s256}", ha="center", va="top", fontsize=7.8, color="#444")
        seta(ax, (x + 1.52, 3.65), (x + 1.82, 3.65))
        x += 1.82
    caixa(ax, x, 2.7, 1.3, 1.9, "Média global\n(GAP)\n512", LARANJA, fs=8.2)
    seta(ax, (x + 1.32, 3.65), (x + 1.62, 3.65))
    caixa(ax, x + 1.65, 2.7, 1.2, 1.9, "Linear\n512 → 10", VERDE, fs=8.2)
    ax.text(0.15, 1.2, "Cada bloco repete Conv 3×3 → BatchNorm → ReLU, depois um MaxPool 2×2 que divide o lado do mapa por 2. Os parâmetros são os mesmos em 128 e 256.\n"
            "Comparada à ResNet18, a VGG16 é uma pilha simples (sem atalhos), mais larga e com mais operações por imagem.", fontsize=9, color="#333", va="top")
    salvar(fig, "arq_3_vgg16.png")


# ------------------------------------------------------------------------------------------------ atenção
def atencao():
    fig = plt.figure(figsize=(15, 9.2))
    # SE
    ax = fig.add_axes([0, 0.52, 0.5, 0.46])
    ax.set_xlim(0, 7.5)
    ax.set_ylim(0, 4.8)
    ax.axis("off")
    ax.text(0.1, 4.45, "SE (Squeeze-and-Excitation)", fontsize=11, fontweight="bold")
    ax.text(0.1, 4.1, "pondera cada CANAL pela média espacial (Hu et al., 2018)", fontsize=8.8, color="#444")
    passos = [("Mapa\nC×H×W", CINZA), ("Média\nglobal\n→ C", ROXO), ("Linear\nC → C/16\nReLU", ROXO), ("Linear\nC/16 → C\nSigmoid", ROXO), ("× canal\na canal", VERDE)]
    x = 0.1
    for t, c in passos:
        caixa(ax, x, 1.7, 1.2, 1.5, t, c, fs=8.2)
        if x < 5:
            seta(ax, (x + 1.22, 2.45), (x + 1.42, 2.45))
        x += 1.45
    seta(ax, (0.7, 1.68), (0.7, 1.15), cor=CINZA)
    ax.plot([0.7, 6.55], [1.15, 1.15], color=CINZA, lw=1.5)
    seta(ax, (6.55, 1.15), (6.55, 1.68), cor=CINZA)
    ax.text(3.6, 0.85, "o mapa original segue direto para a multiplicação", ha="center", fontsize=8, color="#666")
    ax.text(0.1, 0.25, "Custo: 2 × C²/16 parâmetros por módulo (≈ 0,1 M a mais na rede inteira).", fontsize=8.8, color="#333")
    # CBAM
    ax = fig.add_axes([0.5, 0.52, 0.5, 0.46])
    ax.set_xlim(0, 7.5)
    ax.set_ylim(0, 4.8)
    ax.axis("off")
    ax.text(0.1, 4.45, "CBAM (Convolutional Block Attention Module)", fontsize=11, fontweight="bold")
    ax.text(0.1, 4.1, "atenção de CANAL e depois ESPACIAL (Woo et al., 2018)", fontsize=8.8, color="#444")
    caixa(ax, 0.1, 2.5, 1.1, 1.1, "Mapa\nC×H×W", CINZA, fs=8.2)
    caixa(ax, 1.55, 3.0, 1.45, 0.9, "Média e máx.\nglobais → C", ROXO, fs=8)
    caixa(ax, 1.55, 1.75, 1.45, 0.9, "MLP comum\nC → C/16 → C", ROXO, fs=8)
    caixa(ax, 3.35, 2.4, 1.1, 1.1, "soma +\nSigmoid\n× canal", VERDE, fs=8)
    caixa(ax, 4.8, 2.4, 1.3, 1.1, "média e máx.\nentre canais\n→ 2×H×W", ROXO, fs=7.8)
    caixa(ax, 6.2, 2.4, 1.25, 1.1, "Conv 7×7\nSigmoid\n× posição", VERDE, fs=7.8)
    seta(ax, (1.22, 3.1), (1.55, 3.4))
    seta(ax, (2.28, 3.0), (2.28, 2.67))
    seta(ax, (3.02, 2.2), (3.37, 2.7))
    seta(ax, (4.47, 2.95), (4.8, 2.95))
    seta(ax, (6.12, 2.95), (6.2, 2.95))
    ax.text(0.1, 1.2, "atenção de canal: \"o que olhar\"", fontsize=8.8, color=ROXO)
    ax.text(4.8, 1.75, "atenção espacial: \"onde olhar\"", fontsize=8.8, color=ROXO)
    ax.text(0.1, 0.25, "Custo: poucos parâmetros extras (≈ 0,1 M); mais operações por imagem.", fontsize=8.8, color="#333")
    # onde entra
    ax = fig.add_axes([0, 0.02, 1, 0.46])
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 4.4)
    ax.axis("off")
    ax.text(0.15, 4.1, "Onde os módulos entram nas redes", fontsize=11, fontweight="bold")
    # resnet
    ax.text(0.15, 3.6, "ResNet18: depois das duas convoluções do bloco, antes da soma com o atalho (8 módulos)", fontsize=9, color="#333")
    xs = 0.15
    for t, c in [("Conv 3×3\nBN + ReLU", AZUL), ("Conv 3×3\nBN", AZUL), ("SE\n(opcional)", ROXO), ("CBAM\n(opcional)", ROXO), ("+ atalho\nReLU", VERDE)]:
        caixa(ax, xs, 2.35, 1.3, 0.95, t, c, fs=8)
        if t != "+ atalho\nReLU":
            seta(ax, (xs + 1.32, 2.82), (xs + 1.55, 2.82))
        xs += 1.58
    ax.text(8.5, 3.6, "VGG16: depois das convoluções de cada bloco, antes do MaxPool (5 módulos)", fontsize=9, color="#333")
    xs = 8.5
    for t, c in [("Convs 3×3\nBN + ReLU", AZUL), ("SE\n(opcional)", ROXO), ("CBAM\n(opcional)", ROXO), ("MaxPool\n2×2", AZUL)]:
        caixa(ax, xs, 2.35, 1.35, 0.95, t, c, fs=8)
        if t != "MaxPool\n2×2":
            seta(ax, (xs + 1.37, 2.82), (xs + 1.62, 2.82))
        xs += 1.65
    tabela = [("sem atenção", "—", "11.181.642", "14.724.042"), ("SE", "pondera canais", "11.270.722", "14.801.894"),
              ("CBAM", "canal + espaço", "11.271.506", "14.802.384"), ("SE + CBAM", "os dois em sequência", "11.360.586", "14.880.236")]
    ax.text(0.15, 1.75, "Variante", fontsize=8.8, fontweight="bold")
    ax.text(2.5, 1.75, "O que faz", fontsize=8.8, fontweight="bold")
    ax.text(6.0, 1.75, "ResNet18 (parâmetros)", fontsize=8.8, fontweight="bold")
    ax.text(9.2, 1.75, "VGG16 (parâmetros)", fontsize=8.8, fontweight="bold")
    for i, (a, b, c, d) in enumerate(tabela):
        y = 1.4 - i * 0.33
        ax.text(0.15, y, a, fontsize=8.8)
        ax.text(2.5, y, b, fontsize=8.8)
        ax.text(6.0, y, c, fontsize=8.8)
        ax.text(9.2, y, d, fontsize=8.8)
    salvar(fig, "arq_4_atencao.png")


if __name__ == "__main__":
    mlps()
    resnet()
    vgg()
    atencao()
    print("figuras em", SAIDA)
