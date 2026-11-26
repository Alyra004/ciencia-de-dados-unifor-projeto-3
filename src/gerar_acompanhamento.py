"""Gera docs/acompanhamento.pdf: documento (A4, vertical) para o acompanhamento, só com o que a entrega pede.

Conteúdo: dados filtrados e preparados, exploração, partições, modelos, resultados de validação de MLPs e CNNs, próximos passos e
dúvidas. Só usa treino e validação (o teste não é usado). Os números vêm de resultados/experimentos.csv; as figuras, de
resultados/figuras/. O HTML é impresso em PDF pelo Google Chrome (modo headless).

Uso (a partir da raiz do projeto): python src/gerar_acompanhamento.py
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
RES = RAIZ / "resultados"
SAIDA = RAIZ / "docs" / "acompanhamento.pdf"
e = pd.read_csv(RES / "experimentos.csv").set_index("nome_execucao")
C = ("airport", "beach", "circular_farmland", "dense_residential", "desert", "forest", "harbor", "mountain", "river", "storage_tank")
PT = {"airport": "Aeroporto", "beach": "Praia", "circular_farmland": "Áreas agrícolas circulares", "dense_residential": "Área residencial densa",
      "desert": "Deserto", "forest": "Floresta", "harbor": "Porto", "mountain": "Montanha", "river": "Rio", "storage_tank": "Tanques de armazenamento"}


def f(v, n=4):
    return f"{v:.{n}f}".replace(".", ",")


def mi(v):
    return f"{int(v):,}".replace(",", ".")


NF = [0]


def fig(rel, legenda, largura="100%", altura="none"):
    NF[0] += 1
    return (f'<figure><img src="{(RAIZ / rel).as_uri()}" style="width:{largura};max-height:{altura}">'
            f'<figcaption>Figura {NF[0]}. {legenda}</figcaption></figure>')


def tabela(cab, linhas, classe="", destaque=None):
    h = f'<table class="{classe}"><thead><tr>' + "".join(f"<th>{c}</th>" for c in cab) + "</tr></thead><tbody>"
    for i, l in enumerate(linhas):
        h += f'<tr class="{"dest" if destaque is not None and i in destaque else ""}">' + "".join(f"<td>{c}</td>" for c in l) + "</tr>"
    return h + "</tbody></table>"


def kpi(v, t, cor="az"):
    return f'<div class="kpi {cor}"><b>{v}</b><span>{t}</span></div>'


def fx(itens, extra=""):
    h = f'<div class="fx {extra}">'
    for i, (t, d, cor) in enumerate(itens):
        h += f'<div class="b {cor}"><b>{t}</b><span>{d}</span></div>'
        if i < len(itens) - 1:
            h += "<i>→</i>"
    return h + "</div>"


def dif(a, b):
    d = e.loc[b, "f1_macro_validacao"] - e.loc[a, "f1_macro_validacao"]
    return (f"{d * 100:+.1f}".replace(".", ",") if abs(d) >= 0.0005 else "0,0") + " pt"


def tab_curvas(runs):
    """Acurácia e perda de treino e validação no checkpoint de cada execução."""
    lin = []
    for nm, r in runs:
        h = pd.read_csv(RES / "historicos" / f"{r}_historico.csv")
        x = h[h.epoca == int(e.loc[r, "epoca_checkpoint"])].iloc[0]
        lac = x.acuracia_treino - x.acuracia_validacao
        lin.append((f"<b>{nm}</b>", f(x.acuracia_treino, 3), f(x.acuracia_validacao, 3), f"{lac * 100:+.1f}".replace(".", ",") + " pt", f(x.perda_treino, 3), f(x.perda_validacao, 3)))
    return tabela(["Execução (no checkpoint)", "Acurácia treino", "Acurácia validação", "Treino − validação", "Perda treino", "Perda validação"], lin, "t mini")


def tab_classe(nm, r):
    """Precisão, recall, F1 e suporte por classe de uma execução (validação)."""
    m = pd.read_csv(RES / "metricas_por_classe" / f"{r}.csv", index_col=0)
    lin = [(f"<code>{c}</code>", f(m.loc[c, "precision"], 3), f(m.loc[c, "recall"], 3), f(m.loc[c, "f1-score"], 3), int(m.loc[c, "support"])) for c in C]
    lin.append(("<b>macro</b>", f(m.loc["macro avg", "precision"], 3), f(m.loc["macro avg", "recall"], 3), f(m.loc["macro avg", "f1-score"], 3), int(m.loc["macro avg", "support"])))
    return f'<div class="leg">{nm}</div>' + tabela(["Classe", "Precisão", "Recall", "F1", "Suporte"], lin, "t mini", destaque={10})


def sci(v):
    return f"{v:.1e}".replace(".", ",") if v < 1e-3 else f"{v:.4f}".rstrip("0").replace(".", ",")


def tab_progressao():
    """Progressão das configurações das MLPs (resultados/progressao_mlp.csv, gerado por src/graficos_resumo.py)."""
    d = pd.read_csv(RES / "progressao_mlp.csv")
    lin = []
    for r in d.itertuples():
        a = r.mlp.endswith("A-128")
        lin.append((r.passo, r.descricao if a else "", r.mlp.replace("-128", ""), r.aumento, "Optuna" if r.hiperparametros == "Optuna" else "iniciais", sci(r.lr), sci(r.weight_decay),
                    f(r.dropout, 1), r.lote, f"{r.epocas} ({r.melhor_epoca})", f"<b>{f(r.f1_macro_validacao)}</b>", "—" if str(r.passo) == "0" else f"{r.ganho_vs_passo0_pontos:+.1f}".replace(".", ",")))
    return tabela(["Passo", "O que mudou", "MLP", "Aumento", "Hiperpar.", "lr", "Weight decay", "Dropout", "Lote", "Ép. (melhor)", "F1 macro", "Δ vs. 0"], lin, "t mini prog", destaque={8, 9})


def tab_top_optuna():
    lin = []
    for a in "AB":
        t = pd.read_csv(RES / "optuna" / f"tentativas_MLP-{a}-128.csv")
        t = t[t.state == "COMPLETE"].sort_values("value", ascending=False).head(3)
        for i, x in enumerate(t.itertuples(), 1):
            lin.append((f"MLP-{a}", i, int(x.number), sci(x.params_lr), sci(x.params_weight_decay), f(x.params_dropout, 1), int(x.params_lote), f(x.value)))
    return tabela(["Busca", "Posição", "Tentativa", "lr", "Weight decay", "Dropout", "Lote", "F1 (até 60 épocas)"], lin, "t mini")


FINAIS = [("MLP-A-128", "MLP-A-128_aug_opt_preliminar"), ("MLP-B-128", "MLP-B-128_aug_opt_preliminar"), ("ResNet18-128", "ResNet18-128_preliminar"),
          ("VGG16-128", "VGG16-128_preliminar"), ("ResNet18-256", "ResNet18-256_preliminar"), ("VGG16-256", "VGG16-256_preliminar")]
part = pd.read_csv(RAIZ / "dados" / "splits" / "particoes.csv")
tot = part.particao.value_counts()
mlpA, mlpB = e.loc["MLP-A-128_aug_opt_preliminar"], e.loc["MLP-B-128_aug_opt_preliminar"]
N_CNN = len(e[e.index.str.match(r"^(ResNet|VGG)")])

P = []  # blocos do documento

# ---------------------------------------------------------------- cabeçalho
P.append("""<header class="topo"><div class="kick">Projeto aplicado · Unidade III · Ciência de Dados</div>
<h1>Classificação de Cenas Aéreas com Redes Neurais</h1>
<p class="sub">Documento de acompanhamento: dados, exploração, partições e resultados de validação</p>
<p class="eq"><b>Equipe:</b> João Igor Vidal de Andrade, Raquel Albuquerque Quirino, Amanda Lira e Sofya Oliveira &nbsp;·&nbsp; <b>Repositório:</b> https://github.com/Alyra004/ciencia-de-dados-unifor-projeto-3</p></header>
<div class="aviso">Todos os resultados são de <b>treino e validação</b>. O <b>conjunto de teste não foi utilizado</b> em nenhum treino, escolha ou gráfico.</div>""")

# ---------------------------------------------------------------- 1 resumo
P.append(f"""<h2>1. Resumo do projeto</h2>
<p>Investigamos como a <b>arquitetura</b> (MLP × CNN) e a <b>resolução</b> das imagens (128 × 128 × 256 × 256) afetam a classificação de cenas aéreas do
<b>NWPU-RESISC45</b> (10 classes) e <b>quais ganhos justificam o custo computacional</b>. Todos os modelos são escritos do zero em PyTorch, sem pré-treinamento, e treinados
na <b>mesma GPU A100</b> (Google Colab).</p>
<div class="kpis">{kpi("7.000", "imagens RGB, 10 classes", "az")}{kpi(f"{mi(tot['treino'])} / {mi(tot['validacao'])} / {mi(tot['teste'])}", "treino / validação / teste", "cz")}
{kpi("0,700", "melhor MLP (F1 macro, validação)", "lj")}{kpi("0,957", "melhor CNN final (VGG16-256)", "vd")}</div>
<p><b>Os seis treinamentos obrigatórios</b> (todos do zero) foram definidos assim, olhando só a validação:</p>
{tabela(["#", "Execução", "Receita"], [(1, "MLP-A-128", "aumento leve + hiperparâmetros do Optuna"), (2, "MLP-B-128", "aumento leve + hiperparâmetros do Optuna"),
(3, "ResNet18-128 (CNN A)", "aumento leve + hiperparâmetros iniciais"), (4, "VGG16-128 (CNN B)", "aumento leve + hiperparâmetros iniciais"),
(5, "ResNet18-256 (CNN A)", "mesma receita da 3; muda só a resolução"), (6, "VGG16-256 (CNN B)", "mesma receita da 4; muda só a resolução")], "t")}
<p class="peq">Por que a mesma receita nas duas resoluções: o PDF exige manter lr, otimizador, semente e parada ao passar de 128 para 256, e os valores do Optuna das CNNs
mudam entre as resoluções. Com os valores iniciais, a resolução melhora as duas CNNs de forma clara.</p>
<h3>1.1 Equipe e divisão de tarefas</h3>
{tabela(["Integrante", "Responsabilidades no projeto"], [("João Igor Vidal de Andrade", "CNNs: ResNet18 e VGG16 (e variantes com atenção), treinos em 128 e 256, busca de hiperparâmetros das CNNs e comparação de resolução e custo."),
("Amanda Lira", "Análise exploratória dos dados e limpeza dos dados (integridade, duplicatas, ambiguidades visuais e comparação de resoluções)."),
("Raquel Albuquerque Quirino", "Análise exploratória dos dados; criação dos documentos e do artigo; organização das tarefas do projeto."),
("Sofya Oliveira", "MLPs: MLP-A e MLP-B, treinos em 128 e busca de hiperparâmetros das MLPs.")], "t")}
<p class="peq">Código, partições, configurações, históricos, métricas e figuras estão no repositório do GitHub (link no cabeçalho).</p>""")

# ---------------------------------------------------------------- 2 dados
lin = [(i + 1, f"<code>{c}</code>", PT[c], f"{int(((part.classe == c) & (part.particao == 'treino')).sum())} / "
        f"{int(((part.classe == c) & (part.particao == 'validacao')).sum())} / {int(((part.classe == c) & (part.particao == 'teste')).sum())}") for i, c in enumerate(C)]
P.append(f"""<h2>2. Dados filtrados</h2>
<p>Do dataset completo (45 classes, 700 imagens por classe) mantivemos <b>exclusivamente as 10 pastas</b> pedidas, o que dá <b>7.000 imagens RGB de 256 × 256</b>.
A ordem das classes é <b>fixa</b> em todos os modelos, métricas e matrizes.</p>
{fig("resultados/figuras/eda/eda_1_exemplos_classes.png", "Uma imagem de cada uma das 10 classes (amostra aleatória, semente 42).", "100%", "230px")}
{tabela(["#", "Pasta original", "Classe", "Treino / validação / teste"], lin, "t mini")}""")

# ---------------------------------------------------------------- 3 exploração
P.append(f"""<h2>3. Análise exploratória</h2>
<h3>3.1 Ambiguidades visuais e resolução</h3>
<p>Antes de treinar, levantamos pares de classes que podem se confundir (<b>praia × rio</b> e <b>deserto × montanha</b>), como hipótese a conferir depois nas matrizes de confusão.
Também comparamos a mesma cena em 128 × 128 e 256 × 256: ao reduzir, perdem-se detalhes finos (pistas, tanques, barcos).</p>
<div class="duas">{fig("resultados/figuras/eda/eda_2_ambiguidades.png", "Pares de classes com possível confusão.", "100%", "300px")}
{fig("resultados/figuras/eda/eda_3_resolucao_128_vs_256.png", "A mesma cena em 128 × 128 e 256 × 256.", "100%", "300px")}</div>
<h3>3.2 Integridade e duplicatas</h3>
{tabela(["Verificação", "Resultado"], [("Imagens lidas", "7.000, todas legíveis"), ("Tamanho e canais", "todas 256 × 256, RGB (3 canais)"), ("Duplicatas exatas (MD5)", "<b>6 pares</b>, todos em <code>airport</code>; nenhuma foi excluída"),
("Originais", "não foram alterados"), ("Quase-duplicatas (hash perceptual)", "<b>5 pares entre partições</b> (4 em <code>beach</code>, 1 em <code>mountain</code>); limitação registrada")], "t")}
<div class="duas">{fig("resultados/figuras/eda/eda_4_duplicatas_proporcao.png", "Proporção de arquivos repetidos.", "78%", "250px")}
{fig("resultados/figuras/eda/eda_5_duplicatas_por_classe.png", "Arquivos repetidos por classe.", "100%", "250px")}</div>""")

# ---------------------------------------------------------------- 4 partições
P.append(f"""<h2>4. Partições e prevenção de vazamento</h2>
<p>Divisão <b>70 / 15 / 15</b>, estratificada, com semente <b>42</b>, salva em <code>dados/splits/particoes.csv</code> (arquivo, classe e partição, com caminhos relativos) e <b>reutilizada
em todos os experimentos</b>. As cópias idênticas ficam na mesma partição (o sorteio é feito sobre os hashes MD5), por isso <code>airport</code> tem 489 / 105 / 106 e as demais 490 / 105 / 105.</p>
{fig("resultados/figuras/resumo/resumo_1_particoes.png", "Imagens por classe e partição (4.899 treino, 1.050 validação, 1.051 teste).", "88%", "290px")}
{tabela(["Regra de proteção contra vazamento", "Como foi feito"], [("Ordem dos caminhos", "ordenados antes do sorteio"), ("Cópias idênticas", "agrupadas por hash na mesma partição"),
("Mesma divisão em tudo", "MLPs, CNNs e bônus usam o mesmo arquivo de partições"), ("Normalização", "média e desvio calculados <b>só no treino</b> e reaplicados à validação e ao teste"),
("Aumento de dados", "somente no treino"), ("Escolhas (hiperparâmetros, checkpoint, parada)", "feitas com a <b>validação</b>; o teste não é usado")], "t")}""")

# ---------------------------------------------------------------- 5 preparo
P.append(f"""<h2>5. Preparação das imagens</h2>
{fx([("Imagem original", "256 × 256, RGB", "cz"), ("Redimensiona", "a partir do original: 128 ou 256", "az"), ("÷ 255 e normaliza", "(x − média) ÷ desvio por canal, do treino", "rx"),
("Aumento de dados", "só no treino", "lj"), ("Entrada", "MLP: vetor · CNN: tensor 3 × R × R", "vd")])}
<div class="duas">{tabela(["Canal", "Média (treino)", "Desvio (treino)"], [("R", "0,4009", "0,2235"), ("G", "0,4073", "0,1860"), ("B", "0,3541", "0,1724")], "t mini")}
{tabela(["Modelo", "Entrada", "Números por imagem"], [("MLPs", "vetor (128 × 128)", "49.152"), ("CNNs em 128", "tensor 3 × 128 × 128", "49.152"), ("CNNs em 256", "tensor 3 × 256 × 256", "196.608")], "t mini")}</div>
{tabela(["Receita de aumento (só no treino)", "O que faz", "Uso"], [("Sem aumento", "nada", "exploratório"), ("<b>Leve</b>", "espelhamento horizontal e vertical e rotação de 0°, 90°, 180° ou 270°", "<b>execuções finais</b>"),
("Forte", "leve + recorte com zoom (50–100% da área), brilho/contraste/saturação ±20% e apagamento aleatório", "exploratório")], "t")}""")

# ---------------------------------------------------------------- 6 modelos
P.append(f"""<h2>6. Arquiteturas</h2>
<p>Duas famílias, cada uma com <b>duas arquiteturas que diferem na organização</b>: MLP-A × MLP-B (profundidade) e ResNet18 × VGG16 (blocos residuais × pilha de convoluções). Nas CNNs usamos
<b>agregação global</b> (média dos mapas) antes da camada de saída, então os parâmetros são <b>os mesmos em 128 e 256</b>.</p>
<h3>6.1 MLP-A × MLP-B</h3>
{fx([("Entrada", "49.152", "cz"), ("512", "Linear·BN·ReLU·Dropout", "az"), ("128", "Linear·BN·ReLU·Dropout", "az"), ("Saída", "10 classes", "vd")])}
<div class="leg">MLP-A: 2 camadas escondidas · {mi(mlpA.parametros)} parâmetros</div>
{fx([("Entrada", "49.152", "cz"), ("512", "Linear·BN·ReLU·Dropout", "az"), ("256", "Linear·BN·ReLU·Dropout", "az"), ("128", "Linear·BN·ReLU·Dropout", "az"), ("64", "Linear·BN·ReLU·Dropout", "az"), ("Saída", "10 classes", "vd")])}
<div class="leg">MLP-B: 4 camadas escondidas · {mi(mlpB.parametros)} parâmetros</div>
<p class="peq">Hipótese: mais profundidade combina melhor os pixels, mas aumenta o risco de decorar o treino. A 1ª camada guarda quase todos os parâmetros (49.152 × 512 ≈ 25,2 M).</p>
<h3>6.2 ResNet18 (CNN A) e VGG16 (CNN B)</h3>
<div class="leg">ResNet18 · 11.181.642 parâmetros · mapas em 128 × 128 (entre parênteses, 256 × 256)</div>
{fx([("Entrada", "3×128×128", "cz"), ("Stem", "Conv 7×7 /2 + MaxPool · 64 · 32×32 (64×64)", "az"), ("Est. 1", "2 blocos · 64 · 32×32 (64×64)", "az"), ("Est. 2", "2 blocos · 128 · 16×16 (32×32)", "az"),
("Est. 3", "2 blocos · 256 · 8×8 (16×16)", "az"), ("Est. 4", "2 blocos · 512 · 4×4 (8×8)", "az"), ("Média global", "512", "lj"), ("Linear", "512→10", "vd")], "peq")}
<div class="leg">Bloco residual: o atalho soma a entrada x à saída das duas convoluções (identidade, ou Conv 1×1 quando o tamanho muda)</div>
{fx([("x", "entrada", "cz"), ("Conv 3×3", "BatchNorm · ReLU", "az"), ("Conv 3×3", "BatchNorm", "az"), ("+ x", "soma com o atalho", "vd"), ("ReLU", "saída", "cz")], "peq")}
<div class="leg">VGG16 com BatchNorm · 14.724.042 parâmetros · 13 convoluções 3×3 (BatchNorm + ReLU) em 5 blocos, cada um terminando em MaxPool 2×2</div>
{fx([("Entrada", "3×128×128", "cz"), ("Bloco 1", "2×Conv · 64 → 64×64 (128×128)", "az"), ("Bloco 2", "2×Conv · 128 → 32×32 (64×64)", "az"), ("Bloco 3", "3×Conv · 256 → 16×16 (32×32)", "az"),
("Bloco 4", "3×Conv · 512 → 8×8 (16×16)", "az"), ("Bloco 5", "3×Conv · 512 → 4×4 (8×8)", "az"), ("Média global", "512", "lj"), ("Linear", "512→10", "vd")], "peq")}
{tabela(["", "ResNet18 (CNN A)", "VGG16 (CNN B)"], [("Organização", "8 blocos residuais com atalho", "pilha de 13 convoluções em 5 blocos"), ("Parâmetros / modelo salvo", "11.181.642 / 42,7 MB", "14.724.042 / 56,2 MB"),
("Hipótese", "atalhos estabilizam o treino com menos parâmetros e operações", "pilha mais larga extrai mais detalhe das texturas, a custo maior")], "t")}
<h3>6.3 Extra: mecanismos de atenção</h3>
<p>Testamos, como experimento extra, <b>SE</b> (pondera cada canal pela média espacial) e <b>CBAM</b> (atenção de canal e depois espacial), isolados e juntos, nas duas famílias
(+0,5 a +1,6% de parâmetros). As execuções finais ficam <b>sem atenção</b>; os resultados estão na seção 8.6.</p>""")

# ---------------------------------------------------------------- 7 protocolo
P.append(f"""<h2>7. Protocolo de treino e métricas</h2>
<div class="duas">{tabela(["Item", "Valor"], [("Framework / hardware", "PyTorch 2.14 · GPU A100 40 GB (Colab)"), ("Pesos iniciais", "novos (sem pré-treino)"), ("Perda / otimizador", "CrossEntropy · Adam"),
("lr / weight decay / lote", "1e-4 / 1e-4 / 64 (CNNs e treinos iniciais)"), ("<b>Limite de épocas</b>", "<b>200</b>"), ("<b>Parada antecipada</b>", "10 épocas sem melhora do F1 macro de validação"),
("Checkpoint", "época de maior F1 macro de validação"), ("Semente", "42 (determinístico)")], "t mini")}
{tabela(["Métrica", "Para quê"], [("<b>F1 macro</b>", "principal: escolhe checkpoint, parada e Optuna"), ("Acurácia", "secundária (classes balanceadas)"), ("Precisão, recall e F1 por classe", "diagnóstico por classe"),
("Matriz de confusão", "quais classes se confundem"), ("Curvas de perda e acurácia", "subajuste e sobreajuste"), ("Custo", "parâmetros, MB, épocas, tempo total e por época")], "t mini")}</div>
{tabela(["Etapa", "Limite de épocas", "Para quê"], [("Treino geral (finais e exploratórios)", "<b>200</b> (parada com 10 épocas sem melhora)", "produzir o modelo de cada configuração"),
("Busca Optuna das MLPs (20 tentativas por MLP)", "<b>60</b> por tentativa", "achar lr, weight decay, dropout e lote"), ("Busca Optuna das CNNs (10 tentativas por busca)", "<b>30</b> por tentativa", "idem, com orçamento menor")], "t")}
<p class="peq">O tempo é medido dentro do laço de treino e validação de cada época, sem o carregamento dos dados, sempre na mesma A100.</p>""")

# ---------------------------------------------------------------- 8 resultados
lin = []
for r_, t in (("sem aumento", ""), ("leve", "_aug"), ("forte", "_aug2")):
    for a in ("A", "B"):
        n0, n1 = f"MLP-{a}-128{t}_preliminar", f"MLP-{a}-128{t}_opt_preliminar"
        lin.append((f"MLP-{a}-128", r_, f(e.loc[n0, "f1_macro_validacao"]), f(e.loc[n0, "acuracia_validacao"]), int(e.loc[n0, "epocas_executadas"]),
                    f(e.loc[n1, "f1_macro_validacao"]), f(e.loc[n1, "acuracia_validacao"]), int(e.loc[n1, "epocas_executadas"])))
P.append(f"""<h2>8. Resultados de validação</h2>
<p class="peq">Cada configuração foi treinada <b>uma vez</b>; diferenças de poucos décimos de ponto não provam superioridade.</p>
<h3>8.1 MLPs</h3>
<p><b>O aumento de dados ajuda muito as MLPs e o Optuna ajuda mais ainda.</b> A melhor MLP é a <b>MLP-B com aumento leve e Optuna (F1 macro 0,700)</b>.
A acurácia de treino é de só 0,74 a 0,80: a MLP é limitada pela falta de noção espacial.</p>
{tabela(["MLP", "Aumento", "F1 inicial", "Acurácia", "Épocas", "F1 Optuna", "Acurácia", "Épocas"], lin, "t mini", destaque={3})}
{fig("resultados/figuras/mlp/comparacao_1_f1_validacao.png", "F1 macro de validação das MLPs por receita de aumento (hiperparâmetros iniciais).", "82%", "280px")}
<h4>Progressão das MLPs até as configurações boas</h4>
<p>Partimos de uma configuração simples e mudamos <b>uma coisa de cada vez</b>, olhando o F1 macro de validação. Tudo está em <code>resultados/progressao_mlp.csv</code>.</p>
{fig("resultados/figuras/resumo/resumo_10_progressao_mlp.png", "Progressão do F1 macro de validação das MLPs, do ponto de partida às configurações escolhidas.", "100%", "250px")}
{tab_progressao()}
{tabela(["Passo", "O que aprendemos"], [("0 → 1", "<b>O aumento de dados foi o maior salto</b> (+10,1 pontos na MLP-A, +7,2 na MLP-B). Sem aumento, a MLP decora o treino e para cedo (melhor época 17 e 31)."),
("1 → 2", "Aumento <b>forte</b> com os hiperparâmetros iniciais <b>piorou</b> em relação ao leve; a busca do Optuna veio para tentar consertar isso."),
("Optuna", "20 tentativas por MLP, com aumento leve: preferiu <b>menos dropout</b> (0,3 e 0,2 em vez de 0,5), <b>muito menos weight decay</b> (7e-6 e 2e-6) e <b>lote maior</b> (128). Mais pesaram o weight decay e o lr."),
("3", "Optuna <b>sem</b> aumento ajuda pouco (+2,0 e +1,4 pontos)."),
("4", "<b>Optuna + aumento leve: combinação escolhida</b> (0,682 e 0,700). Com ela a MLP-B passa a MLP-A."),
("5", "Optuna + aumento forte: praticamente empatado com o leve (0,699 e 0,696); escolhemos o leve por ser a receita comum às CNNs.")], "t mini")}
<p class="peq">Melhores tentativas de cada busca (todas as 40 tentativas estão em <code>resultados/optuna/tentativas_MLP-*-128.csv</code>; a busca treina até 60 épocas por tentativa, por isso o F1 da busca fica abaixo do treino completo):</p>
{tab_top_optuna()}
<h3>8.2 CNNs e as seis execuções finais</h3>
{fig("resultados/figuras/resumo/resumo_2_seis_finais.png", "F1 macro e custo (tempo e épocas) das seis execuções finais.", "100%", "250px")}""")
lin = []
for nm, r in FINAIS:
    x = e.loc[r]
    lin.append((f"<b>{nm}</b>", f(x.f1_macro_validacao), f(x.acuracia_validacao), f(x.perda_validacao), int(x.epocas_executadas), int(x.epoca_checkpoint), mi(x.parametros), f"{x.tamanho_modelo_mb:.1f}".replace(".", ","),
                f"{x.tempo_total_s:.0f}", f(x.tempo_medio_epoca_s, 2)))
P.append(f"""{tabela(["Execução", "F1 macro", "Acurácia", "Perda val.", "Épocas", "Melhor época", "Parâmetros", "MB", "Tempo (s)", "s/época"], lin, "t mini", destaque={4, 5})}
<p class="peq">Números dos treinos preliminares feitos com <b>a mesma receita</b> das execuções finais (semente 42, mesma A100; reproduzem os mesmos resultados). O retreino final e a avaliação no teste vêm depois.</p>
<h3>8.3 Curvas de aprendizagem (subajuste e sobreajuste)</h3>
<p>As curvas mostram a perda e a acurácia de treino e de validação por época, com o <b>checkpoint</b> (linha tracejada) na época de maior F1 macro de validação. A tabela compara treino e validação nesse ponto.</p>
{fig("resultados/figuras/mlp/MLP-B-128_aug_opt_preliminar_curvas.png", "MLP-B-128 (aumento leve, Optuna): a acurácia de treino fica baixa e a perda de validação sobe acima da de treino.", "100%", "190px")}
{fig("resultados/figuras/cnn/ResNet18-128_preliminar_curvas.png", "ResNet18-128: treino bem acima da validação (sobreajuste).", "100%", "190px")}
{fig("resultados/figuras/cnn/VGG16-256_preliminar_curvas.png", "VGG16-256: treino e validação próximos (a acurácia de treino é medida com o aumento ligado).", "100%", "190px")}
{tab_curvas(FINAIS)}
<p><b>MLPs:</b> a acurácia é baixa até no treino (0,74 e 0,80), sinal de <b>limite da arquitetura</b> (falta de noção espacial), com lacuna crescente na MLP-B (≈ 9 pontos), que começa a decorar o treino.
<b>CNNs:</b> a <b>ResNet18-128 sobreajusta</b> (lacuna de ≈ 6 pontos); ao subir para 256 a lacuna cai para ≈ 2 pontos. A VGG16 praticamente não tem lacuna. Todas param por parada antecipada (10 épocas sem melhora),
bem antes do limite de 200 épocas, e as CNNs convergem mais rápido que as MLPs (checkpoint entre as épocas 42 e 48).</p>
<h3>8.4 Efeito da resolução e custo</h3>
{fig("resultados/figuras/resumo/resumo_8_resolucao_custo.png", "Resolução 128 × 128 contra 256 × 256 nas CNNs finais: F1, tempo por época e tempo total.", "100%", "230px")}
{tabela(["CNN", "F1 em 128", "F1 em 256", "Ganho", "Custo por época", "Custo total"], [("ResNet18", "0,9132", "0,9426", "<b>+2,9 pontos</b>", "1,1 → 2,9 s (2,5×)", "65 → 149 s (2,3×)"),
("VGG16", "0,9488", "0,9570", "<b>+0,8 ponto</b>", "3,8 → 14,3 s (3,8×)", "217 → 802 s (3,7×)")], "t")}
<p>A resolução melhorou <b>as duas CNNs</b>. Na ResNet18 o ganho é grande para um custo moderado; na VGG16, o ganho é pequeno para um custo alto (13 minutos de treino). Os parâmetros são idênticos nas duas resoluções.
Em 128 × 128, as CNNs superam as MLPs em 21 a 27 pontos de F1 macro (0,913 e 0,949 contra 0,682 e 0,700).</p>
<h3>8.5 Busca de hiperparâmetros (Optuna): fase exploratória</h3>
<p>A busca é <b>adicional e separada das seis execuções finais</b>. Métrica otimizada: F1 macro de validação; amostrador TPE (semente 42) com poda pela mediana.
Só 3 a 4 tentativas de cada busca das CNNs chegam ao fim, então a busca é pequena. Melhor F1 da busca: MLPs 0,675 e 0,673; CNNs de 0,908 a 0,939 (com 30 épocas por tentativa).</p>
{fig("resultados/figuras/resumo/resumo_7_optuna_busca.png", "Tentativas concluídas e podadas, e melhor F1 de cada busca.", "100%", "240px")}
<p><b>Com e sem otimização</b> (aumento leve, mesma receita nas duas colunas):</p>
{tabela(["Modelo", "F1 sem otimização", "F1 com Optuna", "Diferença"], [(nm, f(e.loc[a, "f1_macro_validacao"]), f(e.loc[b, "f1_macro_validacao"]), dif(a, b))
for nm, a, b in [("MLP-A-128", "MLP-A-128_aug_preliminar", "MLP-A-128_aug_opt_preliminar"), ("MLP-B-128", "MLP-B-128_aug_preliminar", "MLP-B-128_aug_opt_preliminar"),
("ResNet18-128", "ResNet18-128_preliminar", "ResNet18-128_opt_preliminar"), ("ResNet18-256", "ResNet18-256_preliminar", "ResNet18-256_opt_preliminar"),
("VGG16-128", "VGG16-128_preliminar", "VGG16-128_opt_preliminar"), ("VGG16-256", "VGG16-256_preliminar", "VGG16-256_opt_preliminar")]], "t mini")}
<p>O Optuna <b>ajuda muito as MLPs</b> (+3,4 e +9,8 pontos) e <b>quase não muda as CNNs</b> (−0,7 a +0,3). Como os valores do Optuna das CNNs mudam entre 128 e 256 e o PDF exige manter lr entre resoluções,
as CNNs finais usam os valores iniciais.</p>
<h3>8.6 Mecanismos de atenção</h3>
{fig("resultados/figuras/resumo/resumo_6_atencao_efeito_medio.png", "Diferença média de F1 contra a mesma rede sem atenção, em 12 pares por barra (2 resoluções × 2 rodadas × 3 receitas de aumento).", "78%", "230px")}
<p><b>VGG16:</b> a atenção não ajudou (SE ≈ 0; CBAM e SE + CBAM pioram cerca de 1,2 ponto em média). <b>ResNet18:</b> melhora em média de +0,8 a +1,2 ponto, mas de forma <b>instável</b> (de −4,7 a +5,4).
O custo é de +0,5 a +1,6% de parâmetros e, na VGG16 em 256, o tempo por época vai de 14,3 s para 15,9 a 22,3 s. Com um treino por configuração, não tratamos isso como prova de superioridade;
as execuções finais ficam <b>sem atenção</b>.</p>
<h3>8.7 Classes confundidas e métricas por classe</h3>
{fig("resultados/figuras/resumo/resumo_4_f1_por_classe.png", "F1 por classe nas seis execuções finais.", "78%", "270px")}
{fig("resultados/figuras/mlp/matrizes/MLP-B-128_aug_opt_preliminar_confusao.png", "Matriz de confusão da MLP-B-128 (contagens e normalizada pela classe real; reais nas linhas, previstas nas colunas).", "100%", "290px")}
{fig("resultados/figuras/cnn/matrizes/VGG16-256_preliminar_confusao.png", "Matriz de confusão da VGG16-256 (contagens e normalizada pela classe real; reais nas linhas, previstas nas colunas).", "100%", "300px")}
<p><b>Precisão, recall, F1 e suporte por classe</b> (validação), de uma MLP e de uma CNN. Não houve divisão por zero: todas as classes foram previstas pelo menos uma vez (a menor precisão é 0,58 na MLP-B e 0,92 na VGG16-256).</p>
<div class="duas">{tab_classe("MLP-B-128 (aumento leve, Optuna)", "MLP-B-128_aug_opt_preliminar")}{tab_classe("VGG16-256 (aumento leve, inicial)", "VGG16-256_preliminar")}</div>
<p><b>MLPs:</b> erram cerca de 30% das imagens; confundem <code>airport</code> × <code>storage_tank</code> e <code>mountain</code> com <code>river</code> e <code>forest</code>.
<b>CNNs:</b> erram de 4% a 9%, sobretudo <code>desert</code> → <code>mountain</code> e <code>airport</code> ↔ <code>river</code>, <code>storage_tank</code> e <code>desert</code>. A hipótese prévia se confirmou em parte:
<code>desert</code> × <code>mountain</code> aparece; <code>beach</code> × <code>river</code> quase não. Em 256 melhora o F1 de <code>airport</code>, <code>storage_tank</code> e <code>mountain</code> na ResNet18.
As classes <code>forest</code>, <code>harbor</code>, <code>circular_farmland</code> e <code>dense_residential</code> são quase sempre acertadas pelas CNNs (F1 de 0,96 a 0,99).</p>
<h3>8.8 Panorama</h3>
{fig("resultados/figuras/resumo/resumo_9_panorama.png", "F1 macro de validação: das MLPs ao YOLO (bônus, pré-treinado). O YOLO não faz parte dos seis treinos e não isola a arquitetura.", "88%", "260px")}
<p class="peq">Além das seis execuções finais, treinamos {N_CNN} CNNs exploratórias (8 arquiteturas × 2 resoluções × hiperparâmetros × 3 receitas de aumento) para orientar as escolhas; elas estão identificadas à parte em
<code>resultados/experimentos.csv</code> e em <code>docs/CNN.md</code>.</p>""")

# ---------------------------------------------------------------- 9 próximos passos
P.append(f"""<h2>9. Limitações e cuidados</h2>
{tabela(["Limitação", "O que significa"], [("Um treino por configuração", "não medimos desvio entre sementes; diferenças de cerca de 1 ponto são ruído"),
("Escolhas na validação", "receita e checkpoint foram escolhidos com a validação, então o F1 de validação é um pouco otimista; o teste, avaliado uma vez, mostrará o quanto"),
("Quase-duplicatas entre partições", "5 pares (4 em <code>beach</code>, 1 em <code>mountain</code>) escaparam do MD5 e podem inflar um pouco essas classes"),
("Busca do Optuna pequena", "20 e 10 tentativas, com 60 e 30 épocas por tentativa (o treino final vai até 200); só 3 a 4 tentativas das CNNs concluem"),
("Hardware", "MLPs, CNNs e bônus treinaram na mesma A100; a busca das MLPs foi feita em outro computador e só os retreinos foram refeitos na A100"),
("Generalização", "não testamos outras regiões, sensores ou condições de captura; os resultados descrevem só estas partições do RESISC45")], "t")}
<h2>10. Reprodutibilidade</h2>
{tabela(["Item", "Como"], [("Semente e determinismo", "semente 42 em tudo, <code>cudnn.deterministic</code>; repetir um treino na mesma A100 reproduz F1, acurácia e época do checkpoint"),
("Versões", "Python 3, PyTorch 2.14.1 + CUDA 13, scikit-learn, pandas, matplotlib, Optuna (≥ 3.6); lista em <code>requirements.txt</code>"),
("Ordem de execução", "<code>setup_dataset</code> → <code>check_integrity</code> → <code>create_splits</code> → notebooks 01 (exploração), 02 (MLPs) e 03 (CNNs); gráficos em <code>src/graficos_*.py</code>"),
("Onde estão os resultados", "<code>resultados/experimentos.csv</code> (uma linha por treino), históricos, configurações, matrizes de confusão, métricas por classe, previsões e figuras"),
("Avaliar sem treinar de novo", "as previsões de validação estão em <code>resultados/previsoes_validacao_*.csv</code>; os checkpoints ficarão em armazenamento externo (link no README)")], "t")}
<h2>11. Próximos passos</h2>
{tabela(["Item", "Estado"], [("Dados, partições, exploração e duplicatas", "<span class='ok'>pronto</span>"), ("MLPs, CNNs, busca Optuna e bônus YOLO (validação)", "<span class='ok'>pronto</span>"),
("Escolha das seis execuções finais", "<span class='ok'>decidida</span>"), ("Retreino das seis execuções finais com os nomes finais", "<span class='fa'>a fazer</span>"),
("<b>Avaliação única no teste</b> (acurácia, F1 macro, métricas por classe, matrizes e até 6 erros)", "<span class='fa'>a fazer</span>"),
("Artigo (até 10 páginas) e pitch (10 a 15 min)", "<span class='fa'>a fazer</span>"), ("Modelos em armazenamento externo com link de download", "<span class='fa'>a fazer</span>"),
("Opcional: corrigir as quase-duplicatas, busca Optuna maior, inspeção visual dos erros", "<span class='op'>opcional</span>")], "t")}
<p class="peq">Regra: nenhuma escolha depois de olhar o teste; os seis modelos são avaliados juntos, uma única vez.</p>
<h2>12. Dúvidas para o professor</h2>
<ol class="duv">
<li>As <b>seis execuções finais</b> podem ser a receita comum (aumento leve; hiperparâmetros iniciais nas CNNs), com as MLPs usando os valores do Optuna? O Optuna das CNNs muda entre 128 e 256.</li>
<li>Podemos <b>reaproveitar</b> os treinos preliminares com essa receita como finais, ou o retreino com os nomes finais é obrigatório?</li>
<li>As <b>5 quase-duplicatas</b> entre partições (hash perceptual) exigem refazer as partições, ou basta registrar a limitação?</li>
<li>As buscas do Optuna e as {N_CNN} CNNs exploratórias estão bem <b>identificadas como buscas adicionais</b>, separadas das seis finais?</li>
<li>O modelo de maior F1 de validação para a análise de erros deve ser escolhido entre as seis finais ou entre todos os treinos?</li>
<li>Qual <b>armazenamento externo</b> é aceito para os checkpoints (5,5 GB de CNNs)?</li></ol>""")

CSS = """
@page { size: A4; margin: 16mm 15mm 18mm 15mm;
  @bottom-left { content: "Projeto 3 · Classificação de cenas aéreas · Acompanhamento"; font: 8pt sans-serif; color: #667 }
  @bottom-right { content: "Página " counter(page) " de " counter(pages); font: 8pt sans-serif; color: #667 } }
* { box-sizing: border-box; margin: 0; padding: 0 }
:root { --az:#2a78d6; --lj:#eb6834; --vd:#2e9d6a; --cz:#8a8f98; --rx:#7a5cc7; --tx:#1b2430; --mu:#5b6573 }
html, body { font-family: "Inter","Segoe UI","DejaVu Sans",Arial,sans-serif; color: var(--tx); font-size: 10.2pt; line-height: 1.42; -webkit-print-color-adjust: exact; print-color-adjust: exact }
.topo { border-left: 8px solid var(--az); padding: 4px 0 4px 14px; margin-bottom: 8px }
.kick { font-size: 8.5pt; letter-spacing: .13em; text-transform: uppercase; color: var(--lj); font-weight: 700 }
h1 { font-size: 21pt; line-height: 1.15; margin: 4px 0 }
.sub { font-size: 11.5pt; color: var(--mu) } .eq { font-size: 9.5pt; margin-top: 6px }
.aviso { background: #fff4e5; border: 1px solid #f3c78b; border-radius: 6px; padding: 6px 10px; font-size: 9.5pt; margin: 8px 0 4px }
h2 { font-size: 14.5pt; color: var(--az); margin: 16px 0 6px; padding-bottom: 3px; border-bottom: 2px solid #d9e6f8; break-after: avoid }
h3 { font-size: 11.2pt; margin: 11px 0 4px; break-after: avoid }
h4 { font-size: 10.2pt; margin: 10px 0 3px; color: var(--az); break-after: avoid }
p { margin: 5px 0 } p.peq { font-size: 9pt; color: var(--mu) }
table { border-collapse: collapse; width: 100%; font-size: 9.2pt; margin: 6px 0; break-inside: avoid }
table.mini { font-size: 8.4pt }
table.prog { font-size: 7.6pt } table.prog td, table.prog th { padding: 2px 4px; white-space: nowrap } table.prog td:nth-child(2) { white-space: normal; min-width: 34mm }
th { background: var(--az); color: #fff; text-align: left; padding: 3px 7px; font-weight: 600 }
td { padding: 3px 7px; border-bottom: 1px solid #e3e8ef; vertical-align: top }
tr:nth-child(even) td { background: #f6f8fc } tr.dest td { background: #fff2cc !important; font-weight: 700 }
code { font-size: 8.6pt; background: #eef2f8; padding: 0 3px; border-radius: 3px }
figure { margin: 8px 0; text-align: center; break-inside: avoid }
figure img { display: block; margin: 0 auto; object-fit: contain; max-width: 100% }
figcaption { font-size: 8.5pt; color: var(--mu); margin-top: 3px }
.duas { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; align-items: start; break-inside: avoid }
.kpis { display: grid; grid-template-columns: repeat(4,1fr); gap: 8px; margin: 8px 0 }
.kpi { border-radius: 8px; padding: 8px 10px; color: #fff; display: flex; flex-direction: column; gap: 1px }
.kpi b { font-size: 14pt; line-height: 1.1 } .kpi span { font-size: 8.3pt }
.kpi.az { background: var(--az) } .kpi.lj { background: var(--lj) } .kpi.vd { background: var(--vd) } .kpi.cz { background: var(--cz) }
.fx { display: flex; align-items: stretch; gap: 3px; margin: 6px 0 2px; break-inside: avoid }
.fx .b { flex: 1; border-radius: 6px; padding: 5px 3px; text-align: center; color: #fff; display: flex; flex-direction: column; justify-content: center; gap: 1px; min-height: 40px }
.fx .b b { font-size: 8.8pt; line-height: 1.1 } .fx .b span { font-size: 7pt; line-height: 1.15; opacity: .96 }
.fx i { font-style: normal; align-self: center; font-size: 10pt; color: var(--mu) }
.fx.peq .b span { font-size: 6.6pt }
.fx .b.cz { background: var(--cz) } .fx .b.az { background: var(--az) } .fx .b.lj { background: var(--lj) } .fx .b.vd { background: var(--vd) } .fx .b.rx { background: var(--rx) }
.leg { font-size: 8.8pt; color: var(--mu); margin: 2px 0 6px; font-weight: 600 }
.ok { color: #1d7a50; font-weight: 700 } .fa { color: #c4501a; font-weight: 700 } .op { color: #6b7380; font-weight: 700 }
.duv { margin: 6px 0 0 20px; display: flex; flex-direction: column; gap: 5px }
"""
HTML = f"<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'><title>Acompanhamento — Projeto 3</title><style>{CSS}</style></head><body>{''.join(P)}</body></html>"


def main():
    chrome = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")
    if not chrome:
        raise SystemExit("Chrome/Chromium não encontrado para imprimir o PDF")
    with tempfile.TemporaryDirectory() as d:
        html = Path(d) / "acompanhamento.html"
        html.write_text(HTML, encoding="utf-8")
        SAIDA.parent.mkdir(exist_ok=True)
        subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-sandbox", "--allow-file-access-from-files", "--no-pdf-header-footer",
                        f"--print-to-pdf={SAIDA}", html.as_uri()], check=True, capture_output=True, timeout=180)
    print("PDF:", SAIDA)


if __name__ == "__main__":
    main()
