"""Treina as 8 CNNs do notebook 03_CNNs.ipynb com as receitas de aumento de dados que faltavam, numa resolução.

O notebook treinou as CNNs só com o aumento LEVE (espelhamentos e rotações de 90°). Aqui entram:
  - SEM aumento de dados;
  - aumento FORTE (o mesmo das MLPs: leve + recorte com zoom + variação de cor + apagamento aleatório).
Cada receita roda com os hiperparâmetros INICIAIS (lr 1e-4, weight decay 1e-4, lote 64) e com os hiperparâmetros do
OPTUNA. Como nas MLPs, a busca do Optuna NÃO é refeita por receita: usam-se os melhores valores da busca feita com
aumento leve (resultados/optuna/tentativas_<família>-<resolução>.csv).

Total por resolução: 8 CNNs x 2 receitas x 2 rodadas = 32 treinos. Nomes dos arquivos (sem colidir com os do aumento leve):
  <CNN>-<R>_semaug_preliminar      sem aumento, hiperparâmetros iniciais
  <CNN>-<R>_aug2_preliminar        aumento forte, hiperparâmetros iniciais
  <CNN>-<R>_semaug_opt_preliminar  sem aumento, hiperparâmetros do Optuna
  <CNN>-<R>_aug2_opt_preliminar    aumento forte, hiperparâmetros do Optuna

O script executa as células do notebook que definem as arquiteturas, os dados e o laço de treino (mesmo código dos
treinos anteriores) e só valida (o teste não é usado).

Uso (a partir da raiz do projeto, na VM com GPU): python src/rodar_cnns_aumentos.py <128|256> [raiz]
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd

RES = int(sys.argv[1])
assert RES in (128, 256), "resolução deve ser 128 ou 256"
RAIZ = Path(sys.argv[2] if len(sys.argv) > 2 else Path(__file__).resolve().parent.parent)
os.environ.setdefault("MPLBACKEND", "Agg")
os.chdir(RAIZ / "notebooks")

nb = json.loads((RAIZ / "notebooks" / "03_CNNs.ipynb").read_text(encoding="utf-8"))
nb_mlp = json.loads((RAIZ / "notebooks" / "02_mlp_preliminar.ipynb").read_text(encoding="utf-8"))


def celula(notebook, i):
    c = notebook["cells"][i]
    assert c["cell_type"] == "code", i
    return "".join(c["source"])


ns = {"__name__": "__main__"}
cel7 = celula(nb, 7)
assert "RESOLUCAO = 128 " in cel7
cel7 = cel7.replace("RESOLUCAO = 128 ", f"RESOLUCAO = {RES} ", 1)
# arquiteturas (3 e 5), configuração e semente (7), dados e aumento leve (9), laço de treino (11)
for origem, codigo in (("3", celula(nb, 3)), ("5", celula(nb, 5)), ("7", cel7), ("9", celula(nb, 9)), ("11", celula(nb, 11))):
    exec(compile(codigo, f"<celula {origem}>", "exec"), ns)

# aumento forte: a função das MLPs (célula 10 do notebook 02), aplicada depois do aumento leve
cel10 = celula(nb_mlp, 10)
forte = cel10[cel10.index("def aumentar_forte"):cel10.index("def treinar(")]
exec(compile(forte, "<aumentar_forte>", "exec"), ns)
import optuna  # noqa: E402  (a função de treino cita o módulo, mesmo sem rodar a busca)

ns["optuna"] = optuna
ns["DESCRICOES_AUMENTO"]["forte"] = (
    "espelhamentos e rotações de 90°, recorte com zoom (50-100% da área), variação de cor (brilho, contraste e saturação ±20%) "
    "e apagamento aleatório (p = 0,5; 10-40% do lado), só no treino")
aumentar_leve = ns["aumentar"]


def aumentar_composto(x, g):
    return ns["aumentar_forte"](aumentar_leve(x, g), g)


MODELOS = {a: ns[a] for a in ("ResNet18", "ResNet18SE", "ResNet18CBAM", "ResNet18SECBAM",
                              "VGG16", "VGG16SE", "VGG16CBAM", "VGG16SECBAM")}
FAMILIA = {a: ("ResNet18" if a.startswith("ResNet") else "VGG16") for a in MODELOS}


def melhores_optuna(familia):
    d = pd.read_csv(RAIZ / "resultados" / "optuna" / f"tentativas_{familia}-{RES}.csv")
    b = d.loc[d["value"].idxmax()]
    return {"lr": float(b["params_lr"]), "weight_decay": float(b["params_weight_decay"]), "lote": int(b["params_lote"])}


HP_OPT = {f: melhores_optuna(f) for f in ("ResNet18", "VGG16")}
print("hiperparâmetros do Optuna usados:", HP_OPT, flush=True)

RECEITAS = (("nenhum", False, "semaug"), ("forte", "forte", "aug2"))
for rodada in ("inicial", "optuna"):
    for receita, aumento, tag in RECEITAS:
        ns["aumentar"] = aumentar_composto if aumento == "forte" else aumentar_leve
        for arq, Modelo in MODELOS.items():
            nome = f"{arq}-{RES}_{tag}" + ("_opt" if rodada == "optuna" else "") + "_preliminar"
            if ns["resultado_salvo"](nome) is not None:
                print(f"##### {nome}: já treinado, pulando", flush=True)
                continue
            hp = HP_OPT[FAMILIA[arq]] if rodada == "optuna" else \
                {"lr": ns["LR_CNN"], "weight_decay": ns["WD_CNN"], "lote": ns["LOTE_CNN"]}
            print(f"##### {nome} | aumento {receita} | rodada {rodada} | lr {hp['lr']:.2e} wd {hp['weight_decay']:.2e} lote {hp['lote']}", flush=True)
            resumo, hist = ns["treinar_cnn"](
                Modelo, arq, nome, aumento=aumento, lr=hp["lr"], weight_decay=hp["weight_decay"], lote=hp["lote"],
                max_epocas=ns["MAX_EPOCAS_CNN"], tipo="preliminar_optuna" if rodada == "optuna" else "preliminar",
                extras={"familia": FAMILIA[arq], "busca": "optuna (feita com aumento leve)" if rodada == "optuna" else "hiperparâmetros iniciais",
                        "receita_aumento": receita})
            ns["plotar_curvas_cnn"](hist, nome)
            ns["liberar_memoria"]()
print(f"AUMENTOS {RES} TERMINARAM", flush=True)
