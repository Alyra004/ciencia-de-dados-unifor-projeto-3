"""Retreina as 12 MLPs do notebook 02_mlp_preliminar.ipynb numa GPU (A100) e avalia cada uma na VALIDAÇÃO.

Objetivo: ter as MLPs no MESMO hardware das CNNs e do YOLO (o PDF pede para preservar o hardware), sem mexer nos
resultados do Mac. A receita é exatamente a do notebook (mesmo código, mesma semente, mesmos hiperparâmetros):
  - 6 treinos com hiperparâmetros fixos (sem aumento, aumento leve e aumento forte, MLP-A e MLP-B);
  - 6 treinos com os hiperparâmetros que o Optuna achou na primeira rodada (resultados/optuna/melhores_hiperparametros.json).
A busca do Optuna NÃO é refeita: só o retreino com os valores já encontrados.

O script executa as células do notebook (as que só definem funções e as de treino) e grava tudo numa árvore separada:
  <destino>/modelos/, <destino>/resultados/{historicos,configs,figuras/mlp,experimentos.csv}
e, para cada modelo, na validação (o teste não é usado):
  <destino>/resultados/matrizes_confusao/{nome}_cm.csv, metricas_por_classe/{nome}.csv e previsoes_validacao_mlp.csv

Uso (na VM, com o projeto em <origem>): python src/rodar_mlps_a100.py [origem] [destino]
"""
import contextlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

ORIGEM = Path(sys.argv[1] if len(sys.argv) > 1 else "/content/projeto")
DESTINO = Path(sys.argv[2] if len(sys.argv) > 2 else "/content/projeto_a100")
NOTEBOOK = ORIGEM / "notebooks" / "02_mlp_preliminar.ipynb"

# árvore de destino: dados por link simbólico, saídas separadas
(DESTINO / "notebooks").mkdir(parents=True, exist_ok=True)
for pasta in ("modelos", "resultados"):
    (DESTINO / pasta).mkdir(exist_ok=True)
if not (DESTINO / "dados").exists():
    os.symlink(ORIGEM / "dados", DESTINO / "dados")
os.chdir(DESTINO / "notebooks")

nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def celula(i):
    c = nb["cells"][i]
    assert c["cell_type"] == "code", i
    return "".join(c["source"])


ns = {"__name__": "__main__"}
log = open(DESTINO / "treino_mlps_a100.log", "w", encoding="utf-8")


def rodar(indices):
    for i in indices:
        with contextlib.redirect_stdout(log):
            exec(compile(celula(i), f"<celula {i}>", "exec"), ns)
        log.flush()


# 1) imports, dados, arquiteturas e funções de treino (células que só definem coisas)
rodar([2, 4, 8, 10])
print("hardware:", ns["hardware"], "| treino:", ns["X_tr"].shape, "| validação:", ns["X_va"].shape, flush=True)
# 2) seis treinos com hiperparâmetros fixos, na mesma ordem do notebook
rodar([12, 15, 20, 23, 28, 31])
print("6 treinos com hiperparâmetros fixos concluídos", flush=True)
# 3) seis treinos com os hiperparâmetros do Optuna (primeira rodada, no Mac)
import optuna  # noqa: E402  (as células 43 e 51 citam o módulo, mesmo sem rodar a busca)
ns["optuna"] = optuna
rodar([41, 43])
salvos = json.loads((ORIGEM / "resultados" / "optuna" / "melhores_hiperparametros.json").read_text(encoding="utf-8"))
ns["melhores"] = {a: {k: salvos[f"MLP-{a}-128"][k] for k in ("lr", "weight_decay", "dropout", "lote")} for a in "AB"}
rodar([51])
print("6 treinos com Optuna concluídos", flush=True)

# 4) avaliação na validação: matriz de confusão, métricas por classe e previsões
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402

CLASSES, X_va, y_va, dispositivo = ns["CLASSES"], ns["X_va"], ns["y_va"], ns["dispositivo"]
(DESTINO / "resultados" / "matrizes_confusao").mkdir(exist_ok=True)
(DESTINO / "resultados" / "metricas_por_classe").mkdir(exist_ok=True)
previsoes = []
runs = [(f"MLP-{a}-128{s}_preliminar", a, None) for a in "AB" for s in ("", "_aug", "_aug2")] + \
       [(f"MLP-{a}-128{s}_opt_preliminar", a, ns["melhores"][a]["dropout"]) for a in "AB" for s in ("", "_aug", "_aug2")]
for nome, arq, dropout in runs:
    modelo = (ns["MLPA"]() if arq == "A" else ns["MLPB"]()) if dropout is None else \
             (ns["MLPA_HP"](dropout) if arq == "A" else ns["MLPB_HP"](dropout))
    modelo.load_state_dict(torch.load(DESTINO / "modelos" / f"{nome}.pt", map_location=dispositivo)["state_dict"])
    modelo = modelo.to(dispositivo).eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X_va), 256):
            probs.append(torch.softmax(modelo(torch.from_numpy(X_va[i:i + 256]).to(dispositivo)), 1).cpu())
    probs = torch.cat(probs).numpy()
    prev = probs.argmax(1)
    pd.DataFrame(confusion_matrix(y_va, prev, labels=range(10)), index=CLASSES, columns=CLASSES).to_csv(
        DESTINO / "resultados" / "matrizes_confusao" / f"{nome}_cm.csv")
    rel = classification_report(y_va, prev, labels=range(10), target_names=CLASSES, output_dict=True, zero_division=0)
    pd.DataFrame(rel).T.to_csv(DESTINO / "resultados" / "metricas_por_classe" / f"{nome}.csv")
    particoes = ns["particoes"]
    val = particoes[particoes["particao"] == "validacao"]
    previsoes.append(pd.DataFrame({"nome_execucao": nome, "caminho_relativo": val["caminho_relativo"].to_numpy(),
                                   "classe_real": [CLASSES[k] for k in y_va], "classe_prevista": [CLASSES[k] for k in prev],
                                   "confianca": probs.max(1)}))
    print(f"{nome}: acurácia {(prev == y_va).mean():.4f} | F1 macro {rel['macro avg']['f1-score']:.4f}", flush=True)
pd.concat(previsoes, ignore_index=True).to_csv(DESTINO / "resultados" / "previsoes_validacao_mlp.csv", index=False)
print("MLPS A100 TERMINARAM", flush=True)
