"""Avalia na VALIDAÇÃO os checkpoints das CNNs salvos em modelos/ (o teste não é usado).

Gera, para cada modelo:
  resultados/matrizes_confusao/{nome}_cm.csv        contagens (linhas = real, colunas = prevista)
  resultados/metricas_por_classe/{nome}.csv         precisão, recall, F1 e suporte por classe
e, para todos juntos:
  resultados/previsoes_validacao_cnn.csv            arquivo, classe real, prevista e confiança

As classes das arquiteturas são lidas do próprio notebook 03_CNNs.ipynb (células com `class ...(nn.Module)`),
para que a avaliação use exatamente o mesmo código do treino.

Uso (a partir da raiz do projeto): python src/avaliar_cnn_validacao.py
"""
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

RAIZ = Path(__file__).resolve().parent.parent
CLASSES = ("airport", "beach", "circular_farmland", "dense_residential", "desert",
           "forest", "harbor", "mountain", "river", "storage_tank")
# AVALIAR_DISPOSITIVO=cpu: avalia na CPU (use enquanto um treino ocupa a GPU, para não tirar memória dele)
# AVALIAR_FORCAR=1: recalcula mesmo os modelos que já têm matriz e métricas salvas
dispositivo = torch.device(os.environ.get("AVALIAR_DISPOSITIVO") or ("cuda" if torch.cuda.is_available() else "cpu"))
FORCAR = os.environ.get("AVALIAR_FORCAR") == "1"


def carregar_arquiteturas():
    """Executa as células do notebook que definem as classes das CNNs e devolve o namespace."""
    nb = json.loads((RAIZ / "notebooks" / "03_CNNs.ipynb").read_text(encoding="utf-8"))
    ns = {}
    for celula in nb["cells"]:
        codigo = "".join(celula["source"])
        if celula["cell_type"] == "code" and re.search(r"^class \w+\(nn\.Module\)", codigo, re.M):
            exec(codigo, ns)
    return ns


def main():
    ns = carregar_arquiteturas()
    part = pd.read_csv(RAIZ / "dados" / "splits" / "particoes.csv").sort_values("caminho_relativo")
    val = part[part["particao"] == "validacao"].reset_index(drop=True)
    rotulo = {c: i for i, c in enumerate(CLASSES)}
    y = val["classe"].map(rotulo).to_numpy()

    cache_imagens = {}

    def imagens(res):
        if res not in cache_imagens:
            arr = np.empty((len(val), 3, res, res), np.uint8)
            for i, caminho in enumerate(val["caminho_relativo"]):
                with Image.open(RAIZ / caminho) as img:
                    arr[i] = np.asarray(img.convert("RGB").resize((res, res), Image.BILINEAR), dtype=np.uint8).transpose(2, 0, 1)
            cache_imagens[res] = torch.from_numpy(arr)
        return cache_imagens[res]

    (RAIZ / "resultados" / "matrizes_confusao").mkdir(parents=True, exist_ok=True)
    (RAIZ / "resultados" / "metricas_por_classe").mkdir(parents=True, exist_ok=True)
    previsoes = []
    for ckpt in sorted((RAIZ / "modelos").glob("*.pt")):
        nome = ckpt.stem
        m = re.match(r"^([A-Za-z0-9]+)-(\d+)_", nome)
        if not m or m.group(1) not in ns:
            continue                                   # ignora MLPs e YOLO
        arq, res = m.group(1), int(m.group(2))
        if not (RAIZ / "resultados" / "configs" / f"{nome}.json").is_file():
            print(f"{nome}: treino ainda em andamento (sem config), pulando", flush=True)
            continue
        if not FORCAR and (RAIZ / "resultados" / "matrizes_confusao" / f"{nome}_cm.csv").is_file() \
                and (RAIZ / "resultados" / "metricas_por_classe" / f"{nome}.csv").is_file():
            continue                                   # já avaliado
        cfg = json.loads((RAIZ / "resultados" / "configs" / f"{nome}.json").read_text(encoding="utf-8"))
        media = torch.tensor(cfg["normalizacao"]["media"], dtype=torch.float32, device=dispositivo).view(1, 3, 1, 1)
        desvio = torch.tensor(cfg["normalizacao"]["desvio"], dtype=torch.float32, device=dispositivo).view(1, 3, 1, 1)
        modelo = ns[arq]().to(dispositivo)
        modelo.load_state_dict(torch.load(ckpt, map_location=dispositivo)["state_dict"])
        modelo.eval()
        X = imagens(res)
        probs = []
        with torch.no_grad():
            for i in range(0, len(X), 128):
                xb = (X[i:i + 128].to(dispositivo).float() / 255.0 - media) / desvio
                probs.append(torch.softmax(modelo(xb), 1).cpu())
        probs = torch.cat(probs).numpy()
        prev = probs.argmax(1)
        cm = confusion_matrix(y, prev, labels=range(len(CLASSES)))
        pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(RAIZ / "resultados" / "matrizes_confusao" / f"{nome}_cm.csv")
        rel = classification_report(y, prev, labels=range(len(CLASSES)), target_names=CLASSES, output_dict=True, zero_division=0)
        pd.DataFrame(rel).T.to_csv(RAIZ / "resultados" / "metricas_por_classe" / f"{nome}.csv")
        previsoes.append(pd.DataFrame({"nome_execucao": nome, "caminho_relativo": val["caminho_relativo"],
                                       "classe_real": val["classe"], "classe_prevista": [CLASSES[p] for p in prev],
                                       "confianca": probs.max(1)}))
        print(f"{nome}: acurácia {(prev == y).mean():.4f} | F1 macro {rel['macro avg']['f1-score']:.4f}", flush=True)
        del modelo
        torch.cuda.empty_cache()
    if previsoes:
        # junta com as previsões já salvas (de modelos avaliados antes), trocando as dos modelos reavaliados agora
        destino = RAIZ / "resultados" / "previsoes_validacao_cnn.csv"
        novas = pd.concat(previsoes, ignore_index=True)
        if destino.is_file():
            antigas = pd.read_csv(destino)
            novas = pd.concat([antigas[~antigas["nome_execucao"].isin(novas["nome_execucao"])], novas], ignore_index=True)
        novas.to_csv(destino, index=False)


if __name__ == "__main__":
    main()
