"""Avalia na VALIDAÇÃO o YOLO (modo classify) treinado em notebooks/03_CNNs.ipynb, seção 7 (o teste não é usado).

Lê resultados/yolo/yolo_cls_256/weights/best.pt e grava em resultados/yolo/:
  yolo_cls_256_matriz_confusao.csv      contagens (linhas = real, colunas = prevista), ordem fixa das classes
  yolo_cls_256_metricas_por_classe.csv  precisão, recall, F1 e suporte por classe
  yolo_cls_256_previsoes_validacao.csv  arquivo, classe real, prevista e confiança
  yolo_cls_256_resumo.json              F1 macro, acurácia, custo (parâmetros, tamanho, épocas, tempo) e hardware

Uso (a partir da raiz do projeto): python src/avaliar_yolo_validacao.py
"""
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from ultralytics import YOLO

RAIZ = Path(__file__).resolve().parent.parent
CLASSES = ("airport", "beach", "circular_farmland", "dense_residential", "desert",
           "forest", "harbor", "mountain", "river", "storage_tank")
IMGSZ, LOTE = 256, 32
SAIDA = RAIZ / "resultados" / "yolo"
EXEC = SAIDA / f"yolo_cls_{IMGSZ}"


def main():
    pesos = EXEC / "weights" / "best.pt"
    modelo = YOLO(str(pesos))
    part = pd.read_csv(RAIZ / "dados" / "splits" / "particoes.csv").sort_values("caminho_relativo")
    val = part[part["particao"] == "validacao"].reset_index(drop=True)
    previsoes = modelo.predict([str(RAIZ / c) for c in val["caminho_relativo"]], imgsz=IMGSZ, batch=LOTE, verbose=False)
    previstas = [modelo.names[r.probs.top1] for r in previsoes]
    confianca = [float(r.probs.top1conf) for r in previsoes]
    reais = val["classe"].tolist()

    cm = confusion_matrix(reais, previstas, labels=list(CLASSES))
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(SAIDA / f"yolo_cls_{IMGSZ}_matriz_confusao.csv")
    rel = classification_report(reais, previstas, labels=list(CLASSES), output_dict=True, zero_division=0)
    pd.DataFrame(rel).T.to_csv(SAIDA / f"yolo_cls_{IMGSZ}_metricas_por_classe.csv")
    pd.DataFrame({"caminho_relativo": val["caminho_relativo"], "classe_real": reais,
                  "classe_prevista": previstas, "confianca": confianca}).to_csv(
        SAIDA / f"yolo_cls_{IMGSZ}_previsoes_validacao.csv", index=False)

    res = pd.read_csv(EXEC / "results.csv")
    res.columns = [c.strip() for c in res.columns]
    epocas = int(res["epoch"].max())
    tempo_total = float(res["time"].iloc[-1])           # tempo acumulado (s) informado pelo Ultralytics
    melhor_epoca = int(res.loc[res["metrics/accuracy_top1"].idxmax(), "epoch"])
    parametros = sum(p.numel() for p in modelo.model.parameters())
    resumo = {
        "nome_execucao": f"YOLO26n-cls-{IMGSZ}_bonus",
        "f1_macro_validacao": round(float(f1_score(reais, previstas, labels=list(CLASSES), average="macro", zero_division=0)), 4),
        "acuracia_validacao": round(float(accuracy_score(reais, previstas)), 4),
        "parametros": int(parametros),
        "tamanho_modelo_mb": round(pesos.stat().st_size / 1024 ** 2, 2),
        "epocas_executadas": epocas, "epoca_melhor_acuracia": melhor_epoca,
        "tempo_total_s": round(tempo_total, 1), "tempo_medio_epoca_s": round(tempo_total / epocas, 2),
        "hardware": f"GPU {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU",
        "torch": torch.__version__,
    }
    (SAIDA / f"yolo_cls_{IMGSZ}_resumo.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resumo, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
