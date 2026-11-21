"""Loop de treino genérico (MLP e CNN) com seleção por F1 macro na validação.

Protocolo:
- Perda: CrossEntropyLoss (classificação multiclasse, 10 classes balanceadas).
- Seleção do checkpoint: maior F1 macro na VALIDAÇÃO (métrica principal do projeto).
- Parada antecipada: monitora o F1 macro de validação com paciência configurável.
- O conjunto de TESTE não é usado aqui.
- Tempo medido apenas no laço de treino+validação de cada época (exclui leitura
  dos arquivos e pré-processamento prévio), com torch.cuda.synchronize em GPU.
"""

from __future__ import annotations

import json
import os
import platform
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class ConfigTreino:
    nome_execucao: str
    otimizador: str = "adam"
    taxa_aprendizagem: float = 1e-4
    weight_decay: float = 1e-4
    tamanho_lote: int = 64
    max_epocas: int = 60
    paciencia: int = 10
    semente: int = 42
    tipo_execucao: str = "preliminar"  # "preliminar" ou "final"
    extras: dict = field(default_factory=dict)


def fixar_semente(semente: int) -> None:
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)
    torch.cuda.manual_seed_all(semente)
    os.environ["PYTHONHASHSEED"] = str(semente)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def obter_dispositivo() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def descrever_hardware(dispositivo: torch.device) -> str:
    if dispositivo.type == "cuda":
        return f"GPU {torch.cuda.get_device_name(0)}"
    if dispositivo.type == "mps":
        return f"Apple MPS ({platform.processor() or platform.machine()})"
    return f"CPU {platform.processor() or platform.machine()} ({os.cpu_count()} núcleos lógicos)"


def criar_loader(X: np.ndarray, y: np.ndarray, lote: int, embaralhar: bool, semente: int) -> DataLoader:
    X = X if X.flags.writeable else X.copy()
    ds = TensorDataset(torch.from_numpy(X), torch.tensor(y))
    gerador = torch.Generator().manual_seed(semente)
    return DataLoader(ds, batch_size=lote, shuffle=embaralhar, generator=gerador)


def _sincronizar(dispositivo: torch.device) -> None:
    if dispositivo.type == "cuda":
        torch.cuda.synchronize()


def _rodar_epoca(modelo, loader, perda_fn, dispositivo, otimizador=None):
    """Treina (se otimizador) ou avalia. Retorna perda média, acurácia, F1 macro."""
    treinando = otimizador is not None
    modelo.train(treinando)
    soma_perda, preds, reais = 0.0, [], []
    with torch.set_grad_enabled(treinando):
        for xb, yb in loader:
            xb, yb = xb.to(dispositivo), yb.to(dispositivo)
            logits = modelo(xb)
            perda = perda_fn(logits, yb)
            if treinando:
                otimizador.zero_grad(set_to_none=True)
                perda.backward()
                otimizador.step()
            soma_perda += perda.item() * yb.size(0)
            preds.append(logits.argmax(1).cpu())
            reais.append(yb.cpu())
    p = torch.cat(preds).numpy()
    r = torch.cat(reais).numpy()
    return (
        soma_perda / len(r),
        accuracy_score(r, p),
        f1_score(r, p, average="macro", zero_division=0),
    )


def criar_otimizador(cfg: ConfigTreino, params):
    if cfg.otimizador == "adam":
        return torch.optim.Adam(params, lr=cfg.taxa_aprendizagem, weight_decay=cfg.weight_decay)
    if cfg.otimizador == "sgd":
        return torch.optim.SGD(params, lr=cfg.taxa_aprendizagem, momentum=0.9, weight_decay=cfg.weight_decay)
    raise ValueError(f"Otimizador desconhecido: {cfg.otimizador}")


def treinar(
    modelo: nn.Module,
    dados_treino: tuple[np.ndarray, np.ndarray],
    dados_val: tuple[np.ndarray, np.ndarray],
    cfg: ConfigTreino,
    dir_modelos: Path,
    dir_resultados: Path,
) -> dict:
    """Treina do zero, salva o melhor checkpoint (F1 macro val) e o histórico."""
    dispositivo = obter_dispositivo()
    modelo = modelo.to(dispositivo)
    perda_fn = nn.CrossEntropyLoss()
    otimizador = criar_otimizador(cfg, modelo.parameters())

    loader_tr = criar_loader(*dados_treino, cfg.tamanho_lote, True, cfg.semente)
    loader_val = criar_loader(*dados_val, cfg.tamanho_lote, False, cfg.semente)

    dir_modelos.mkdir(parents=True, exist_ok=True)
    (dir_resultados / "historicos").mkdir(parents=True, exist_ok=True)
    caminho_ckpt = dir_modelos / f"{cfg.nome_execucao}.pt"

    historico: list[dict] = []
    melhor_f1, melhor_epoca, sem_melhora = -1.0, 0, 0
    tempo_total = 0.0

    for epoca in range(1, cfg.max_epocas + 1):
        _sincronizar(dispositivo)
        t0 = time.perf_counter()
        tr_loss, tr_acc, tr_f1 = _rodar_epoca(modelo, loader_tr, perda_fn, dispositivo, otimizador)
        va_loss, va_acc, va_f1 = _rodar_epoca(modelo, loader_val, perda_fn, dispositivo)
        _sincronizar(dispositivo)
        dt = time.perf_counter() - t0
        tempo_total += dt

        historico.append({
            "epoca": epoca,
            "perda_treino": tr_loss, "acuracia_treino": tr_acc, "f1_macro_treino": tr_f1,
            "perda_validacao": va_loss, "acuracia_validacao": va_acc, "f1_macro_validacao": va_f1,
            "tempo_epoca_s": dt,
        })

        melhorou = va_f1 > melhor_f1
        if melhorou:
            melhor_f1, melhor_epoca, sem_melhora = va_f1, epoca, 0
            torch.save({
                "state_dict": modelo.state_dict(),
                "epoca": epoca,
                "f1_macro_validacao": va_f1,
                "config_treino": asdict(cfg),
            }, caminho_ckpt)
        else:
            sem_melhora += 1

        print(
            f"[{cfg.nome_execucao}] época {epoca:3d} | "
            f"treino perda {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val perda {va_loss:.4f} acc {va_acc:.4f} F1 {va_f1:.4f} | "
            f"{dt:6.1f}s" + ("  *checkpoint*" if melhorou else "")
        )
        if sem_melhora >= cfg.paciencia:
            print(f"Parada antecipada: {cfg.paciencia} épocas sem melhora no F1 macro de validação.")
            break

    df_hist = pd.DataFrame(historico)
    df_hist["checkpoint"] = df_hist["epoca"] == melhor_epoca
    df_hist.to_csv(dir_resultados / "historicos" / f"{cfg.nome_execucao}_historico.csv", index=False)

    linha_ckpt = df_hist.loc[df_hist["epoca"] == melhor_epoca].iloc[0]
    total_params = sum(p.numel() for p in modelo.parameters())
    resumo = {
        "nome_execucao": cfg.nome_execucao,
        "tipo_execucao": cfg.tipo_execucao,
        "parametros": total_params,
        "tamanho_modelo_mb": round(caminho_ckpt.stat().st_size / 1024**2, 2),
        "epocas_executadas": len(df_hist),
        "epoca_checkpoint": melhor_epoca,
        "f1_macro_validacao": round(float(linha_ckpt["f1_macro_validacao"]), 4),
        "acuracia_validacao": round(float(linha_ckpt["acuracia_validacao"]), 4),
        "perda_validacao": round(float(linha_ckpt["perda_validacao"]), 4),
        "tempo_total_s": round(tempo_total, 1),
        "tempo_medio_epoca_s": round(tempo_total / len(df_hist), 2),
        "hardware": descrever_hardware(dispositivo),
        "otimizador": cfg.otimizador,
        "taxa_aprendizagem": cfg.taxa_aprendizagem,
        "weight_decay": cfg.weight_decay,
        "tamanho_lote": cfg.tamanho_lote,
        "max_epocas": cfg.max_epocas,
        "paciencia": cfg.paciencia,
        "criterio_parada": "F1 macro de validação",
        "semente": cfg.semente,
        "torch": torch.__version__,
        "extras": json.dumps(cfg.extras, ensure_ascii=False),
    }
    registrar_experimento(resumo, dir_resultados / "experimentos.csv")
    return {"resumo": resumo, "historico": df_hist, "checkpoint": caminho_ckpt}


def registrar_experimento(resumo: dict, csv_path: Path) -> None:
    """Acrescenta/substitui a linha da execução em resultados/experimentos.csv."""
    novo = pd.DataFrame([resumo])
    if csv_path.is_file():
        antigo = pd.read_csv(csv_path)
        mask = (antigo["nome_execucao"] == resumo["nome_execucao"]) & (
            antigo["tipo_execucao"] == resumo["tipo_execucao"]
        )
        novo = pd.concat([antigo.loc[~mask], novo], ignore_index=True)
    novo.to_csv(csv_path, index=False)
