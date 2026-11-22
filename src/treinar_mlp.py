"""Treina uma MLP (Experimento 1) em RGB 128x128 achatado.

Uso:
    python src/treinar_mlp.py --variante A                 # MLP-A-128, execução preliminar
    python src/treinar_mlp.py --variante A --tipo final    # execução final (Colab)

Saídas:
    modelos/<nome>.pt                                  melhor checkpoint (F1 macro val)
    resultados/historicos/<nome>_historico.csv         métricas por época
    resultados/experimentos.csv                        uma linha por execução
    resultados/figuras/mlp/<nome>_curvas.png               curvas de perda/acurácia/F1
    resultados/configs/<nome>.json                     configuração + normalização + versões
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import PIL
import sklearn
import torch

from dados_imagens import CLASSES, PROJECT_ROOT, preparar_dados_mlp
from graficos import plotar_curvas
from modelos_mlp import MLP_CONFIGS, criar_mlp, resumo
from treino import ConfigTreino, fixar_semente, treinar


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variante", choices=sorted(MLP_CONFIGS), default="A")
    ap.add_argument("--resolucao", type=int, default=128)
    ap.add_argument("--tipo", choices=["preliminar", "final"], default="preliminar")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lote", type=int, default=64)
    ap.add_argument("--epocas", type=int, default=60)
    ap.add_argument("--paciencia", type=int, default=10)
    ap.add_argument("--semente", type=int, default=42)
    args = ap.parse_args()

    nome = f"MLP-{args.variante}-{args.resolucao}"
    sufixo = "" if args.tipo == "final" else "_preliminar"
    nome_execucao = nome + sufixo

    fixar_semente(args.semente)

    print("Carregando e pré-processando imagens (treino e validação)...")
    dados, norm = preparar_dados_mlp(resolucao=args.resolucao)
    X_tr, y_tr, _ = dados["treino"]
    X_va, y_va, _ = dados["validacao"]
    print(f"Treino: {X_tr.shape}  Validação: {X_va.shape}  (entrada vetorial de {X_tr.shape[1]:,} valores)")
    print(f"Normalização (stats do treino) média={np.round(norm.media, 4)} desvio={np.round(norm.desvio, 4)}")

    modelo = criar_mlp(args.variante, args.resolucao, n_classes=len(CLASSES))
    print(resumo(modelo, X_tr.shape[1]))

    cfg = ConfigTreino(
        nome_execucao=nome_execucao,
        taxa_aprendizagem=args.lr,
        tamanho_lote=args.lote,
        max_epocas=args.epocas,
        paciencia=args.paciencia,
        semente=args.semente,
        tipo_execucao=args.tipo,
        extras={"arquitetura": nome, **MLP_CONFIGS[args.variante]},
    )

    dir_res = PROJECT_ROOT / "resultados"
    saida = treinar(modelo, (X_tr, y_tr), (X_va, y_va), cfg, PROJECT_ROOT / "modelos", dir_res)

    fig = plotar_curvas(saida["historico"], nome_execucao, dir_res / "figuras" / "mlp" / f"{nome_execucao}_curvas.png")

    (dir_res / "configs").mkdir(parents=True, exist_ok=True)
    with open(dir_res / "configs" / f"{nome_execucao}.json", "w", encoding="utf-8") as f:
        json.dump({
            "arquitetura": nome,
            "camadas": MLP_CONFIGS[args.variante],
            "entrada": f"RGB {args.resolucao}x{args.resolucao} achatado ({X_tr.shape[1]} valores)",
            "classes": list(CLASSES),
            "pre_processamento": "RGB, resize bilinear a partir do original, /255, padronização por canal (stats do treino)",
            "normalizacao": norm.to_dict(),
            "aumento_de_dados": "nenhum",
            "perda": "CrossEntropyLoss",
            "config_treino": asdict(cfg),
            "resultado": saida["resumo"],
            "versoes": {
                "python": sys.version.split()[0], "torch": torch.__version__,
                "numpy": np.__version__, "scikit-learn": sklearn.__version__, "pillow": PIL.__version__,
            },
        }, f, indent=2, ensure_ascii=False)

    r = saida["resumo"]
    print(
        f"\nConcluído: melhor F1 macro val = {r['f1_macro_validacao']:.4f} "
        f"(acc {r['acuracia_validacao']:.4f}) na época {r['epoca_checkpoint']} "
        f"de {r['epocas_executadas']}; {r['tempo_total_s']}s no total."
    )
    print(f"Checkpoint: {saida['checkpoint'].relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Curvas:     {fig.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
