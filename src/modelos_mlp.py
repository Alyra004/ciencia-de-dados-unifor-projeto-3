"""Arquiteturas MLP (Experimento 1).

MLP-A: rasa e larga  -> 2 camadas ocultas (512, 128).
MLP-B: mais profunda -> 4 camadas ocultas (512, 256, 128, 64)  [a cargo da Pessoa responsável pela MLP-B].

Hipótese da comparação: com pixels achatados a MLP perde a estrutura espacial;
mais profundidade pode compor padrões de cor/textura mais abstratos, mas
aumenta o risco de sobreajuste com apenas 4.900 imagens de treino. A maior parte
dos parâmetros está na 1ª camada (49.152 x 512), então A e B têm custo parecido
e a diferença isola o efeito da PROFUNDIDADE.

Bloco oculto: Linear -> BatchNorm1d -> ReLU -> Dropout.
- BatchNorm estabiliza o treino com entradas de alta dimensão.
- Dropout reduz sobreajuste (a MLP memoriza pixels com facilidade).
"""

from __future__ import annotations

import torch
from torch import nn

MLP_CONFIGS: dict[str, dict] = {
    "A": {"ocultas": [512, 128], "dropout": 0.5},
    "B": {"ocultas": [512, 256, 128, 64], "dropout": 0.5},
}


class MLP(nn.Module):
    def __init__(
        self,
        n_entradas: int,
        ocultas: list[int],
        n_classes: int = 10,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        camadas: list[nn.Module] = []
        anterior = n_entradas
        for largura in ocultas:
            camadas += [
                nn.Linear(anterior, largura),
                nn.BatchNorm1d(largura),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            anterior = largura
        camadas.append(nn.Linear(anterior, n_classes))  # logits (softmax na perda)
        self.rede = nn.Sequential(*camadas)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rede(x.flatten(1))


def criar_mlp(variante: str, resolucao: int = 128, n_classes: int = 10) -> MLP:
    cfg = MLP_CONFIGS[variante]
    return MLP(
        n_entradas=3 * resolucao * resolucao,
        ocultas=cfg["ocultas"],
        n_classes=n_classes,
        dropout=cfg["dropout"],
    )


def contar_parametros(modelo: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in modelo.parameters())
    treinaveis = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    return total, treinaveis


def resumo(modelo: nn.Module, n_entradas: int) -> str:
    """Resumo camada a camada (formato de tabela simples)."""
    linhas = [f"{'Camada':<28}{'Saída':<16}{'Parâmetros':>14}", "-" * 58]
    forma = n_entradas
    for nome, m in modelo.named_modules():
        if isinstance(m, nn.Sequential) or m is modelo:
            continue
        n = sum(p.numel() for p in m.parameters(recurse=False))
        if isinstance(m, nn.Linear):
            forma = m.out_features
        linhas.append(f"{nome + ' (' + type(m).__name__ + ')':<28}{'(N, ' + str(forma) + ')':<16}{n:>14,}")
    total, trein = contar_parametros(modelo)
    linhas += ["-" * 58, f"Total de parâmetros: {total:,}  (treináveis: {trein:,})"]
    return "\n".join(linhas)
