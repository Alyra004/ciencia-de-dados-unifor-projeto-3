"""Carregamento e pré-processamento das imagens a partir de dados/splits/particoes.csv.

Decisões (registrar no artigo):
- Cada imagem é lida do arquivo ORIGINAL, convertida para RGB e redimensionada
  diretamente para a resolução-alvo (sem redimensionar uma versão já reduzida).
- Interpolação: bilinear com antialias (PIL.Image.BILINEAR).
- Normalização: pixels / 255 -> [0, 1]; em seguida padronização por canal
  (x - média) / desvio, com média e desvio calculados SOMENTE no treino e
  reaplicados à validação e ao teste (evita vazamento).
- Para a MLP, o tensor (3, H, W) é achatado em um vetor de 3*H*W posições
  (128 x 128 x 3 = 49.152 entradas).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

CLASSES: tuple[str, ...] = (
    "airport",
    "beach",
    "circular_farmland",
    "dense_residential",
    "desert",
    "forest",
    "harbor",
    "mountain",
    "river",
    "storage_tank",
)
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_CSV = PROJECT_ROOT / "dados" / "splits" / "particoes.csv"
CACHE_DIR = PROJECT_ROOT / "dados" / "cache"


def carregar_particoes(csv_path: Path = SPLITS_CSV) -> pd.DataFrame:
    """Lê o CSV de partições e devolve um DataFrame ordenado por caminho."""
    df = pd.read_csv(csv_path)
    desconhecidas = set(df["classe"]) - set(CLASSES)
    if desconhecidas:
        raise ValueError(f"Classes inesperadas no CSV de partições: {desconhecidas}")
    df["rotulo"] = df["classe"].map(CLASS_TO_IDX).astype(np.int64)
    return df.sort_values("caminho_relativo").reset_index(drop=True)


def ler_imagem(caminho: Path, resolucao: int) -> np.ndarray:
    """Lê uma imagem original, converte para RGB e redimensiona -> uint8 (H, W, 3)."""
    with Image.open(caminho) as img:
        img = img.convert("RGB")
        if img.size != (resolucao, resolucao):
            img = img.resize((resolucao, resolucao), Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8)


def carregar_particao(
    df: pd.DataFrame,
    particao: str,
    resolucao: int,
    usar_cache: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Carrega todas as imagens de uma partição em memória.

    Retorna (X uint8 [N, H, W, 3], y int64 [N], caminhos relativos).
    Usa um cache .npz em dados/cache/ para não reler os JPGs a cada execução.
    """
    sub = df.loc[df["particao"] == particao].reset_index(drop=True)
    caminhos = sub["caminho_relativo"].tolist()
    y = sub["rotulo"].to_numpy(dtype=np.int64)

    cache_file = CACHE_DIR / f"{particao}_{resolucao}.npz"
    if usar_cache and cache_file.is_file():
        dados = np.load(cache_file, allow_pickle=False)
        if dados["caminhos"].tolist() == caminhos:
            return dados["X"], y, caminhos

    X = np.stack([ler_imagem(PROJECT_ROOT / c, resolucao) for c in caminhos])

    if usar_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(cache_file, X=X, caminhos=np.array(caminhos))
    return X, y, caminhos


@dataclass
class Normalizador:
    """Padronização por canal com estatísticas aprendidas apenas no treino."""

    media: np.ndarray  # shape (3,), em escala [0, 1]
    desvio: np.ndarray  # shape (3,)

    @classmethod
    def ajustar(cls, X_treino_uint8: np.ndarray) -> "Normalizador":
        # Acumula soma e soma dos quadrados em blocos (evita alocar o treino inteiro em float64).
        soma = np.zeros(3, dtype=np.float64)
        soma_q = np.zeros(3, dtype=np.float64)
        n = 0
        for i in range(0, len(X_treino_uint8), 256):
            bloco = X_treino_uint8[i : i + 256].reshape(-1, 3).astype(np.float64) / 255.0
            soma += bloco.sum(axis=0)
            soma_q += (bloco**2).sum(axis=0)
            n += bloco.shape[0]
        media = soma / n
        desvio = np.sqrt(soma_q / n - media**2)
        return cls(media=media.astype(np.float32), desvio=desvio.astype(np.float32))

    def transformar(self, X_uint8: np.ndarray) -> np.ndarray:
        """uint8 (N, H, W, 3) -> float32 (N, 3, H, W) normalizado."""
        X = X_uint8.astype(np.float32) / 255.0
        X = (X - self.media) / self.desvio
        return np.ascontiguousarray(X.transpose(0, 3, 1, 2))

    def to_dict(self) -> dict[str, list[float]]:
        return {"media": self.media.tolist(), "desvio": self.desvio.tolist()}


def achatar(X_chw: np.ndarray) -> np.ndarray:
    """(N, 3, H, W) -> (N, 3*H*W): entrada vetorial para a MLP."""
    return X_chw.reshape(X_chw.shape[0], -1)


def preparar_dados_mlp(
    resolucao: int = 128,
    particoes: tuple[str, ...] = ("treino", "validacao"),
    usar_cache: bool = True,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, list[str]]], Normalizador]:
    """Pipeline completo da MLP: lê -> normaliza (stats do treino) -> achata.

    Por padrão NÃO carrega o teste: na fase preliminar só usamos validação.
    """
    df = carregar_particoes()
    X_tr, y_tr, c_tr = carregar_particao(df, "treino", resolucao, usar_cache)
    norm = Normalizador.ajustar(X_tr)

    saida: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}
    for p in particoes:
        if p == "treino":
            X, y, c = X_tr, y_tr, c_tr
        else:
            X, y, c = carregar_particao(df, p, resolucao, usar_cache)
        saida[p] = (achatar(norm.transformar(X)), y, c)
    return saida, norm
