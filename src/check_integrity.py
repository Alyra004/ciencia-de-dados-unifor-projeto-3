from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, UnidentifiedImageError

EXPECTED_CLASSES: tuple[str, ...] = (
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

IGNORED_FILES: set[str] = {".gitkeep", ".DS_Store", "Thumbs.db", "README.md"}


def resolve_dataset_root(raw_dir: Path) -> Path:
    """Resolve o diretório raiz contendo as pastas de classes."""
    nested_dir = raw_dir / "NWPU-RESISC45"
    if nested_dir.is_dir():
        return nested_dir
    return raw_dir


def compute_md5(file_path: Path, chunk_size: int = 8192) -> str:
    hasher = hashlib.md5()
    with file_path.open("rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def inspect_image(
    file_path: Path,
    project_root: Path,
    class_name: str,
) -> dict[str, Any]:
    rel_path = file_path.relative_to(project_root).as_posix()
    record: dict[str, Any] = {
        "caminho_relativo": rel_path,
        "classe": class_name,
        "status_leitura": "erro",
        "dimensao": None,
        "canais": None,
        "hash_md5": None,
    }

    try:
        record["hash_md5"] = compute_md5(file_path)
        with Image.open(file_path) as img:
            img.load()
            altura, largura = img.height, img.width
            num_canais = len(img.getbands())
            is_rgb = img.mode == "RGB" and num_canais == 3

            record["dimensao"] = f"{altura}x{largura}"
            record["canais"] = num_canais
            record["status_leitura"] = "ok" if is_rgb else f"canais_invalidos ({img.mode})"
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        record["status_leitura"] = f"erro ({type(exc).__name__})"

    return record


def scan_dataset(raw_dir: Path, project_root: Path) -> pd.DataFrame:
    dataset_root = resolve_dataset_root(raw_dir)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Diretório não encontrado: {dataset_root}")

    records: list[dict[str, Any]] = []
    class_dirs = sorted(
        (d for d in dataset_root.iterdir() if d.is_dir()),
        key=lambda d: (
            EXPECTED_CLASSES.index(d.name) if d.name in EXPECTED_CLASSES else len(EXPECTED_CLASSES),
            d.name,
        ),
    )

    for class_dir in class_dirs:
        files = sorted(
            f for f in class_dir.rglob("*")
            if f.is_file() and f.name not in IGNORED_FILES
        )
        for file_path in files:
            records.append(inspect_image(file_path, project_root, class_dir.name))

    columns = [
        "caminho_relativo",
        "classe",
        "status_leitura",
        "dimensao",
        "canais",
        "hash_md5",
    ]
    return pd.DataFrame(records, columns=columns)


def print_summary(df: pd.DataFrame) -> None:
    print("=== Resumo do Inventário de Dados ===")
    print(f"Total de arquivos analisados: {len(df)}")

    print("\nTotal de imagens por classe:")
    if df.empty:
        print("  Nenhum arquivo encontrado.")
    else:
        counts = df["classe"].value_counts()
        ordered_classes = [c for c in EXPECTED_CLASSES if c in counts.index]
        extra_classes = sorted(c for c in counts.index if c not in EXPECTED_CLASSES)
        for cls in ordered_classes + extra_classes:
            print(f"  - {cls}: {counts[cls]}")

    failures = df[df["status_leitura"] != "ok"] if not df.empty else df
    print(f"\nFalhas de leitura / canais inválidos: {len(failures)}")
    if not failures.empty:
        for _, row in failures.iterrows():
            print(f"  - {row['caminho_relativo']} [{row['status_leitura']}]")

    valid_hashes = df.dropna(subset=["hash_md5"]) if not df.empty else df
    duplicates = (
        valid_hashes[valid_hashes.duplicated(subset=["hash_md5"], keep=False)]
        if not valid_hashes.empty
        else valid_hashes
    )
    print(f"\nArquivos identificados como duplicatas (hashes repetidos): {len(duplicates)}")
    if not duplicates.empty:
        for md5_hash, group in duplicates.groupby("hash_md5", sort=False):
            print(f"  Hash {md5_hash}:")
            for path in group["caminho_relativo"]:
                print(f"    - {path}")


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "dados" / "raw"
    output_csv = project_root / "resultados" / "inventario_dados.csv"

    df = scan_dataset(raw_dir=raw_dir, project_root=project_root)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")

    print_summary(df)
    print(f"\nInventário salvo em: {output_csv.relative_to(project_root).as_posix()}")


if __name__ == "__main__":
    main()
