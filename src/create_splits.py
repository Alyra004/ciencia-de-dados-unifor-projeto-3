from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

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

PARTITION_ORDER: tuple[str, ...] = ("treino", "validacao", "teste")
RANDOM_STATE: int = 42


def load_valid_inventory(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Arquivo de inventário não encontrado: {csv_path}")

    df = pd.read_csv(csv_path)
    df_valid = df.loc[df["status_leitura"] == "ok"].copy()
    df_valid = df_valid.dropna(subset=["caminho_relativo", "classe", "hash_md5"])
    df_valid = df_valid.sort_values("caminho_relativo", ascending=True).reset_index(drop=True)

    if df_valid.empty:
        raise ValueError("Nenhuma imagem válida (status_leitura == 'ok') encontrada no inventário.")

    return df_valid


def build_hash_table(df_valid: pd.DataFrame) -> pd.DataFrame:
    hashes_df = (
        df_valid.groupby("hash_md5", as_index=False, sort=False)
        .agg(classe=("classe", "first"), n_classes=("classe", "nunique"))
    )

    cross_class = hashes_df.loc[hashes_df["n_classes"] > 1]
    if not cross_class.empty:
        print(
            f"Aviso: {len(cross_class)} hash(es) aparecem em mais de uma classe; "
            "utilizando a primeira ocorrência ordenada para estratificação."
        )

    return hashes_df[["hash_md5", "classe"]].copy()


def assign_partitions(
    df_valid: pd.DataFrame,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Realiza a divisão estratificada 70/15/15 no nível de hash e mapeia para as imagens."""
    hashes_df = build_hash_table(df_valid)

    train_hashes, temp_hashes = train_test_split(
        hashes_df,
        test_size=0.30,
        stratify=hashes_df["classe"],
        random_state=random_state,
    )
    val_hashes, test_hashes = train_test_split(
        temp_hashes,
        test_size=0.50,
        stratify=temp_hashes["classe"],
        random_state=random_state,
    )

    hash_to_partition: dict[str, str] = {}
    for h in train_hashes["hash_md5"]:
        hash_to_partition[h] = "treino"
    for h in val_hashes["hash_md5"]:
        hash_to_partition[h] = "validacao"
    for h in test_hashes["hash_md5"]:
        hash_to_partition[h] = "teste"

    df_splits = df_valid.copy()
    df_splits["particao"] = df_splits["hash_md5"].map(hash_to_partition)

    return df_splits[["caminho_relativo", "classe", "particao"]].copy()


def print_split_report(df_splits: pd.DataFrame) -> None:
    total = len(df_splits)
    counts = df_splits["particao"].value_counts().reindex(PARTITION_ORDER, fill_value=0)

    print("=== Resumo do Particionamento (70% / 15% / 15%) ===")
    print(f"Total de imagens válidas: {total}")
    print("\nContagem total por partição:")
    for part in PARTITION_ORDER:
        qtd = int(counts[part])
        pct = (qtd / total * 100.0) if total else 0.0
        print(f"  - {part:<10}: {qtd:>5} ({pct:5.2f}%)")

    table = pd.crosstab(df_splits["classe"], df_splits["particao"])
    table = table.reindex(columns=list(PARTITION_ORDER), fill_value=0)

    ordered_classes = [c for c in EXPECTED_CLASSES if c in table.index]
    extra_classes = sorted(c for c in table.index if c not in EXPECTED_CLASSES)
    table = table.reindex(index=ordered_classes + extra_classes, fill_value=0)
    table["total"] = table.sum(axis=1)

    print("\nContagem por classe e partição:")
    print(table.to_string())


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    inventory_csv = project_root / "resultados" / "inventario_dados.csv"
    output_csv = project_root / "dados" / "splits" / "particoes.csv"

    df_valid = load_valid_inventory(inventory_csv)
    df_splits = assign_partitions(df_valid, random_state=RANDOM_STATE)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_splits.to_csv(output_csv, index=False, encoding="utf-8")

    print_split_report(df_splits)
    print(f"\nPartições salvas em: {output_csv.relative_to(project_root).as_posix()}")


if __name__ == "__main__":
    main()
