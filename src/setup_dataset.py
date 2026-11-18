import os
import shutil

ALLOWED_CLASSES = {
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
}


def resolve_dataset_dir() -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(project_root, "dados", "raw", "NWPU-RESISC45"),
        os.path.join(os.getcwd(), "NWPU-RESISC45"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(
        "Pasta 'NWPU-RESISC45' não encontrada. Caminhos verificados: "
        + ", ".join(candidates)
    )


def filter_dataset_classes(dataset_dir: str) -> None:
    removed = []
    kept = []

    for entry in sorted(os.listdir(dataset_dir)):
        entry_path = os.path.join(dataset_dir, entry)
        if not os.path.isdir(entry_path):
            continue

        if entry in ALLOWED_CLASSES:
            kept.append(entry)
        else:
            shutil.rmtree(entry_path)
            removed.append(entry)

    missing = sorted(ALLOWED_CLASSES - set(kept))

    print(f"Diretório processado: {dataset_dir}")
    print(f"Subpastas mantidas ({len(kept)}): {', '.join(kept) if kept else 'nenhuma'}")
    print(f"Subpastas removidas ({len(removed)}): {len(removed)}")
    if missing:
        print(f"Aviso - classes esperadas ausentes ({len(missing)}): {', '.join(missing)}")


if __name__ == "__main__":
    dataset_path = resolve_dataset_dir()
    filter_dataset_classes(dataset_path)
