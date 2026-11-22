# Projeto 3 — Classificação de Cenas Aéreas com Redes Neurais

Investigação de como a arquitetura (MLP vs. CNN) e a resolução das imagens (`128 × 128` vs. `256 × 256`) afetam a classificação de cenas aéreas no dataset **NWPU-RESISC45**, avaliando desempenho e custo computacional.

## Classes selecionadas (ordem fixa)

1. `airport` — Aeroporto
2. `beach` — Praia
3. `circular_farmland` — Áreas agrícolas circulares
4. `dense_residential` — Área residencial densa
5. `desert` — Deserto
6. `forest` — Floresta
7. `harbor` — Porto
8. `mountain` — Montanha
9. `river` — Rio
10. `storage_tank` — Tanques de armazenamento

## Estrutura do repositório

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── dados/
│   ├── raw/              # Originais (não versionados) ou instruções de obtenção
│   └── splits/           # Partições (arquivo, classe e partição)
├── notebooks/            # Exploração, treino e avaliação (01 dados, 02 MLPs, 03 CNNs e YOLO)
├── src/                  # Scripts reutilizáveis (treino na GPU, avaliação na validação, gráficos)
├── modelos/              # Checkpoints (não versionados; ver docs/PROGRESSO.md)
├── resultados/
│   ├── experimentos.csv  # Uma linha por treino (MLPs, CNNs e YOLO)
│   ├── historicos/       # Métricas por época de cada treino
│   ├── configs/          # Configuração de cada treino
│   ├── matrizes_confusao/, metricas_por_classe/   # Avaliação na validação, por modelo
│   ├── previsoes_validacao_{mlp,cnn}.csv
│   ├── optuna/           # Busca de hiperparâmetros
│   ├── yolo/             # Bônus YOLO (execução do Ultralytics e avaliação)
│   └── figuras/          # Gráficos, curvas e matrizes de confusão
│       ├── mlp/  cnn/  yolo/
└── docs/
    ├── PROGRESSO.md      # Até onde fomos, o que falta e como retomar
    ├── MLP.md  CNN.md  YOLO.md   # Documentação de cada família de modelos
    └── (artigo, pitch e acompanhamento ainda não feitos)
```

## Instalação e ambiente

O projeto utiliza **PyTorch** tanto para experimentação local (Windows/macOS) quanto para a reprodução padronizada dos treinos finais em GPU no **Google Colab** (garantindo mesmo hardware e isolamento na comparação entre resoluções).

```bash
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

> **Nota (GPU NVIDIA no Windows/Linux):** o comando deve ser executado em cada máquina com GPU NVIDIA que criar seu próprio `.venv` (no **macOS** e no **Google Colab** não é necessário, pois `pip install -r requirements.txt` já atende o MacBook e o Colab já traz PyTorch com CUDA pré-instalado). Após ativar o `.venv` e **antes** de `pip install -r requirements.txt`, rode o comando indicado em [pytorch.org](https://pytorch.org/get-started/locally/) conforme a versão de CUDA do seu driver (`nvidia-smi`), por exemplo:
>
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
> ```

## Dados e obtenção

- Fonte: **NWPU-RESISC45** (filtrar exclusivamente as 10 classes listadas acima, totalizando 7.000 imagens RGB de `256 × 256` pixels).
- Colocar a pasta extraída `NWPU-RESISC45/` em `dados/raw/` (não versionado no Git).
- Para remover automaticamente as demais subpastas de `NWPU-RESISC45/` e manter apenas as 10 classes selecionadas, execute:

```bash
python src/setup_dataset.py
```

> **Atenção:** o script `src/setup_dataset.py` exclui permanentemente as subpastas diretas de `NWPU-RESISC45/` que não fazem parte das 10 classes listadas acima (procurando em `dados/raw/NWPU-RESISC45` ou no diretório atual).

## Duplicatas e partições

**Inventário** (`python src/check_integrity.py` → `resultados/inventario_dados.csv`): 7.000 imagens, todas legíveis, RGB (3 canais) e `256 × 256`. Foram encontrados **6 pares de duplicatas exatas** (mesmo hash MD5), todos na classe `airport`:

| Par | Partição |
|---|---|
| `airport_050` / `airport_420` | treino |
| `airport_064` / `airport_527` | treino |
| `airport_118` / `airport_315` | treino |
| `airport_361` / `airport_509` | treino |
| `airport_243` / `airport_397` | teste |
| `airport_289` / `airport_635` | validação |

**Tratamento:** nenhuma imagem foi excluída e os originais não foram alterados. As duplicatas são mantidas e **agrupadas na mesma partição**.

**Regra de partição** (`python src/create_splits.py` → `dados/splits/particoes.csv`):

1. Carregar o inventário e ordenar os caminhos antes do sorteio.
2. Agrupar as imagens por hash MD5, para que cópias idênticas fiquem juntas.
3. Dividir os **hashes** em `70% / 15% / 15%` com `train_test_split`, estratificado por classe e com semente `42`.
4. Atribuir a cada imagem a partição do seu hash.

Como a divisão é por hash, as contagens por classe não são exatamente `490 / 105 / 105`. Contagens finais (7.000 imagens):

| Partição | Imagens |
|---|---|
| treino | 4.899 |
| validação | 1.050 |
| teste | 1.051 |

Todas as classes têm `490 / 105 / 105`, exceto `airport`, que tem `489 / 105 / 106`. Nenhum par de duplicatas cruza partições, portanto não há vazamento por cópia exata. Esta mesma divisão é reutilizada em todos os experimentos e no bônus.

**Limitação:** o MD5 detecta apenas cópias idênticas byte a byte. Versões da mesma imagem recortadas, rotacionadas, redimensionadas ou recomprimidas teriam hashes diferentes e não são detectadas.

## Resultados até agora (validação, uma execução por modelo; o teste não foi usado)

| Família | Melhor modelo | F1 macro | Acurácia | Documento |
|---|---|---|---|---|
| YOLO (bônus, pré-treinado) | YOLO26n-cls, 256 × 256 | 0,9895 | 0,9895 | `docs/YOLO.md` |
| CNN (VGG16) | VGG16SE, 256 × 256, Optuna | 0,9645 | 0,9648 | `docs/CNN.md` |
| CNN (ResNet18) | ResNet18SE, 256 × 256, inicial | 0,9481 | 0,9486 | `docs/CNN.md` |
| MLP | MLP-B, 128 × 128, aumento leve, Optuna | 0,7000 | 0,7067 | `docs/MLP.md` |

MLPs, CNNs e YOLO foram treinados na **mesma GPU A100 do Colab**. O estado detalhado, o que falta (inclusive as variações de aumento de dados das CNNs, ainda parciais) e como retomar estão em **`docs/PROGRESSO.md`**.

## Ordem de execução

1. Baixar e extrair o dataset em `dados/raw/NWPU-RESISC45/` e filtrar as 10 classes com `python src/setup_dataset.py`.
2. Auditar integridade, canais RGB e duplicatas com `python src/check_integrity.py` (gera `resultados/inventario_dados.csv`).
3. Análise exploratória e geração das partições estratificadas (`70%` treino, `15%` validação, `15%` teste, semente `42`) em `dados/splits/`.
4. MLPs: `notebooks/02_mlp_preliminar.ipynb` (código e Optuna) e, para refazer os 12 treinos numa GPU e avaliar na validação, `python src/rodar_mlps_a100.py` (gera matrizes de confusão, métricas por classe e previsões).
5. CNNs: `notebooks/03_CNNs.ipynb` (8 arquiteturas com aumento leve, Optuna, bônus YOLO) e `python src/rodar_cnns_aumentos.py <128|256>` (receitas sem aumento e aumento forte; retomável). Depois `python src/avaliar_cnn_validacao.py` e `python src/avaliar_yolo_validacao.py`.
6. Gráficos: `python src/graficos_mlp.py`, `python src/graficos_cnn.py`, `python src/graficos_cnn_aumentos.py` e `python src/graficos_yolo.py`.
7. Avaliação no conjunto de teste (ainda não feita) e geração das métricas, tabelas e figuras em `resultados/`.
