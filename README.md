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
├── notebooks/            # Exploração, treino e avaliação
├── src/                  # Funções reutilizáveis
├── modelos/              # Checkpoints ou instruções de download
├── resultados/
│   ├── experimentos.csv
│   ├── previsoes_teste.csv
│   └── figuras/          # Gráficos, curvas e matrizes de confusão
└── docs/                 # Documentos do projeto (acompanhamento, artigo, pitch)
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

## Ordem de execução

1. Baixar e extrair o dataset em `dados/raw/NWPU-RESISC45/` e filtrar as 10 classes com `python src/setup_dataset.py`.
2. Auditar integridade, canais RGB e duplicatas com `python src/check_integrity.py` (gera `resultados/inventario_dados.csv`).
3. Análise exploratória e geração das partições estratificadas (`70%` treino, `15%` validação, `15%` teste, semente `42`) em `dados/splits/`.
4. MLP preliminar (validação apenas): `python src/treinar_mlp.py --variante A` ou `notebooks/02_mlp_preliminar.ipynb`. Gera `modelos/MLP-A-128_preliminar.pt`, `resultados/historicos/`, `resultados/configs/`, `resultados/figuras/MLP-A-128_preliminar_curvas.png` e uma linha em `resultados/experimentos.csv`. Use `--variante B` para a MLP-B e `--tipo final` para as execuções finais.
5. Experimentos preliminares locais e treinamento padronizado dos 6 modelos obrigatórios (`MLP-A-128`, `MLP-B-128`, `CNN-A-128`, `CNN-B-128`, `CNN-A-256`, `CNN-B-256`) no Google Colab, salvando checkpoints em `modelos/` e históricos.
6. Avaliação no conjunto de teste e geração das métricas, tabelas e figuras em `resultados/`.
