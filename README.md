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

```bash
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

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
2. Análise exploratória e geração das partições estratificadas (`70%` treino, `15%` validação, `15%` teste, semente `42`) em `dados/splits/`.
3. Treinamento dos 6 modelos obrigatórios (`MLP-A-128`, `MLP-B-128`, `CNN-A-128`, `CNN-B-128`, `CNN-A-256`, `CNN-B-256`) salvando checkpoints em `modelos/` e históricos.
4. Avaliação no conjunto de teste e geração das métricas, tabelas e figuras em `resultados/`.
