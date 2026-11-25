# Progresso do projeto: até onde fomos

Este documento registra o estado do projeto, o que falta e como retomar de onde paramos. Os detalhes de cada parte estão em `docs/MLP.md`, `docs/CNN.md` e `docs/YOLO.md`.

> Todos os números são da **validação**, de **uma execução por modelo**. O **teste não foi usado** em nenhum treino nem em nenhuma escolha.

---

## 1. O que está pronto

| Parte | Estado | Detalhes |
|---|---|---|
| Dados, partições e análise de duplicatas | pronto | 7.000 imagens, divisão 4.899 / 1.050 / 1.051 (semente 42). 6 pares de duplicatas exatas ficaram na mesma partição. Há 5 pares de **quase**-duplicatas entre partições (4 em `beach`, 1 em `mountain`), que o MD5 não detecta (`README.md`, `docs/CNN.md` seção 9) |
| **MLPs** (12 treinos) | pronto, **refeito na A100** | 2 arquiteturas × 3 receitas de aumento (nenhum, leve, forte) × 2 rodadas (hiperparâmetros fixos e do Optuna). Melhor: MLP-B, aumento leve, Optuna, **F1 0,700**. Com matrizes de confusão (`docs/MLP.md`) |
| **CNNs** (96 treinos) | **pronto** | 8 arquiteturas × 128 e 256 × inicial e Optuna × 3 receitas de aumento (leve, nenhum e forte). Melhor: **VGG16SE, 256 × 256, aumento forte, Optuna, F1 0,9695**. Com matrizes de confusão e métricas por classe dos 96 (`docs/CNN.md`) |
| **YOLO** (bônus, pré-treinado) | pronto | YOLO26n-cls em 256 × 256: **F1 0,9895**, o melhor de todos (`docs/YOLO.md`) |

Melhor modelo de cada família (validação):

| Família | Melhor modelo | F1 macro | Acurácia |
|---|---|---|---|
| YOLO (pré-treinado) | YOLO26n-cls, 256 | 0,9895 | 0,9895 |
| CNN (VGG16) | VGG16SE, 256, aumento forte, Optuna | 0,9695 | 0,9695 |
| CNN (ResNet18) | ResNet18, 256, aumento forte, inicial | 0,9514 | 0,9514 |
| MLP | MLP-B, 128, aumento leve, Optuna | 0,7000 | 0,7067 |

**Hardware:** MLPs, CNNs e YOLO foram treinados na **mesma GPU A100 (40 GB) do Google Colab**, como o PDF pede para a comparação de custo. As MLPs foram feitas primeiro no Mac (MPS) e refeitas na A100; os resultados do Mac foram substituídos (continuam no histórico do git) e a comparação está na seção 11 do `docs/MLP.md`.

---

## 2. Variações de aumento de dados das CNNs: concluídas

As CNNs foram treinadas com três receitas de aumento: **leve** (espelhamentos e rotações de 90°, o do notebook), **sem aumento** e **forte** (leve mais recorte com zoom, cor e apagamento). Cada receita com hiperparâmetros iniciais e do Optuna (a busca do Optuna foi feita com aumento leve e não foi refeita), em 128 e 256. Os 64 treinos novos rodaram por `src/rodar_cnns_aumentos.py` em cerca de 6,2 horas de A100.

Resultados (detalhes e as tabelas de cada bloco de 8 treinos em `docs/CNN.md`, seção 13; média de 8 modelos por célula):

| Resolução | Rodada | Sem aumento | Aumento leve | Aumento forte |
|---|---|---|---|---|
| 128 | inicial | 0,8841 | 0,9273 | 0,9171 |
| 128 | Optuna | 0,8756 | 0,9346 | 0,9195 |
| 256 | inicial | 0,9214 | 0,9444 | 0,9413 |
| 256 | Optuna | 0,9147 | 0,9465 | 0,9399 |

- **Sem aumento é sempre pior que o leve** (32 de 32 pares; média de −3,9 pontos). O aumento forte ficou, na média, 0,9 ponto abaixo do leve (piorou em 23 pares e melhorou em 9), mas o **melhor treino de todos usa o forte** (VGG16SE em 256 com Optuna, 0,9695, contra 0,9645 do melhor com aumento leve). Com um treino por modelo, essa diferença de 0,5 ponto é ruído.
- Os hiperparâmetros do Optuna (achados com aumento leve) **não se transferem** para sem aumento (pioram em média) nem de forma clara para o aumento forte.
- Os 10 melhores treinos são todos VGGs em 256.
- Os 5 treinos sem aumento em 128 cujos artefatos tinham se perdido numa primeira tentativa foram refeitos, e reproduziram os checkpoints e F1 anteriores (`docs/CNN.md`, seção 10).
- Os resultados de **todos** os 96 treinos têm linha em `resultados/experimentos.csv`, histórico, configuração, curvas, matriz de confusão, métricas por classe e checkpoint (`modelos/`, não versionado). A avaliação final por classe foi refeita na GPU para os 96, e as métricas reproduzem as do treino.

---

## 3. Outras pendências do projeto

- **Escolher as seis execuções finais** do PDF (2 MLP e 2 CNN em 128, as mesmas 2 CNN em 256), treinar **do zero** e **avaliar no teste, uma única vez**. Foram treinadas 8 CNNs por resolução e receita; falta decidir qual é a CNN A e qual é a CNN B, com a justificativa (hipótese de comparação, parâmetros). Sugestão para discussão do grupo, não decidida: uma ResNet18 e uma VGG16, para comparar as duas famílias (as melhores variantes são a ResNet18 em 256 com aumento forte, 0,9514, e a VGG16SE em 256 com aumento forte e Optuna, 0,9695). Também falta decidir **qual receita de aumento e quais hiperparâmetros** entram nos treinos finais: os melhores resultados usam aumento forte ou leve em 256, e os valores do Optuna só foram buscados com aumento leve. A escolha deve ser feita com a validação, sem olhar o teste.
- **Artigo** (`docs/artigo.pdf`), **slides do pitch** (`docs/pitch.pdf`) e **slides do acompanhamento** (`docs/acompanhamento.pdf`). Nada disso foi feito.
- **Armazenamento externo dos modelos** com link de download (o PDF permite). Os checkpoints (4,7 GB dos 96 treinos de CNN e 1,2 GB das MLPs) estão só na pasta local `modelos/`, que não vai para o git. O `best.pt` do YOLO (3 MB) está no repositório.
- **Corrigir as quase-duplicatas** entre partições, se o grupo quiser: exigiria refazer as partições e retreinar tudo.
- **Melhorar a busca do Optuna** (hoje a melhor tentativa da busca fica abaixo dos hiperparâmetros iniciais): enfileirar os valores iniciais, dar às tentativas o mesmo orçamento de épocas e usar mais tentativas (`docs/CNN.md` seção 7).
- Inspeção visual dos erros do YOLO, e do modelo de maior F1 de validação, para o texto de análise de erros.

---

## 4. Como retomar

Pré-requisitos: conta Colab com A100 (saldo de compute units), o dataset RESISC45 e o CLI do Colab (`uv tool install google-colab-cli`, depois `colab usage` para fazer o login).

1. **Criar a VM:** `colab new -s nome --gpu A100`. O Colab às vezes responde `Service Unavailable` por falta de A100 livre; é só insistir.
2. **Enviar o código e preparar o ambiente na VM:** empacotar o projeto sem `dados/raw`, `modelos` e `.git` (`tar czf projeto.tgz ...`), enviar com `colab upload` e, na VM, extrair em `/content/projeto`, instalar `kagglehub optuna ultralytics`, baixar o dataset (`kagglehub.dataset_download("aqibrehmanpirzada/nwpuresisc45")`), juntar as pastas `train/train` e `test/test` em `dados/raw/NWPU-RESISC45/<classe>/` apenas para as 10 classes e conferir que as 7.000 linhas de `dados/splits/particoes.csv` apontam para arquivos existentes.
3. **Treinar:** `python src/rodar_cnns_aumentos.py 128 /content/projeto` e depois `... 256 ...`. O script é retomável: pula os treinos já concluídos (com histórico, linha em `experimentos.csv` e checkpoint) e refaz os demais.
4. **Avaliar na validação:** `python src/avaliar_cnn_validacao.py` (matriz de confusão, métricas por classe e previsões). Durante um treino de 256, rode com `AVALIAR_DISPOSITIVO=cpu`, porque o treino ocupa quase toda a VRAM; ao fim, use `AVALIAR_FORCAR=1` na GPU livre.
5. **Gráficos:** `python src/graficos_cnn.py` e `python src/graficos_cnn_aumentos.py` (CNNs), `python src/graficos_mlp.py` (MLPs) e `python src/graficos_yolo.py` (YOLO).
6. **Desligar a VM:** `colab stop -s nome`. Uma VM parada não consome compute units.

**Lições da execução** (para não repetir os problemas):

- **Baixar os resultados aos poucos, desde antes de o primeiro treino terminar.** Tudo em `/content` se perde se a VM for recolhida. Foi o que aconteceu na primeira execução, que teve de ser refeita inteira, e de novo, em parte (5 treinos), com as variações de aumento.
- **Manter o computador acordado:** quando o computador local entra em suspensão, o Colab recolhe a VM por inatividade.
- **Manter o kernel da VM ocupado** (rodar o treino dentro de um `colab exec` em segundo plano) ajuda a VM a não ser considerada ociosa.
- **O Colab aceita só uma sessão `colab ssh` por vez** (erro 429 se houver outra).
- O `notebooks/03_CNNs.ipynb` reescreve `resultados/experimentos.csv` com pandas e trunca dígitos de ponto flutuante das outras linhas; por isso o arquivo foi sempre fundido linha a linha, sem sobrescrever as existentes.

---

## 5. Custo até aqui

Saldo de compute units: **375,06** no início, **250,57** agora (cerca de 124,5 unidades gastas): a primeira execução das CNNs, que se perdeu, custou cerca de 36; a reexecução das CNNs, cerca de 32; o YOLO, cerca de 1; e as MLPs mais as 64 variações de aumento (incluindo a refação de 5 treinos), cerca de 56 no total (302,96 para 250,57 só nas variações de aumento: cerca de 52). A A100 gasta 6,73 unidades por hora quando ocupada. A VM do Colab está **parada** (não há sessão ativa).
