# Progresso do projeto: até onde fomos

Este documento registra o estado do projeto, o que falta e como retomar de onde paramos. Os detalhes de cada parte estão em `docs/MLP.md`, `docs/CNN.md` e `docs/YOLO.md`.

> Todos os números são da **validação**, de **uma execução por modelo**. O **teste não foi usado** em nenhum treino nem em nenhuma escolha.

---

## 1. O que está pronto

| Parte | Estado | Detalhes |
|---|---|---|
| Dados, partições e análise de duplicatas | pronto | 7.000 imagens, divisão 4.899 / 1.050 / 1.051 (semente 42). 6 pares de duplicatas exatas ficaram na mesma partição. Há 5 pares de **quase**-duplicatas entre partições (4 em `beach`, 1 em `mountain`), que o MD5 não detecta (`README.md`, `docs/CNN.md` seção 9) |
| **MLPs** (12 treinos) | pronto, **refeito na A100** | 2 arquiteturas × 3 receitas de aumento (nenhum, leve, forte) × 2 rodadas (hiperparâmetros fixos e do Optuna). Melhor: MLP-B, aumento leve, Optuna, **F1 0,700**. Com matrizes de confusão (`docs/MLP.md`) |
| **CNNs com aumento leve** (32 treinos) | pronto | 8 arquiteturas × 128 e 256 × inicial e Optuna. Melhor: **VGG16SE, 256 × 256, Optuna, F1 0,9645**. Com matrizes de confusão e métricas por classe (`docs/CNN.md`) |
| **YOLO** (bônus, pré-treinado) | pronto | YOLO26n-cls em 256 × 256: **F1 0,9895**, o melhor de todos (`docs/YOLO.md`) |
| **CNNs com outras receitas de aumento** | **parcial** | ver seção 2 |

Melhor modelo de cada família (validação):

| Família | Melhor modelo | F1 macro | Acurácia |
|---|---|---|---|
| YOLO (pré-treinado) | YOLO26n-cls, 256 | 0,9895 | 0,9895 |
| CNN (VGG16) | VGG16SE, 256, Optuna | 0,9645 | 0,9648 |
| CNN (ResNet18) | ResNet18SE, 256, inicial | 0,9481 | 0,9486 |
| MLP | MLP-B, 128, aumento leve, Optuna | 0,7000 | 0,7067 |

**Hardware:** MLPs, CNNs e YOLO foram treinados na **mesma GPU A100 (40 GB) do Google Colab**, como o PDF pede para a comparação de custo. As MLPs foram feitas primeiro no Mac (MPS) e refeitas na A100; os resultados do Mac foram substituídos (continuam no histórico do git) e a comparação está na seção 11 do `docs/MLP.md`.

---

## 2. Variações de aumento de dados das CNNs: onde paramos

Até aqui as CNNs foram treinadas só com o aumento **leve** (espelhamentos e rotações de 90°). Para fechar a comparação (como nas MLPs), faltam as receitas **sem aumento** e **aumento forte**, cada uma com hiperparâmetros iniciais e do Optuna (a busca do Optuna foi feita com aumento leve e não é refeita), em 128 e 256. São **64 treinos** (8 CNNs × 2 receitas × 2 rodadas × 2 resoluções), executados por `src/rodar_cnns_aumentos.py`, na ordem: sem aumento inicial, aumento forte inicial, sem aumento Optuna, aumento forte Optuna, primeiro em 128 e depois em 256.

**Feito até a interrupção (128 × 128, hiperparâmetros iniciais, sem aumento): 8 treinos concluídos**

| Modelo | F1 sem aumento | F1 aumento leve | Diferença | Acurácia | Checkpoint | Tempo | s/época |
|---|---|---|---|---|---|---|---|
| ResNet18 | 0,8209 | 0,9132 | −9,2 pts | 0,8238 | 12/22 | 28 s | 1,25 |
| ResNet18SE | 0,8481 | 0,8766 | −2,9 pts | 0,8486 | 9/19 | 28 s | 1,48 |
| ResNet18CBAM | 0,8753 | 0,9245 | −4,9 pts | 0,8762 | 22/32 | 68 s | 2,14 |
| ResNet18SECBAM | 0,8695 | 0,9265 | −5,7 pts | 0,8724 | 14/24 | 61 s | 2,54 |
| VGG16 | 0,9369 | 0,9488 | −1,2 pts | 0,9371 | 37/47 | 177 s | 3,76 |
| VGG16SE | 0,9201 | 0,9377 | −1,8 pts | 0,9210 | 25/35 | 147 s | 4,19 |
| VGG16CBAM | 0,9014 | 0,9444 | −4,3 pts | 0,9029 | 18/28 | 156 s | 5,57 |
| VGG16SECBAM | 0,9005 | 0,9464 | −4,6 pts | 0,9029 | 15/25 | 150 s | 6,00 |

- **Sem aumento, as 8 CNNs perdem em média 4,3 pontos de F1** em relação ao aumento leve (de −1,2 a −9,2), então o aumento leve ajuda as CNNs, como ajudou as MLPs. As ResNets sofrem mais (média de −5,7) que as VGGs (média de −3,0).
- Esses números são os registrados no próprio log do treino (linha de cada modelo no `resultados/experimentos.csv`).

**Atenção: os artefatos de 5 desses 8 treinos se perderam.** Eu liguei o backup incremental depois de os primeiros treinos terem começado, e a VM foi desligada antes de eu recuperar tudo:

| Treinos | O que existe no repositório | O que se perdeu |
|---|---|---|
| VGG16, VGG16SE, VGG16CBAM (128, sem aumento, inicial) | **tudo**: linha de resultados, histórico por época, config, matriz de confusão, métricas por classe, curvas e checkpoint (`modelos/`, não versionado) | nada |
| ResNet18, ResNet18SE, ResNet18CBAM, ResNet18SECBAM (idem) | linha de resultados, matriz de confusão e métricas por classe | checkpoint, histórico por época, config e curvas |
| VGG16SECBAM (idem) | só a linha de resultados | tudo o mais, inclusive matriz e métricas por classe |

Reexecutar o script refaz esses 5 treinos automaticamente (ele só pula treinos que tenham histórico, linha de resultados **e** checkpoint).

**Interrompido:** a próxima rodada (aumento forte, hiperparâmetros iniciais, 128) tinha começado pela ResNet18 e foi encerrada, sem nenhum treino concluído.

**O que falta (56 treinos mais a refação de 5):**

| Bloco | Treinos |
|---|---|
| 128: sem aumento, inicial (refazer 5 com artefatos) | 5 |
| 128: aumento forte, inicial | 8 |
| 128: sem aumento, Optuna | 8 |
| 128: aumento forte, Optuna | 8 |
| 256: as 4 rodadas × 2 receitas | 32 |

Estimativa pelos tempos já medidos: **cerca de 7 horas de A100 e uns 50 compute units** (o 256 pesa mais, por volta de 5 horas). Saldo em 302,96 unidades.

---

## 3. Outras pendências do projeto

- **Escolher as seis execuções finais** do PDF (2 MLP e 2 CNN em 128, as mesmas 2 CNN em 256) e **avaliar no teste, uma única vez**. O notebook treina 8 CNNs por resolução; falta decidir qual é a CNN A e qual é a CNN B de cada família, com a justificativa (hipótese de comparação, parâmetros).
- **Artigo** (`docs/artigo.pdf`), **slides do pitch** (`docs/pitch.pdf`) e **slides do acompanhamento** (`docs/acompanhamento.pdf`). Nada disso foi feito.
- **Armazenamento externo dos modelos** com link de download (o PDF permite). Os checkpoints (1,6 GB das CNNs e 1,2 GB das MLPs) estão só na pasta local `modelos/`, que não vai para o git. O `best.pt` do YOLO (3 MB) está no repositório.
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

- **Baixar os resultados aos poucos, desde antes de o primeiro treino terminar.** Tudo em `/content` se perde se a VM for recolhida. Foi o que aconteceu na primeira execução, que teve de ser refeita inteira, e de novo, em parte, com as variações de aumento.
- **Manter o computador acordado:** quando o computador local entra em suspensão, o Colab recolhe a VM por inatividade.
- **Manter o kernel da VM ocupado** (rodar o treino dentro de um `colab exec` em segundo plano) ajuda a VM a não ser considerada ociosa.
- **O Colab aceita só uma sessão `colab ssh` por vez** (erro 429 se houver outra).
- O `notebooks/03_CNNs.ipynb` reescreve `resultados/experimentos.csv` com pandas e trunca dígitos de ponto flutuante das outras linhas; por isso o arquivo foi sempre fundido linha a linha, sem sobrescrever as existentes.

---

## 5. Custo até aqui

Saldo de compute units: **375,06** no início, **302,96** agora (cerca de 72 unidades gastas): a primeira execução das CNNs, que se perdeu, custou cerca de 36; a reexecução das CNNs, cerca de 32; o YOLO, cerca de 1; e as MLPs mais as variações de aumento até a interrupção (mesma VM), cerca de 4. A A100 gasta 6,73 unidades por hora.
