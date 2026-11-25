# Bônus: YOLO em modo classify

Este documento resume o ajuste fino de um **YOLO de classificação** (Ultralytics) com **pesos pré-treinados**, nas dez classes e **nas mesmas partições** das MLPs e CNNs. O código está em `notebooks/03_CNNs.ipynb` (seção 7) e `src/`, e os números ficam em `resultados/yolo/`.

> **Todos os números são da validação e de uma única execução.** O **teste não foi usado**. O pré-treinamento é permitido **apenas no bônus**: as CNNs e MLPs do projeto partem de pesos novos.

---

## 1. Configuração

| Item | Valor |
|---|---|
| Modelo e pesos | `yolo26n-cls.pt`, pré-treinado no ImageNet (baixado pelo Ultralytics) |
| Biblioteca | `ultralytics` 8.4.216, PyTorch 2.14.1 + CUDA 13.0 |
| Hardware | GPU NVIDIA A100-SXM4-40GB (Google Colab), a mesma das CNNs |
| Resolução | 256 × 256 (resolução original das imagens) |
| Épocas, lote, paciência | até 50, 32 e 10 |
| Semente | 42, com `deterministic=True` |
| Otimizador e lr | `optimizer=auto`: o Ultralytics escolhe sozinho. O valor efetivo **não foi registrado** no log guardado, e `lr0=0,01` do `args.yaml` é ignorado nesse modo |
| Weight decay | 0,0005 |
| Aumento de dados | padrão do Ultralytics: RandAugment, HSV (0,015 / 0,7 / 0,4), translação 0,1, escala 0,5, espelhamento horizontal 0,5 e apagamento 0,4 |
| Dados | `dados/yolo/` com links simbólicos montados a partir de `particoes.csv` (4.899 treino, 1.050 validação, 1.051 teste) |
| Escolha do modelo | o Ultralytics guarda o `best.pt` pela **acurácia top-1 de validação**; o F1 macro é calculado depois, no `best.pt` |

Os valores efetivos de cada hiperparâmetro estão em `resultados/yolo/yolo_cls_256/args.yaml`.

---

## 2. Resultado

| Item | Valor |
|---|---|
| F1 macro (validação) | **0,9895** |
| Acurácia (validação) | **0,9895** |
| Épocas rodadas / melhor época | 34 (parada antecipada) / 24 |
| Tempo total / por época | 243 s / 7,14 s |
| Parâmetros | 1,54 M (1.543.914) |
| Tamanho do modelo salvo | 3,06 MB |
| Perda de validação no melhor ponto | 0,0428 |

As curvas estão em `resultados/figuras/yolo/yolo_2_curvas.png`. A acurácia de validação chega a 0,98 pela primeira vez na época 12, volta a cair até 0,976 entre as épocas 13 e 18 e, da época 19 em diante, fica entre 0,980 e 0,9895. A melhor época (24, com 0,9895) é um pico dessa oscilação.

---

## 3. Comparação com a melhor CNN

A CNN de referência é a de maior F1 macro de validação entre os **112 treinos de CNN** do projeto: **VGG16SE em 256 × 256 com aumento forte e hiperparâmetros do Optuna** (`docs/CNN.md`, seção 13). A melhor CNN com o aumento leve do notebook (VGG16SE em 256 com Optuna, F1 0,9645) fica 0,5 ponto abaixo dela.

| | VGG16SE-256 (aumento forte, Optuna) | YOLO26n-cls-256 | Diferença |
|---|---|---|---|
| F1 macro | 0,9695 | **0,9895** | +2,0 pts |
| Acurácia | 0,9695 | **0,9895** | +2,0 pts |
| Parâmetros | 14,80 M | 1,54 M | 9,6 vezes menos |
| Tamanho do modelo | 56,53 MB | 3,06 MB | 18 vezes menor |
| Épocas (melhor época) | 93 (83) | 34 (24) | |
| Tempo total | 1.501 s | 243 s | 6,2 vezes menos |
| Tempo por época | 16,14 s | 7,14 s | 2,3 vezes menos |
| Pesos iniciais | novos | pré-treinados no ImageNet | |

O YOLO ganha em desempenho **e** em custo, mas a comparação **não isola a arquitetura**: as duas receitas diferem em pré-treinamento, aumento de dados, otimizador, normalização (o Ultralytics usa a própria, e não a média e o desvio do treino do projeto) e número de parâmetros. A diferença vem, em boa parte, do pré-treinamento, que as CNNs do projeto não têm.

**F1 por classe** (`resultados/figuras/yolo/yolo_3_f1_por_classe.png`)

| Classe | VGG16SE-256 (forte, Optuna) | YOLO26n-cls |
|---|---|---|
| airport | 0,956 | **0,995** |
| beach | 0,981 | 1,000 |
| circular_farmland | 0,995 | 1,000 |
| dense_residential | 0,986 | 0,981 |
| desert | 0,952 | 0,972 |
| forest | 0,986 | 1,000 |
| harbor | 1,000 | 0,995 |
| mountain | 0,930 | 0,976 |
| river | 0,948 | 0,986 |
| storage_tank | 0,962 | 0,991 |
| **macro** | **0,9695** | **0,9895** |

Os maiores ganhos do YOLO são em `mountain` (+4,7 pontos), `airport` (+4,0) e `river` (+3,8). O YOLO só não melhora em `dense_residential` (−0,5) e `harbor` (−0,5). `airport` é a pior classe em 89 dos 112 modelos de CNN, mas nesta CNN a pior é `mountain` (0,930).

---

## 4. Erros

O YOLO errou **11 das 1.050 imagens de validação** (matriz em `resultados/figuras/yolo/yolo_1_matriz_confusao.png`).

| Arquivo | Classe real | Prevista | Confiança |
|---|---|---|---|
| dense_residential_036 | dense_residential | harbor | 0,594 |
| dense_residential_164 | dense_residential | desert | 0,599 |
| dense_residential_262 | dense_residential | storage_tank | 0,602 |
| dense_residential_453 | dense_residential | storage_tank | 0,697 |
| desert_032 | desert | mountain | 0,967 |
| desert_542 | desert | mountain | 0,900 |
| mountain_370 | mountain | desert | 0,901 |
| mountain_632 | mountain | desert | 0,952 |
| river_083 | river | airport | 0,779 |
| river_206 | river | mountain | 0,998 |
| river_523 | river | desert | 0,594 |

- A confiança média nos acertos é 0,991, e nos erros 0,780: o modelo erra com menos confiança, mas **não de forma confiável**. Três erros têm confiança acima de 0,95 (`desert_032` com 0,967, `mountain_632` com 0,952 e `river_206` com 0,998), então a confiança sozinha não serve para filtrar erros.
- **`desert` e `mountain` se confundem nos dois sentidos**, com confiança alta (0,90 a 0,97): superfícies arenosas e relevo seco se parecem.
- Os erros de `dense_residential` têm confiança baixa (0,59 a 0,70) e vão para classes com estruturas pequenas ou terreno aberto (`storage_tank`, `desert`, `harbor`).
- Esses casos merecem inspeção visual das imagens, que não foi feita aqui, para saber se a causa é ambiguidade real ou erro de rótulo.

---

## 5. Diferenças de pré-treinamento e preparação

- **Pré-treinamento:** o YOLO parte de pesos aprendidos no ImageNet e só ajusta o modelo ao RESISC45. As CNNs aprendem tudo do zero em 4.899 imagens. Isso ajuda a explicar a convergência rápida (acurácia de 0,931 já na primeira época) e o ganho em classes difíceis como `airport`.
- **Aumento de dados:** a maioria das CNNs usa espelhamentos e rotações de 90° (aumento leve); também foram treinadas sem aumento e com um aumento forte (recorte com zoom, cor e apagamento, `docs/CNN.md` seção 13), mais parecido com o do YOLO, que usa o aumento padrão do Ultralytics (RandAugment, cor, escala, translação, apagamento). A melhor CNN usa o aumento forte, então parte da distância ao YOLO não vem do aumento.
- **Otimização:** as CNNs usam Adam com lr fixo (inicial ou do Optuna). O YOLO usa o otimizador e a programação de lr escolhidos automaticamente pelo Ultralytics, com período de aquecimento.
- **Modelo:** 1,5 M de parâmetros contra 11 a 15 M das CNNs.
- Não verificamos se há imagens do ImageNet muito parecidas com as do RESISC45, o que seria uma forma de vazamento indireto do pré-treinamento.

---

## 6. Limitações

- **Uma execução** (semente 42). Não há desvio entre execuções.
- **Seleção na validação:** o `best.pt` é a época de maior acurácia de validação. Como a acurácia oscila entre 0,980 e 0,9895 a partir da época 19, o valor de 0,9895 é o topo dessa faixa e fica **ligeiramente otimista** (da ordem de meio ponto).
- A comparação com as CNNs é entre **receitas diferentes** e não prova que a arquitetura do YOLO é melhor.
- O **teste não foi usado**. A avaliação final no teste, única, vem depois, junto com a das demais CNNs escolhidas.
- O otimizador efetivo não ficou registrado no log guardado.
- A generalização para novas regiões, sensores e condições de captura não foi medida.

---

## 7. Onde ficam os arquivos

| O quê | Onde |
|---|---|
| Código do treino | `notebooks/03_CNNs.ipynb` (seção 7) |
| Avaliação na validação | `src/avaliar_yolo_validacao.py` |
| Gráficos | `src/graficos_yolo.py` |
| Resumo, matriz, métricas por classe e previsões | `resultados/yolo/yolo_cls_256_*.{json,csv}` |
| Execução do Ultralytics (args, `results.csv`, curvas, `best.pt`) | `resultados/yolo/yolo_cls_256/` |
| Linha de custo e desempenho | `resultados/experimentos.csv` (`YOLO26n-cls-256_bonus`) |
| Figuras | `resultados/figuras/yolo/` |
| Pesos pré-treinados de partida | `modelos/yolo26n-cls.pt` (não versionado) |
