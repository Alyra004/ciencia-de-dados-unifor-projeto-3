# CNNs: o que foi usado em cada treino

Este documento resume como as CNNs foram montadas, treinadas e avaliadas. O código está em `notebooks/03_CNNs.ipynb` e `src/`, e os números ficam em `resultados/`.

> **Todos os treinos aqui são preliminares.** Só foram usados os dados de **treino** e de **validação**. O **teste não foi usado**. Foram **32 treinos de CNN**: 8 arquiteturas × 2 resoluções (128 × 128 e 256 × 256) × 2 rodadas (hiperparâmetros iniciais e hiperparâmetros achados pelo Optuna). Cada treino foi feito **uma vez só**.

> **Estado:** os 32 treinos acima usam só o aumento de dados **leve**. As receitas **sem aumento** e **aumento forte** estão **em andamento** (parciais): ver a seção 13 e o `docs/PROGRESSO.md`.

---

## 1. Os dados

| Item | Valor |
|---|---|
| Imagens | 7.000, em 10 classes (NWPU-RESISC45) |
| Treino / validação / teste | 4.899 / 1.050 / 1.051 (o teste não foi usado) |
| Entrada da CNN | imagem colorida `3 × R × R`, **sem achatar** (`R` = 128 ou 256) |
| Preparação | pixels divididos por 255 e normalizados com média e desvio por canal calculados **só no treino** |
| Aumento de dados | espelhamentos e rotações de 90°, **só no treino** |

As partições são as mesmas das MLPs (`dados/splits/particoes.csv`): estratificadas, semente 42, com cópias idênticas (MD5) agrupadas na mesma partição.

Os dados foram baixados na VM do Colab pelo Kaggle (`aqibrehmanpirzada/nwpuresisc45`). Esse pacote vem dividido em treino (600 por classe) e teste (100 por classe); os dois foram juntados para recuperar as 700 imagens originais por classe, com os nomes originais, e **a divisão usada é a do projeto**, não a do Kaggle. Conferência: as 7.000 linhas de `particoes.csv` apontam para arquivos existentes e os 6 pares de duplicatas exatas do README têm MD5 idêntico.

---

## 2. As arquiteturas

Duas famílias, cada uma em quatro variantes, escritas do zero em PyTorch (sem pesos pré-treinados). Em todas, a cabeça usa **agregação global** antes da camada de saída.

| Família | Base | Variantes |
|---|---|---|
| ResNet-18 | stem 7 × 7, 4 estágios (64 → 512 canais) com blocos residuais | sem atenção, SE, CBAM, SE + CBAM |
| VGG-16 com BatchNorm | 13 convoluções 3 × 3 em 5 blocos, camadas densas trocadas por agregação global | sem atenção, SE, CBAM, SE + CBAM |

- **SE (Squeeze-and-Excitation):** pondera os canais pela média espacial (Hu et al., 2018).
- **CBAM:** atenção de canal e depois espacial (Woo et al., 2018).

Hipótese da comparação: módulos de atenção podem melhorar o F1 a um custo pequeno de parâmetros, e a VGG-16, mais profunda e com mais operações, pode render mais que a ResNet-18.

Ao passar de 128 para 256, filtros, blocos, kernels e cabeça classificadora são os mesmos, e o número de parâmetros não muda. Parâmetros: ResNet18 11,18 M; SE 11,27 M; CBAM 11,27 M; SE + CBAM 11,36 M. VGG16 14,72 M; SE 14,80 M; CBAM 14,80 M; SE + CBAM 14,88 M.

---

## 3. Como foi o treino

| Item | Rodada inicial | Rodada com Optuna |
|---|---|---|
| Otimizador | Adam | Adam |
| Velocidade de aprendizado (lr) | 0,0001 | achada pelo Optuna (seção 7) |
| Weight decay | 0,0001 | achado pelo Optuna |
| Imagens por lote | 64 | achado pelo Optuna |
| Perda | entropia cruzada | entropia cruzada |
| Limite de épocas | 200 | 200 |
| Parada antecipada | 10 épocas sem melhora no F1 macro da validação | igual |
| Modelo guardado | o da época com maior F1 macro na validação | igual |
| Semente | 42 | 42 |
| Hardware | GPU NVIDIA A100-SXM4-40GB (Google Colab), PyTorch 2.14.1 + CUDA 13.0 | igual |

Os hiperparâmetros iniciais são os mesmos das MLPs, para a comparação entre MLP e CNN não depender de ajustes diferentes. O tempo é medido no laço de treino e validação por época, sem o carregamento dos dados. Os tempos variam um pouco entre execuções do mesmo treino (por exemplo 65 s contra 64 s); o F1, a acurácia e a época do checkpoint foram idênticos.

---

## 4. Resultados dos 32 treinos (validação)

Ordenado por F1 macro. "Checkpoint" é a época do modelo guardado e o total de épocas rodadas. O desvio é o desvio padrão amostral (n − 1) do F1 **entre as 10 classes** de cada modelo; **não** é desvio entre sementes, porque cada treino foi feito uma vez.

| # | Modelo | Res. | Rodada | F1 macro | Desvio (10 classes) | Acurácia | Parâmetros | Checkpoint | Tempo total | s/época |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | VGG16SE | 256 | Optuna | **0,9645** | 0,0310 | 0,9648 | 14,80 M | 63/73 | 1152 s | 15,78 |
| 2 | VGG16CBAM | 256 | Optuna | 0,9590 | 0,0324 | 0,9590 | 14,80 M | 76/86 | 1780 s | 20,70 |
| 3 | VGG16 | 256 | Optuna | 0,9579 | 0,0314 | 0,9581 | 14,72 M | 46/56 | 792 s | 14,15 |
| 4 | VGG16 | 256 | inicial | 0,9570 | 0,0285 | 0,9571 | 14,72 M | 46/56 | 802 s | 14,33 |
| 5 | VGG16CBAM | 256 | inicial | 0,9552 | 0,0346 | 0,9552 | 14,80 M | 40/50 | 1037 s | 20,73 |
| 6 | VGG16SECBAM | 256 | Optuna | 0,9541 | 0,0403 | 0,9543 | 14,88 M | 52/62 | 1385 s | 22,35 |
| 7 | VGG16SE | 256 | inicial | 0,9539 | 0,0472 | 0,9543 | 14,80 M | 41/51 | 811 s | 15,89 |
| 8 | VGG16SECBAM | 128 | Optuna | 0,9520 | 0,0372 | 0,9524 | 14,88 M | 53/63 | 410 s | 6,51 |
| 9 | VGG16 | 128 | inicial | 0,9488 | 0,0400 | 0,9486 | 14,72 M | 47/57 | 217 s | 3,81 |
| 10 | VGG16 | 128 | Optuna | 0,9487 | 0,0399 | 0,9495 | 14,72 M | 51/61 | 243 s | 3,99 |
| 11 | ResNet18SE | 256 | inicial | 0,9481 | 0,0449 | 0,9486 | 11,27 M | 58/68 | 212 s | 3,12 |
| 12 | VGG16SECBAM | 128 | inicial | 0,9464 | 0,0377 | 0,9467 | 14,88 M | 39/49 | 297 s | 6,05 |
| 13 | VGG16CBAM | 128 | inicial | 0,9444 | 0,0488 | 0,9448 | 14,80 M | 40/50 | 281 s | 5,62 |
| 14 | VGG16SE | 128 | Optuna | 0,9433 | 0,0412 | 0,9438 | 14,80 M | 35/45 | 203 s | 4,51 |
| 15 | ResNet18 | 256 | inicial | 0,9426 | 0,0416 | 0,9429 | 11,18 M | 42/52 | 149 s | 2,87 |
| 16 | VGG16SE | 128 | inicial | 0,9377 | 0,0479 | 0,9381 | 14,80 M | 25/35 | 149 s | 4,25 |
| 17 | ResNet18SE | 128 | Optuna | 0,9372 | 0,0444 | 0,9371 | 11,27 M | 60/70 | 199 s | 2,84 |
| 18 | ResNet18CBAM | 256 | Optuna | 0,9368 | 0,0536 | 0,9381 | 11,27 M | 37/47 | 244 s | 5,20 |
| 19 | ResNet18CBAM | 256 | inicial | 0,9364 | 0,0455 | 0,9371 | 11,27 M | 41/51 | 200 s | 3,92 |
| 20 | VGG16SECBAM | 256 | inicial | 0,9356 | 0,0531 | 0,9352 | 14,88 M | 21/31 | 692 s | 22,33 |
| 21 | ResNet18 | 256 | Optuna | 0,9356 | 0,0530 | 0,9362 | 11,18 M | 40/50 | 155 s | 3,10 |
| 22 | ResNet18CBAM | 128 | Optuna | 0,9349 | 0,0459 | 0,9352 | 11,27 M | 60/70 | 290 s | 4,15 |
| 23 | VGG16CBAM | 128 | Optuna | 0,9347 | 0,0534 | 0,9352 | 14,80 M | 17/27 | 162 s | 5,99 |
| 24 | ResNet18SECBAM | 256 | Optuna | 0,9326 | 0,0582 | 0,9333 | 11,36 M | 37/47 | 278 s | 5,91 |
| 25 | ResNet18SE | 256 | Optuna | 0,9314 | 0,0529 | 0,9324 | 11,27 M | 37/47 | 177 s | 3,76 |
| 26 | ResNet18SECBAM | 128 | inicial | 0,9265 | 0,0577 | 0,9267 | 11,36 M | 61/71 | 180 s | 2,53 |
| 27 | ResNet18SECBAM | 256 | inicial | 0,9262 | 0,0655 | 0,9276 | 11,36 M | 26/36 | 149 s | 4,15 |
| 28 | ResNet18CBAM | 128 | inicial | 0,9245 | 0,0604 | 0,9248 | 11,27 M | 42/52 | 113 s | 2,18 |
| 29 | ResNet18 | 128 | Optuna | 0,9164 | 0,0550 | 0,9171 | 11,18 M | 38/48 | 89 s | 1,85 |
| 30 | ResNet18 | 128 | inicial | 0,9132 | 0,0625 | 0,9133 | 11,18 M | 48/58 | 65 s | 1,13 |
| 31 | ResNet18SECBAM | 128 | Optuna | 0,9099 | 0,0678 | 0,9095 | 11,36 M | 41/51 | 255 s | 4,99 |
| 32 | ResNet18SE | 128 | inicial | 0,8766 | 0,0882 | 0,8762 | 11,27 M | 8/18 | 26 s | 1,46 |

**Leitura**

- As **VGGs dominam o topo**: as 7 primeiras posições são VGGs em 256, e as VGGs ficam 2 a 7 pontos acima das ResNets em quase todos os cenários. O custo é de 2 a 5 vezes mais tempo por época.
- O **melhor modelo é a VGG16SE em 256 com Optuna** (F1 macro 0,9645, acurácia 0,9648). A vantagem sobre o 2º lugar é de 0,55 ponto, e cada treino foi feito uma vez, então isso não prova superioridade.
- A **melhor ResNet é a ResNet18SE em 256 com hiperparâmetros iniciais** (0,9481), que fica atrás de 10 VGGs.

### Média e desvio entre modelos (F1 macro de cada modelo, n = 4 em cada grupo)

| Resolução | Rodada | Família | Média | Desvio |
|---|---|---|---|---|
| 128 | inicial | ResNet18 | 0,9102 | 0,0232 |
| 128 | Optuna | ResNet18 | 0,9246 | 0,0135 |
| 256 | inicial | ResNet18 | 0,9383 | 0,0094 |
| 256 | Optuna | ResNet18 | 0,9341 | 0,0025 |
| 128 | inicial | VGG16 | 0,9443 | 0,0048 |
| 128 | Optuna | VGG16 | 0,9447 | 0,0076 |
| 256 | inicial | VGG16 | 0,9504 | 0,0100 |
| 256 | Optuna | VGG16 | 0,9589 | 0,0043 |

---

## 5. Efeito da resolução (rodada inicial: mesmos hiperparâmetros, lote 64)

| Modelo | F1 em 128 | F1 em 256 | Ganho | s/época em 128 | s/época em 256 |
|---|---|---|---|---|---|
| ResNet18SE | 0,8766 | 0,9481 | +7,2 pts | 1,46 | 3,12 |
| ResNet18 | 0,9132 | 0,9426 | +2,9 pts | 1,13 | 2,87 |
| VGG16SE | 0,9377 | 0,9539 | +1,6 pts | 4,25 | 15,89 |
| ResNet18CBAM | 0,9245 | 0,9364 | +1,2 pts | 2,18 | 3,92 |
| VGG16CBAM | 0,9444 | 0,9552 | +1,1 pts | 5,62 | 20,73 |
| VGG16 | 0,9488 | 0,9570 | +0,8 pts | 3,81 | 14,33 |
| ResNet18SECBAM | 0,9265 | 0,9262 | −0,0 pts | 2,53 | 4,15 |
| VGG16SECBAM | 0,9464 | 0,9356 | −1,1 pts | 6,05 | 22,33 |
| **Média das 8** | 0,9273 | 0,9444 | +1,7 pts | 3,38 | 10,92 |

- Dobrar a resolução rende em média **+1,7 ponto de F1**, por cerca de **3,2 vezes o tempo por época**. O ganho é desigual: vai de +7,2 a −1,1.
- **ResNets:** média de +2,8 pontos por 1,6 a 2,5 vezes o tempo. Boa parte do +7,2 da SE vem de o treino dela em 128 ter parado cedo (época 8 de 18), então o efeito puro da resolução é menor.
- **VGGs:** média de +0,6 ponto (a VGG16SECBAM piora 1,1 e também parou cedo, na época 21) por cerca de **3,7 vezes o tempo**. A VGG já estava perto do teto em 128 e tem pouco a ganhar com a resolução.
- As duas variantes SE + CBAM não ganharam nada em 256. Com uma execução por modelo, não dá para separar efeito real de sorte de inicialização.

---

## 6. Comparação entre arquiteturas

- **Atenção na VGG não compensou na rodada inicial:** a VGG16 sem atenção ficou em 1º em 128 e em 256, e as variantes com atenção ficaram 0,2 a 1,1 ponto atrás, com 11% a 60% mais tempo por época. Com Optuna a conta muda: em 256 a VGG16SE passa a VGG16 (seção 7).
- **Atenção na ResNet ajudou em 128 (CBAM e SE + CBAM, +1,1 a +1,3 ponto sobre a ResNet18 pura)**, mas a SE sozinha foi a pior de todas em 128 e a melhor das ResNets em 256. A ordem das variantes muda com a resolução, o que mostra o peso do ruído de execução única.
- **Custo:** a ResNet18 pura em 128 treina em 65 s e fica em 0,9132. A VGG16 pura em 256 treina em 802 s (12 vezes mais) e chega a 0,9570. A VGG16 pura em 128 (217 s) chega a 0,9488, e é provavelmente o melhor equilíbrio entre desempenho e custo.

---

## 7. Hiperparâmetros com Optuna

O [Optuna](https://optuna.org) é uma biblioteca que testa combinações de valores e usa o resultado de cada teste para escolher o próximo. Conta como **busca adicional**, separada das execuções finais. Quem decide qual combinação é melhor é o **F1 macro na validação**. O teste **não é usado**.

| Item | Valor |
|---|---|
| Buscas | uma por família (ResNet18 e VGG16) e por resolução (4 buscas) |
| Rede usada na busca | a **versão sem atenção** de cada família; o resultado é aplicado também às variantes SE, CBAM e SE + CBAM |
| Tentativas por busca | 10, com no máximo 30 épocas cada e poda das tentativas ruins |
| Faixas | lr 1e-5 a 3e-3, weight decay 1e-6 a 1e-2, lote 32, 64 ou 128 |
| Aumento de dados | leve |

**Valores escolhidos** (os de 128 saem de `tentativas_*-128.csv`, porque o arquivo `melhores_hiperparametros_cnn.json` é sobrescrito pela rodada de 256 e guarda só os de 256):

| Busca | lr | Weight decay | Lote | Melhor F1 na busca | Concluídas / podadas |
|---|---|---|---|---|---|
| ResNet18, 128 | 8,5e-5 | 6,4e-3 | 32 | 0,9082 | 3 / 7 |
| VGG16, 128 | 2,4e-5 | 1,7e-6 | 32 | 0,9252 | 3 / 7 |
| ResNet18, 256 | 2,4e-5 | 1,7e-6 | 32 | 0,9117 | 3 / 7 |
| VGG16, 256 | 2,8e-5 | 1,6e-5 | 32 | 0,9387 | 4 / 6 |

Os valores se repetem entre estudos (por exemplo, o da VGG16 em 128 é igual ao da ResNet18 em 256) porque as primeiras tentativas usam a mesma semente: as primeiras combinações testadas são as mesmas em todos os estudos.

### Optuna contra a rodada inicial

| Modelo | F1 inicial (128) | F1 Optuna (128) | Ganho | F1 inicial (256) | F1 Optuna (256) | Ganho |
|---|---|---|---|---|---|---|
| ResNet18 | 0,9132 | 0,9164 | +0,3 pts | 0,9426 | 0,9356 | −0,7 pts |
| ResNet18SE | 0,8766 | 0,9372 | +6,1 pts | 0,9481 | 0,9314 | −1,7 pts |
| ResNet18CBAM | 0,9245 | 0,9349 | +1,0 pts | 0,9364 | 0,9368 | +0,0 pts |
| ResNet18SECBAM | 0,9265 | 0,9099 | −1,7 pts | 0,9262 | 0,9326 | +0,6 pts |
| VGG16 | 0,9488 | 0,9487 | −0,0 pts | 0,9570 | 0,9579 | +0,1 pts |
| VGG16SE | 0,9377 | 0,9433 | +0,6 pts | 0,9539 | 0,9645 | +1,1 pts |
| VGG16CBAM | 0,9444 | 0,9347 | −1,0 pts | 0,9552 | 0,9590 | +0,4 pts |
| VGG16SECBAM | 0,9464 | 0,9520 | +0,6 pts | 0,9356 | 0,9541 | +1,9 pts |

- **Em 128:** ajudou 5 dos 8 modelos. Média das ResNets de 0,9102 para 0,9246 (+1,4 ponto, quase todo da SE) e das VGGs praticamente igual (0,9443 para 0,9447).
- **Em 256:** ajudou as **4 VGGs** (+0,1 a +1,9) e foi neutro ou negativo nas **ResNets** (média de 0,9383 para 0,9341). O melhor modelo do projeto é um resultado do Optuna (VGG16SE em 256).
- O custo por época varia de −1% a +97% por causa do lote de 32 contra 64: nas VGGs quase não muda, e nas ResNets em 128 chega a quase dobrar.

**Por que a busca às vezes fica abaixo dos hiperparâmetros iniciais.** A melhor tentativa de cada busca ficou **abaixo** do treino inicial correspondente (por exemplo, ResNet18 em 128: 0,908 contra 0,913; VGG16 em 128: 0,925 contra 0,949). Causas prováveis:

1. As tentativas são cortadas em 30 épocas e podadas, enquanto o treino final vai a até 200 épocas. Combinações com lr baixo, como as escolhidas, parecem piores na busca e só se recuperam no treino longo.
2. Com 10 tentativas e 3 a 4 concluídas, o Optuna pode simplesmente não ter amostrado algo melhor que lr 1e-4, weight decay 1e-4 e lote 64.
3. A busca roda na rede sem atenção e é aplicada às variantes com atenção, o que explica perdas como a da ResNet18SECBAM em 128.
4. Ruído de execução única, e o "melhor da busca" é o máximo de poucas tentativas, portanto otimista.

Melhorias que não foram aplicadas, para manter a reprodução fiel: enfileirar os hiperparâmetros iniciais como a primeira tentativa (`study.enqueue_trial`), dar às tentativas o mesmo orçamento de épocas do treino final, usar mais tentativas e escolher no fim, pela validação, entre inicial e Optuna. A afirmação de que o Optuna "corrige treinos instáveis" (a SE em 128 parou na época 8) é só uma hipótese, apoiada por um caso.

---

## 8. Erros por classe e matrizes de confusão

As matrizes (reais nas linhas, previstas nas colunas, contagens e normalizadas pela classe real) e as métricas por classe são calculadas **na validação**, a partir dos checkpoints salvos, por `src/avaliar_cnn_validacao.py`. O F1 e a acurácia calculados por esse script **reproduzem exatamente** os registrados no treino, para os 32 modelos.

**F1 médio por classe (média dos 32 modelos)**

| Classe | F1 médio |
|---|---|
| forest | 0,9838 |
| harbor | 0,9836 |
| circular_farmland | 0,9751 |
| dense_residential | 0,9751 |
| beach | 0,9666 |
| storage_tank | 0,9345 |
| desert | 0,9265 |
| mountain | 0,8940 |
| river | 0,8922 |
| **airport** | **0,8507** |

- **`airport` é a pior classe em 30 dos 32 modelos** (nos outros 2, é `mountain`). É uma classe visualmente variada (pistas, terminais, pátios e instalações) e é a classe das duplicatas exatas, mas o motivo provável do erro é visual.
- `forest` e `harbor` são quase sempre acertadas, e `beach`, `circular_farmland` e `dense_residential` também.

**Melhor modelo (VGG16SE em 256 com Optuna), por classe (validação, 105 imagens por classe)**

| Classe | Precisão | Recall | F1 |
|---|---|---|---|
| airport | 0,949 | 0,886 | 0,916 |
| beach | 0,991 | 1,000 | 0,995 |
| circular_farmland | 0,981 | 1,000 | 0,991 |
| dense_residential | 0,990 | 0,971 | 0,981 |
| desert | 0,912 | 0,981 | 0,945 |
| forest | 0,981 | 1,000 | 0,991 |
| harbor | 1,000 | 1,000 | 1,000 |
| mountain | 0,960 | 0,914 | 0,937 |
| river | 0,933 | 0,924 | 0,928 |
| storage_tank | 0,953 | 0,971 | 0,962 |
| **macro** | **0,965** | **0,965** | **0,965** |

**Pares de classes mais confundidos nesse modelo** (real → prevista, contagem): `airport → desert` (6), `river → airport` (4), `airport → river` (3), `mountain → desert` (3). `airport` e `desert` se confundem nos dois sentidos (pistas e terminais cercados por terreno arenoso), assim como `mountain` e `desert` (relevo e solo seco), e `dense_residential` aparece confundida com `storage_tank` (estruturas pequenas e agrupadas). **Divisões por zero:** não ocorreram em nenhum dos 32 modelos. Todas as classes têm 105 imagens de suporte e foram previstas pelo menos uma vez em todos os modelos (a menor precisão por classe foi 0,727 e o menor recall foi 0,686), então precisão e recall estão definidos sempre. O cálculo usa `zero_division=0` por precaução.

---

## 9. Sobre vazamento de dados

Verificado no código do notebook:

- A média e o desvio da normalização saem só do treino.
- O aumento de dados só roda no treino.
- O notebook só carrega as partições de treino e validação. O teste não é lido em nenhum treino de CNN.
- O checkpoint e a parada antecipada são decididos pelo F1 macro de validação.

**Quase-duplicatas entre partições.** O MD5 só detecta cópias idênticas byte a byte. Uma busca por hash perceptual (dHash de 64 bits, distância ≤ 4 bits) nas 7.000 imagens achou:

| Resultado | Valor |
|---|---|
| Pares quase idênticos | 16 (11 na mesma partição, 5 entre partições) |
| Entre partições | todos da mesma classe: 4 em `beach` e 1 em `mountain` |
| Tipos | 4 pares treino/validação e 1 par treino/teste |
| Imagens de validação com vizinho no treino | 3 de 1.050 (0,3%) |

O efeito é pequeno demais para mudar o F1 de forma relevante. Mas a regra do projeto pede que "versões da mesma imagem" fiquem no mesmo conjunto, então fica registrado como **limitação**. Corrigir exigiria refazer as partições de todos os modelos, inclusive das MLPs. O hash é grosseiro: recortes, rotações e ampliações da mesma região não são detectados.

**Viés de seleção.** A validação escolhe o checkpoint, a época de parada e os hiperparâmetros do Optuna. O F1 de validação fica, portanto, **ligeiramente otimista**. O desempenho de generalização só aparece no teste, avaliado uma vez ao final, e isso ainda não foi feito.

---

## 10. Reprodutibilidade e incidente

Os 32 treinos foram executados **duas vezes**. Na primeira, a VM do Colab foi recolhida por inatividade (o computador local entrou em suspensão) antes de os resultados serem baixados, e os checkpoints, históricos e matrizes se perderam. Na segunda execução, com a mesma semente (42), **os 30 treinos que já tinham resultado registrado deram exatamente o mesmo F1 macro, a mesma acurácia e a mesma época de checkpoint**. Os 2 treinos que a primeira execução não tinha terminado (VGG16CBAM e VGG16SECBAM em 256 com Optuna) só existem na segunda. Os tempos variam um pouco entre execuções, mas os resultados de qualidade não. Os números deste documento são os da segunda execução.

A avaliação por classe foi feita na GPU após o fim dos treinos. Durante o treino de 256 ela foi feita na CPU, para não tirar memória do treino; os resultados da CPU batiam com os do treino, e a avaliação final na GPU os substituiu.

---

## 11. Limitações

- Cada treino foi feito **uma vez** (semente 42). Diferenças de 1 a 2 pontos de F1 podem ser sorte de inicialização.
- Todos os números são da **validação** e **preliminares**. O teste não foi usado.
- O Optuna usou 10 tentativas por busca, com orçamento menor que o do treino final, e foi aplicado às variantes com atenção sem buscar para elas.
- **Os 32 treinos principais usam a mesma receita de aumento de dados** (espelhamentos e rotações de 90°, o "aumento leve" das MLPs). As receitas sem aumento e com aumento forte só foram treinadas **em parte** (seção 13): faltam a maior parte delas e todas as de 256 × 256, então o efeito do aumento sobre as CNNs ainda não está completo. Nas MLPs, o aumento leve foi o que mais ajudou, mas isso não vale necessariamente para as CNNs.
- A comparação descreve as condições executadas e não prova superioridade universal.
- A generalização para novas regiões, sensores e condições de captura não foi medida.
- As seis execuções finais exigidas pelo PDF ainda precisam ser definidas: o notebook treina 8 variantes por resolução, e o projeto precisa escolher duas arquiteturas (A e B) por família para os treinos finais e a avaliação no teste.

---

## 12. Onde ficam os arquivos

| O quê | Onde |
|---|---|
| Código e explicações | `notebooks/03_CNNs.ipynb` |
| Avaliação na validação (matrizes e métricas por classe) | `src/avaliar_cnn_validacao.py` |
| Gráficos de resumo e matrizes | `src/graficos_cnn.py` |
| Tabela de todos os treinos (MLPs e CNNs) | `resultados/experimentos.csv` |
| Nota por época de cada treino | `resultados/historicos/` |
| Matrizes de confusão (contagens) | `resultados/matrizes_confusao/` |
| Métricas por classe | `resultados/metricas_por_classe/` |
| Previsões na validação (arquivo, real, prevista, confiança) | `resultados/previsoes_validacao_cnn.csv` |
| Configuração de cada treino | `resultados/configs/` |
| Busca do Optuna (tentativas e melhores valores) | `resultados/optuna/` |
| Bônus YOLO (pré-treinado) | `docs/YOLO.md` e `resultados/yolo/` |
| Figuras | `resultados/figuras/cnn/` (resumos, curvas por treino e `matrizes/` com uma figura por modelo) |
| Modelos guardados (32 checkpoints, 1,6 GB, não versionados) | `modelos/` |

Principais figuras: `resumo_1_f1_antes_depois_{128,256}.png` (inicial contra Optuna), `resumo_2_custo_desempenho_{128,256}.png`, `resumo_3_f1_por_classe_{128,256}.png`, `resumo_4_curvas_validacao_{128,256}.png`, `resumo_5_matrizes_{inicial,optuna}_{128,256}.png` (grade das 8 matrizes) e `resumo_6_pares_confundidos_{128,256}.png`.

---

## 13. Variações de aumento de dados (parcial)

Para completar a comparação (como nas MLPs), as CNNs também estão sendo treinadas **sem aumento** e com o **aumento forte** (o mesmo das MLPs: leve mais zoom, cor e apagamento), cada receita com hiperparâmetros iniciais e do Optuna (a busca do Optuna foi feita com aumento leve e **não é refeita**). O código é `src/rodar_cnns_aumentos.py` e os nomes dos treinos são `<CNN>-<R>_semaug[_opt]_preliminar` (sem aumento) e `<CNN>-<R>_aug2[_opt]_preliminar` (aumento forte). São 64 treinos no total; **8 foram concluídos** e o resto ficou para depois (ver `docs/PROGRESSO.md`).

**Concluído: 128 × 128, hiperparâmetros iniciais, sem aumento** (validação, uma execução):

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

- **Sem aumento, as 8 CNNs perdem em média 4,3 pontos de F1** em relação ao aumento leve (de −1,2 a −9,2). As ResNets perdem mais (média de −5,7) que as VGGs (média de −3,0).
- O tempo por época é praticamente o mesmo com e sem aumento (o aumento é barato na GPU); o tempo total muda porque o número de épocas muda.
- **Limite:** são só 8 dos 64 treinos, e cada um foi feito uma vez. Falta o aumento forte, o Optuna com as duas receitas e toda a resolução 256 × 256 para concluir o efeito do aumento nas CNNs.
- **Artefatos incompletos:** dos 8 treinos acima, só VGG16, VGG16SE e VGG16CBAM têm todos os arquivos. Das quatro ResNets existem a linha de resultados, a matriz de confusão e as métricas por classe, mas não o checkpoint, o histórico por época nem as curvas. Do VGG16SECBAM existe só a linha de resultados. Os F1 acima vêm do próprio log do treino. Reexecutar o script refaz esses 5 treinos (`docs/PROGRESSO.md`).
