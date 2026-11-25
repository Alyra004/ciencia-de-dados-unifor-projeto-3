# CNNs: o que foi usado em cada treino

Este documento resume como as CNNs foram montadas, treinadas e avaliadas. O código está em `notebooks/03_CNNs.ipynb` e `src/`, e os números ficam em `resultados/`.

> **Todos os treinos aqui são preliminares.** Só foram usados os dados de **treino** e de **validação**. O **teste não foi usado**. Foram **112 treinos de CNN**: 96 (8 arquiteturas × 2 resoluções (128 × 128 e 256 × 256) × 2 rodadas (hiperparâmetros iniciais e hiperparâmetros achados pelo Optuna) × 3 receitas de aumento de dados: leve, sem aumento e forte) e mais 16 com o Optuna refeito com o aumento forte (seção 14). Cada treino foi feito **uma vez só**.

> **Como ler este documento:** as seções 4 a 8 tratam dos **32 treinos com aumento leve** (a receita do notebook). A seção 13 traz os outros **64 treinos** (sem aumento e aumento forte) e a comparação entre as três receitas, e a seção 14 traz os 16 treinos com o Optuna refeito com o aumento forte. **O melhor treino de CNN é a VGG16SE em 256 com aumento forte e Optuna (F1 macro 0,9695)**, descrito na seção 13; entre os 32 com aumento leve, o melhor é a VGG16SE em 256 com Optuna (0,9645).

---

## 1. Os dados

| Item | Valor |
|---|---|
| Imagens | 7.000, em 10 classes (NWPU-RESISC45) |
| Treino / validação / teste | 4.899 / 1.050 / 1.051 (o teste não foi usado) |
| Entrada da CNN | imagem colorida `3 × R × R`, **sem achatar** (`R` = 128 ou 256) |
| Preparação | pixels divididos por 255 e normalizados com média e desvio por canal calculados **só no treino** |
| Aumento de dados | nos 32 primeiros treinos: espelhamentos e rotações de 90° (aumento leve), **só no treino**. As receitas sem aumento e com aumento forte estão na seção 13 |

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

## 4. Resultados dos 32 treinos com aumento leve (validação)

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
- O **melhor modelo com aumento leve é a VGG16SE em 256 com Optuna** (F1 macro 0,9645, acurácia 0,9648). A vantagem sobre o 2º lugar é de 0,55 ponto, e cada treino foi feito uma vez, então isso não prova superioridade. Considerando também as outras receitas de aumento (seção 13), o melhor treino é a mesma arquitetura com aumento forte e Optuna (0,9695).
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
- **Em 256:** ajudou as **4 VGGs** (+0,1 a +1,9) e foi neutro ou negativo nas **ResNets** (média de 0,9383 para 0,9341). O melhor modelo com aumento leve é um resultado do Optuna (VGG16SE em 256), e o melhor de todas as receitas (seção 13) também.
- O custo por época varia de −1% a +97% por causa do lote de 32 contra 64: nas VGGs quase não muda, e nas ResNets em 128 chega a quase dobrar.

**Por que a busca às vezes fica abaixo dos hiperparâmetros iniciais.** A melhor tentativa de cada busca ficou **abaixo** do treino inicial correspondente (por exemplo, ResNet18 em 128: 0,908 contra 0,913; VGG16 em 128: 0,925 contra 0,949). Causas prováveis:

1. As tentativas são cortadas em 30 épocas e podadas, enquanto o treino final vai a até 200 épocas. Combinações com lr baixo, como as escolhidas, parecem piores na busca e só se recuperam no treino longo.
2. Com 10 tentativas e 3 a 4 concluídas, o Optuna pode simplesmente não ter amostrado algo melhor que lr 1e-4, weight decay 1e-4 e lote 64.
3. A busca roda na rede sem atenção e é aplicada às variantes com atenção, o que explica perdas como a da ResNet18SECBAM em 128.
4. Ruído de execução única, e o "melhor da busca" é o máximo de poucas tentativas, portanto otimista.

Melhorias que não foram aplicadas, para manter a reprodução fiel: enfileirar os hiperparâmetros iniciais como a primeira tentativa (`study.enqueue_trial`), dar às tentativas o mesmo orçamento de épocas do treino final, usar mais tentativas e escolher no fim, pela validação, entre inicial e Optuna. A afirmação de que o Optuna "corrige treinos instáveis" (a SE em 128 parou na época 8) é só uma hipótese, apoiada por um caso.

---

## 8. Erros por classe e matrizes de confusão

As matrizes (reais nas linhas, previstas nas colunas, contagens e normalizadas pela classe real) e as métricas por classe são calculadas **na validação**, a partir dos checkpoints salvos, por `src/avaliar_cnn_validacao.py`. O F1 e a acurácia calculados por esse script **reproduzem exatamente** os registrados no treino, para os 112 modelos. As tabelas desta seção cobrem os **32 modelos com aumento leve**; as matrizes e métricas dos outros 64 estão nos mesmos diretórios.

**F1 médio por classe (média dos 32 modelos com aumento leve)**

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

**Melhor modelo com aumento leve (VGG16SE em 256 com Optuna), por classe (validação, 105 imagens por classe)**

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

**Pares de classes mais confundidos nesse modelo** (real → prevista, contagem): `airport → desert` (6), `river → airport` (4), `airport → river` (3), `mountain → desert` (3). `airport` e `desert` se confundem nos dois sentidos (pistas e terminais cercados por terreno arenoso), assim como `mountain` e `desert` (relevo e solo seco), e `dense_residential` aparece confundida com `storage_tank` (estruturas pequenas e agrupadas). **Divisões por zero:** não ocorreram em nenhum dos 112 modelos. Todas as classes têm 105 imagens de suporte e foram previstas pelo menos uma vez em todos os modelos (nos 32 com aumento leve, a menor precisão por classe foi 0,727 e o menor recall foi 0,686; nos 112, 0,606 e 0,562), então precisão e recall estão definidos sempre. Nos 112 modelos, `airport` é a pior classe em 89, `river` em 12 e `mountain` em 11. O cálculo usa `zero_division=0` por precaução.

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

## 10. Reprodutibilidade e incidentes

**Os 32 treinos com aumento leve** foram executados **duas vezes**. Na primeira, a VM do Colab foi recolhida por inatividade (o computador local entrou em suspensão) antes de os resultados serem baixados, e os checkpoints, históricos e matrizes se perderam. Na segunda execução, com a mesma semente (42), **os 30 treinos que já tinham resultado registrado deram exatamente o mesmo F1 macro, a mesma acurácia e a mesma época de checkpoint**. Os 2 treinos que a primeira execução não tinha terminado (VGG16CBAM e VGG16SECBAM em 256 com Optuna) só existem na segunda. Os tempos variam um pouco entre execuções, mas os resultados de qualidade não. Os números deste documento são os da segunda execução.

**Os 64 treinos da seção 13** rodaram numa VM A100 nova, com backup incremental para o computador local. Na primeira tentativa, o marcador do backup começou depois dos 5 primeiros treinos sem aumento em 128 × 128 (ResNet18, ResNet18SE, ResNet18CBAM, ResNet18SECBAM e VGG16SECBAM): só ficaram as linhas de resultado, as matrizes e as métricas, sem checkpoint, histórico e curvas. Esses 5 treinos foram **refeitos**, e os checkpoints e F1 reproduziram os registrados antes. O script `src/rodar_cnns_aumentos.py` é retomável: pula treinos que já têm histórico, linha de resultados e checkpoint.

**Avaliação por classe.** Durante os treinos, a avaliação (matriz, métricas por classe e previsões) foi feita na CPU, para não tirar memória do treino. No fim, ela foi refeita **na GPU** (`AVALIAR_FORCAR=1`), primeiro para os 96 modelos e depois para os 16 da seção 14, e substituiu a da CPU: 9 das 100.800 previsões individuais dos 96 e 2 das 16.800 dos 16 mudaram entre as duas, e nenhum F1 ou acurácia muda (as métricas calculadas na avaliação reproduzem as do treino nos 112 modelos). Os 16 treinos da seção 14 rodaram numa VM A100 nova, com a mesma rotina de backup, sem perda de artefatos.

---

## 11. Limitações

- Cada treino foi feito **uma vez** (semente 42). Diferenças de 1 a 2 pontos de F1 podem ser sorte de inicialização.
- Todos os números são da **validação** e **preliminares**. O teste não foi usado.
- O Optuna usou 10 tentativas por busca, com orçamento menor que o do treino final, e foi aplicado às variantes com atenção sem buscar para elas.
- **O Optuna foi feito com aumento leve** e usado, sem nova busca, nos treinos sem aumento e com aumento forte (seção 13). Para o aumento forte a busca foi refeita (seção 14) e não melhorou o resultado. Para o treino sem aumento ela **não foi refeita**: os valores "Optuna" desses treinos não são os que uma busca específica acharia. Todas as buscas são pequenas (10 tentativas, 3 ou 4 concluídas).
- **Paradas precoces.** Com paciência de 10 épocas, alguns treinos pararam muito cedo (checkpoint na época 12 ou antes em 8 dos 96, 5 deles sem aumento). Parte do efeito de uma receita ou de um hiperparâmetro pode ser parada precoce por ruído.
- **O melhor dos 96 treinos é o máximo de 96 números com ruído,** escolhido na validação: o F1 dele é o mais otimista de todos. A vantagem de 0,5 ponto sobre o 2º lugar não é distinguível de ruído de inicialização.
- A comparação descreve as condições executadas e não prova superioridade universal.
- A generalização para novas regiões, sensores e condições de captura não foi medida.
- As seis execuções finais exigidas pelo PDF ainda precisam ser definidas: foram treinadas 8 variantes por resolução e receita, e o projeto precisa escolher duas arquiteturas (A e B) para os treinos finais, do zero, e a avaliação no teste.

---

## 12. Onde ficam os arquivos

| O quê | Onde |
|---|---|
| Código e explicações | `notebooks/03_CNNs.ipynb` |
| Avaliação na validação (matrizes e métricas por classe) | `src/avaliar_cnn_validacao.py` |
| Treino das receitas sem aumento e com aumento forte | `src/rodar_cnns_aumentos.py` |
| Gráficos de resumo e matrizes | `src/graficos_cnn.py` e `src/graficos_cnn_aumentos.py` (seção 13) |
| Tabela de todos os treinos (MLPs e CNNs) | `resultados/experimentos.csv` |
| Nota por época de cada treino | `resultados/historicos/` |
| Matrizes de confusão (contagens) | `resultados/matrizes_confusao/` |
| Métricas por classe | `resultados/metricas_por_classe/` |
| Previsões na validação (arquivo, real, prevista, confiança) | `resultados/previsoes_validacao_cnn.csv` |
| Configuração de cada treino | `resultados/configs/` |
| Busca do Optuna (tentativas e melhores valores) | `resultados/optuna/` |
| Bônus YOLO (pré-treinado) | `docs/YOLO.md` e `resultados/yolo/` |
| Figuras | `resultados/figuras/cnn/` (resumos, curvas por treino e `matrizes/` com uma figura por modelo) |
| Optuna refeito com aumento forte (seção 14) | `src/optuna_cnns_aumento_forte.py`; `resultados/optuna/tentativas_*_aug2.csv` e `melhores_hiperparametros_cnn_aug2_<R>.json` |
| Modelos guardados (112 checkpoints, 5,5 GB, não versionados) | `modelos/` |

Principais figuras (aumento leve): `resumo_1_f1_antes_depois_{128,256}.png` (inicial contra Optuna), `resumo_2_custo_desempenho_{128,256}.png`, `resumo_3_f1_por_classe_{128,256}.png`, `resumo_4_curvas_validacao_{128,256}.png`, `resumo_5_matrizes_{inicial,optuna}_{128,256}.png` (grade das 8 matrizes) e `resumo_6_pares_confundidos_{128,256}.png`.

---

## 13. Variações de aumento de dados

Para completar a comparação (como nas MLPs), as 8 CNNs foram treinadas também **sem aumento** e com o **aumento forte** (o mesmo das MLPs: aumento leve mais recorte com zoom de 50 a 100% da área, variação de cor de ±20% e apagamento aleatório), cada receita com hiperparâmetros iniciais e do Optuna, em 128 × 128 e em 256 × 256. São **64 treinos novos**, que somados aos 32 com aumento leve (seções 4 a 8) dão **96 treinos de CNN** (8 arquiteturas × 2 resoluções × 2 rodadas × 3 receitas). Todos foram feitos **uma vez só**, na mesma GPU A100 e com o mesmo código de treino, e só usam treino e validação.

O código é `src/rodar_cnns_aumentos.py` (a avaliação é a mesma de `src/avaliar_cnn_validacao.py`). Nomes dos treinos: `<CNN>-<R>_semaug[_opt]_preliminar` (sem aumento) e `<CNN>-<R>_aug2[_opt]_preliminar` (aumento forte).

**Atenção ao Optuna:** a busca de hiperparâmetros foi feita **uma vez**, com aumento leve e na rede sem atenção (seção 7), e **não foi refeita** para as outras receitas nesta seção. Nos treinos "Optuna" destas duas receitas usam-se os valores da busca com aumento leve. (A busca foi refeita depois só para o aumento forte; resultados na seção 14.) Portanto, a comparação "Optuna contra inicial" nas receitas novas mede se esses valores se transferem, e não o que o Optuna acharia para elas.

### 13.1 Médias por receita (F1 macro; n = 8 modelos em cada linha)

| Resolução | Rodada | Sem aumento | Aumento leve | Aumento forte |
|---|---|---|---|---|
| 128 | inicial | 0,8841 | 0,9273 | 0,9171 |
| 128 | Optuna | 0,8756 | 0,9346 | 0,9195 |
| 256 | inicial | 0,9214 | 0,9444 | 0,9413 |
| 256 | Optuna | 0,9147 | 0,9465 | 0,9399 |
| **Todas** (n = 32) | | **0,8989** | **0,9382** | **0,9295** |

Por família (n = 4 em cada célula):

| Resolução | Rodada | Família | Sem aumento | Aumento leve | Aumento forte |
|---|---|---|---|---|---|
| 128 | inicial | ResNet18 | 0,8534 | 0,9102 | 0,8913 |
| 128 | inicial | VGG16 | 0,9147 | 0,9443 | 0,9429 |
| 128 | Optuna | ResNet18 | 0,8380 | 0,9246 | 0,9124 |
| 128 | Optuna | VGG16 | 0,9133 | 0,9447 | 0,9265 |
| 256 | inicial | ResNet18 | 0,9117 | 0,9383 | 0,9380 |
| 256 | inicial | VGG16 | 0,9310 | 0,9504 | 0,9446 |
| 256 | Optuna | ResNet18 | 0,9000 | 0,9341 | 0,9242 |
| 256 | Optuna | VGG16 | 0,9294 | 0,9589 | 0,9557 |

- **Sem aumento contra aumento leve** (mesma arquitetura, resolução e rodada; n = 32 pares): média de **−3,92 pts**, de −9,82 pts a −0,08 pts; piorou em 32 e melhorou em 0. ResNets −5,10 pts (n = 16) e VGGs −2,75 pts (n = 16).
- **Aumento forte contra aumento leve** (mesma arquitetura, resolução e rodada; n = 32 pares): média de **−0,87 pts**, de −5,62 pts a +1,69 pts; piorou em 23 e melhorou em 9. ResNets −1,03 pts (n = 16) e VGGs −0,71 pts (n = 16).

### 13.2 128 × 128, sem aumento, hiperparâmetros iniciais

| Modelo | F1 macro | F1 aumento leve | Diferença | Acurácia | Checkpoint | Tempo | s/época |
|---|---|---|---|---|---|---|---|
| ResNet18 | 0,8209 | 0,9132 | −9,2 pts | 0,8238 | 12/22 | 27 s | 1,21 |
| ResNet18SE | 0,8481 | 0,8766 | −2,9 pts | 0,8486 | 9/19 | 27 s | 1,40 |
| ResNet18CBAM | 0,8753 | 0,9245 | −4,9 pts | 0,8762 | 22/32 | 65 s | 2,02 |
| ResNet18SECBAM | 0,8695 | 0,9265 | −5,7 pts | 0,8724 | 14/24 | 59 s | 2,46 |
| VGG16 | 0,9369 | 0,9488 | −1,2 pts | 0,9371 | 37/47 | 177 s | 3,76 |
| VGG16SE | 0,9201 | 0,9377 | −1,8 pts | 0,9210 | 25/35 | 147 s | 4,19 |
| VGG16CBAM | 0,9014 | 0,9444 | −4,3 pts | 0,9029 | 18/28 | 156 s | 5,57 |
| VGG16SECBAM | 0,9005 | 0,9464 | −4,6 pts | 0,9029 | 15/25 | 150 s | 6,02 |

Média da diferença: **−4,3 pts**, de −9,2 pts a −1,2 pts; ResNets −5,7 pts e VGGs −3,0 pts; melhorou em 0 dos 8.

### 13.3 128 × 128, sem aumento, hiperparâmetros do Optuna

| Modelo | F1 macro | F1 aumento leve | Diferença | Acurácia | Checkpoint | Tempo | s/época |
|---|---|---|---|---|---|---|---|
| ResNet18 | 0,8186 | 0,9164 | −9,8 pts | 0,8181 | 17/27 | 45 s | 1,67 |
| ResNet18SE | 0,8390 | 0,9372 | −9,8 pts | 0,8390 | 7/17 | 60 s | 3,52 |
| ResNet18CBAM | 0,8408 | 0,9349 | −9,4 pts | 0,8419 | 17/27 | 105 s | 3,89 |
| ResNet18SECBAM | 0,8534 | 0,9099 | −5,6 pts | 0,8552 | 21/31 | 150 s | 4,84 |
| VGG16 | 0,8987 | 0,9487 | −5,0 pts | 0,9000 | 11/21 | 82 s | 3,90 |
| VGG16SE | 0,9377 | 0,9433 | −0,6 pts | 0,9381 | 57/67 | 295 s | 4,41 |
| VGG16CBAM | 0,9037 | 0,9347 | −3,1 pts | 0,9048 | 22/32 | 188 s | 5,87 |
| VGG16SECBAM | 0,9130 | 0,9520 | −3,9 pts | 0,9133 | 25/35 | 223 s | 6,37 |

Média da diferença: **−5,9 pts**, de −9,8 pts a −0,6 pts; ResNets −8,7 pts e VGGs −3,1 pts; melhorou em 0 dos 8.

### 13.4 128 × 128, aumento forte, hiperparâmetros iniciais

| Modelo | F1 macro | F1 aumento leve | Diferença | Acurácia | Checkpoint | Tempo | s/época |
|---|---|---|---|---|---|---|---|
| ResNet18 | 0,9150 | 0,9132 | +0,2 pts | 0,9152 | 45/55 | 74 s | 1,35 |
| ResNet18SE | 0,8698 | 0,8766 | −0,7 pts | 0,8724 | 12/22 | 36 s | 1,61 |
| ResNet18CBAM | 0,8683 | 0,9245 | −5,6 pts | 0,8695 | 11/21 | 46 s | 2,21 |
| ResNet18SECBAM | 0,9122 | 0,9265 | −1,4 pts | 0,9133 | 41/51 | 134 s | 2,62 |
| VGG16 | 0,9469 | 0,9488 | −0,2 pts | 0,9476 | 37/47 | 186 s | 3,96 |
| VGG16SE | 0,9433 | 0,9377 | +0,6 pts | 0,9438 | 42/52 | 229 s | 4,41 |
| VGG16CBAM | 0,9501 | 0,9444 | +0,6 pts | 0,9505 | 52/62 | 358 s | 5,77 |
| VGG16SECBAM | 0,9312 | 0,9464 | −1,5 pts | 0,9324 | 40/50 | 311 s | 6,22 |

Média da diferença: **−1,0 pts**, de −5,6 pts a +0,6 pts; ResNets −1,9 pts e VGGs −0,1 pts; melhorou em 3 dos 8.

### 13.5 128 × 128, aumento forte, hiperparâmetros do Optuna

| Modelo | F1 macro | F1 aumento leve | Diferença | Acurácia | Checkpoint | Tempo | s/época |
|---|---|---|---|---|---|---|---|
| ResNet18 | 0,9120 | 0,9164 | −0,4 pts | 0,9124 | 40/50 | 103 s | 2,06 |
| ResNet18SE | 0,9320 | 0,9372 | −0,5 pts | 0,9324 | 58/68 | 210 s | 3,08 |
| ResNet18CBAM | 0,8989 | 0,9349 | −3,6 pts | 0,8990 | 36/46 | 198 s | 4,31 |
| ResNet18SECBAM | 0,9068 | 0,9099 | −0,3 pts | 0,9076 | 47/57 | 292 s | 5,12 |
| VGG16 | 0,9447 | 0,9487 | −0,4 pts | 0,9448 | 58/68 | 288 s | 4,24 |
| VGG16SE | 0,9355 | 0,9433 | −0,8 pts | 0,9362 | 41/51 | 243 s | 4,77 |
| VGG16CBAM | 0,9137 | 0,9347 | −2,1 pts | 0,9143 | 29/39 | 242 s | 6,20 |
| VGG16SECBAM | 0,9123 | 0,9520 | −4,0 pts | 0,9133 | 26/36 | 242 s | 6,74 |

Média da diferença: **−1,5 pts**, de −4,0 pts a −0,3 pts; ResNets −1,2 pts e VGGs −1,8 pts; melhorou em 0 dos 8.

### 13.6 256 × 256, sem aumento, hiperparâmetros iniciais

| Modelo | F1 macro | F1 aumento leve | Diferença | Acurácia | Checkpoint | Tempo | s/época |
|---|---|---|---|---|---|---|---|
| ResNet18 | 0,8832 | 0,9426 | −5,9 pts | 0,8848 | 14/24 | 67 s | 2,80 |
| ResNet18SE | 0,9179 | 0,9481 | −3,0 pts | 0,9171 | 33/43 | 130 s | 3,03 |
| ResNet18CBAM | 0,9222 | 0,9364 | −1,4 pts | 0,9229 | 43/53 | 204 s | 3,86 |
| ResNet18SECBAM | 0,9235 | 0,9262 | −0,3 pts | 0,9248 | 28/38 | 156 s | 4,11 |
| VGG16 | 0,9457 | 0,9570 | −1,1 pts | 0,9457 | 33/43 | 609 s | 14,16 |
| VGG16SE | 0,9531 | 0,9539 | −0,1 pts | 0,9533 | 55/65 | 1023 s | 15,74 |
| VGG16CBAM | 0,9096 | 0,9552 | −4,6 pts | 0,9095 | 11/21 | 432 s | 20,58 |
| VGG16SECBAM | 0,9158 | 0,9356 | −2,0 pts | 0,9162 | 30/40 | 887 s | 22,18 |

Média da diferença: **−2,3 pts**, de −5,9 pts a −0,1 pts; ResNets −2,7 pts e VGGs −1,9 pts; melhorou em 0 dos 8.

### 13.7 256 × 256, sem aumento, hiperparâmetros do Optuna

| Modelo | F1 macro | F1 aumento leve | Diferença | Acurácia | Checkpoint | Tempo | s/época |
|---|---|---|---|---|---|---|---|
| ResNet18 | 0,8729 | 0,9356 | −6,3 pts | 0,8733 | 15/25 | 75 s | 3,00 |
| ResNet18SE | 0,8998 | 0,9314 | −3,2 pts | 0,9000 | 17/27 | 114 s | 4,22 |
| ResNet18CBAM | 0,9102 | 0,9368 | −2,7 pts | 0,9124 | 26/36 | 199 s | 5,52 |
| ResNet18SECBAM | 0,9171 | 0,9326 | −1,5 pts | 0,9181 | 17/27 | 164 s | 6,05 |
| VGG16 | 0,9489 | 0,9579 | −0,9 pts | 0,9495 | 40/50 | 699 s | 13,99 |
| VGG16SE | 0,9550 | 0,9645 | −1,0 pts | 0,9552 | 40/50 | 781 s | 15,62 |
| VGG16CBAM | 0,9049 | 0,9590 | −5,4 pts | 0,9057 | 13/23 | 473 s | 20,56 |
| VGG16SECBAM | 0,9088 | 0,9541 | −4,5 pts | 0,9095 | 17/27 | 600 s | 22,22 |

Média da diferença: **−3,2 pts**, de −6,3 pts a −0,9 pts; ResNets −3,4 pts e VGGs −2,9 pts; melhorou em 0 dos 8.

### 13.8 256 × 256, aumento forte, hiperparâmetros iniciais

| Modelo | F1 macro | F1 aumento leve | Diferença | Acurácia | Checkpoint | Tempo | s/época |
|---|---|---|---|---|---|---|---|
| ResNet18 | 0,9514 | 0,9426 | +0,9 pts | 0,9514 | 48/58 | 183 s | 3,16 |
| ResNet18SE | 0,9375 | 0,9481 | −1,1 pts | 0,9381 | 40/50 | 177 s | 3,54 |
| ResNet18CBAM | 0,9274 | 0,9364 | −0,9 pts | 0,9286 | 33/43 | 180 s | 4,19 |
| ResNet18SECBAM | 0,9357 | 0,9262 | +0,9 pts | 0,9362 | 49/59 | 269 s | 4,56 |
| VGG16 | 0,9422 | 0,9570 | −1,5 pts | 0,9429 | 44/54 | 786 s | 14,56 |
| VGG16SE | 0,9240 | 0,9539 | −3,0 pts | 0,9238 | 20/30 | 484 s | 16,15 |
| VGG16CBAM | 0,9597 | 0,9552 | +0,4 pts | 0,9600 | 52/62 | 1300 s | 20,97 |
| VGG16SECBAM | 0,9525 | 0,9356 | +1,7 pts | 0,9524 | 42/52 | 1174 s | 22,58 |

Média da diferença: **−0,3 pts**, de −3,0 pts a +1,7 pts; ResNets 0,0 pts e VGGs −0,6 pts; melhorou em 4 dos 8.

### 13.9 256 × 256, aumento forte, hiperparâmetros do Optuna

| Modelo | F1 macro | F1 aumento leve | Diferença | Acurácia | Checkpoint | Tempo | s/época |
|---|---|---|---|---|---|---|---|
| ResNet18 | 0,9089 | 0,9356 | −2,7 pts | 0,9095 | 29/39 | 140 s | 3,58 |
| ResNet18SE | 0,9446 | 0,9314 | +1,3 pts | 0,9448 | 82/92 | 401 s | 4,36 |
| ResNet18CBAM | 0,9266 | 0,9368 | −1,0 pts | 0,9276 | 41/51 | 291 s | 5,70 |
| ResNet18SECBAM | 0,9165 | 0,9326 | −1,6 pts | 0,9171 | 29/39 | 252 s | 6,45 |
| VGG16 | 0,9542 | 0,9579 | −0,4 pts | 0,9543 | 46/56 | 813 s | 14,52 |
| VGG16SE | 0,9695 | 0,9645 | +0,5 pts | 0,9695 | 83/93 | 1501 s | 16,14 |
| VGG16CBAM | 0,9460 | 0,9590 | −1,3 pts | 0,9467 | 46/56 | 1182 s | 21,11 |
| VGG16SECBAM | 0,9532 | 0,9541 | −0,1 pts | 0,9533 | 46/56 | 1274 s | 22,75 |

Média da diferença: **−0,7 pts**, de −2,7 pts a +1,3 pts; ResNets −1,0 pts e VGGs −0,3 pts; melhorou em 2 dos 8.

### 13.10 Os 10 melhores treinos de CNN (validação, 96 treinos)

| # | Modelo | Res. | Aumento | Rodada | F1 macro | Desvio (10 classes) | Acurácia | Checkpoint | Tempo | s/época |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | VGG16SE | 256 | forte | Optuna | **0,9695** | 0,0233 | 0,9695 | 83/93 | 1501 s | 16,14 |
| 2 | VGG16SE | 256 | leve | Optuna | 0,9645 | 0,0310 | 0,9648 | 63/73 | 1152 s | 15,78 |
| 3 | VGG16CBAM | 256 | forte | inicial | 0,9597 | 0,0313 | 0,9600 | 52/62 | 1300 s | 20,97 |
| 4 | VGG16CBAM | 256 | leve | Optuna | 0,9590 | 0,0324 | 0,9590 | 76/86 | 1780 s | 20,70 |
| 5 | VGG16 | 256 | leve | Optuna | 0,9579 | 0,0314 | 0,9581 | 46/56 | 792 s | 14,15 |
| 6 | VGG16 | 256 | leve | inicial | 0,9570 | 0,0285 | 0,9571 | 46/56 | 802 s | 14,33 |
| 7 | VGG16CBAM | 256 | leve | inicial | 0,9552 | 0,0346 | 0,9552 | 40/50 | 1037 s | 20,73 |
| 8 | VGG16SE | 256 | nenhum | Optuna | 0,9550 | 0,0323 | 0,9552 | 40/50 | 781 s | 15,62 |
| 9 | VGG16 | 256 | forte | Optuna | 0,9542 | 0,0305 | 0,9543 | 46/56 | 813 s | 14,52 |
| 10 | VGG16SECBAM | 256 | leve | Optuna | 0,9541 | 0,0403 | 0,9543 | 52/62 | 1385 s | 22,35 |

Melhor treino de cada receita e resolução (qualquer rodada):

| Resolução | Sem aumento | Aumento leve | Aumento forte |
|---|---|---|---|
| 128 | VGG16SE Optuna (0,9377) | VGG16SECBAM Optuna (0,9520) | VGG16CBAM inicial (0,9501) |
| 256 | VGG16SE Optuna (0,9550) | VGG16SE Optuna (0,9645) | VGG16SE Optuna (0,9695) |

### 13.11 Leitura

**O que os números mostram (validação, uma execução por modelo)**

- **Sem aumento é sempre pior.** Em todos os 32 pares, o F1 sem aumento fica abaixo do aumento leve (−0,08 a −9,82 pontos; média de −3,92). As ResNets perdem mais (−5,10) que as VGGs (−2,75). Esse é o resultado mais firme desta seção: nenhum dos 32 pares inverte.
- **Aumento forte ficou, na média, atrás do leve** (−0,87 ponto, n = 32), mas com muita dispersão: piorou em 23 pares e melhorou em 9, de −5,62 a +1,69. Nas quatro combinações de resolução e rodada, a média do forte (0,9171, 0,9195, 0,9413, 0,9399) ficou abaixo da do leve (0,9273, 0,9346, 0,9444, 0,9465).
- **Mesmo assim, o melhor treino de CNN de todos usa aumento forte:** VGG16SE em 256 com forte e Optuna, F1 macro **0,9695** (acurácia 0,9695, checkpoint na época 83 de 93, 1.501 s). O 2º é a mesma arquitetura com aumento leve e Optuna (0,9645), **0,5 ponto abaixo**. Com um treino por modelo, essa diferença cabe no ruído de inicialização, então o que se pode afirmar é que as duas receitas empatam no topo, não que o forte seja melhor. O aumento forte também espalha mais os resultados (desvio entre os 8 modelos de 0,0207 em 256 com Optuna, contra 0,0136 com o leve), o que ajuda a produzir tanto os piores quanto o melhor.
- **Os 10 melhores treinos são todos VGGs em 256:** 6 com aumento leve, 3 com forte e 1 sem aumento (a VGG16SE com Optuna, 0,9550, 8º lugar). No topo, a receita de aumento importa menos que a resolução e a família.
- **A resolução reduz o custo de tirar o aumento.** Passar de 128 para 256 rendeu, na rodada inicial, em média +3,7 pontos sem aumento (8 dos 8 modelos melhoram), +2,4 com o forte e +1,7 com o leve (na rodada Optuna: +3,9, +2,0 e +1,2). Em consequência, a perda por tirar o aumento cai de −4,3 (inicial) e −5,9 (Optuna) em 128 para −2,3 e −3,2 em 256, e a do forte cai de −1,0 e −1,5 para −0,3 e −0,65. É uma associação nos dados executados. Uma explicação plausível (não testada) é que a imagem maior dá mais informação por exemplo e depende menos do aumento para não decorar o treino.
- **Os hiperparâmetros do Optuna não se transferem para outras receitas.** Optuna menos inicial, na média dos 8 modelos: aumento leve +0,74 (128) e +0,21 (256), que é a receita da busca; sem aumento −0,85 e −0,67 (melhorou em só 3 e 2 dos 8); forte +0,24 e −0,14 (melhorou em 2 e 4 dos 8). Para valer em outra receita, a busca teria de ser refeita com ela.
- **Sem aumento, o treino converge e para antes:** média de 33,8 épocas rodadas (checkpoint médio na época 23,8), contra 52,8 e 42,8 no leve e 52,0 e 42,0 no forte. Provavelmente a rede memoriza o treino mais cedo e o F1 de validação deixa de melhorar (explicação não testada). Os 64 treinos novos somaram 369,5 min de GPU (6,2 h) e os 32 com aumento leve 219,9 min (3,7 h).
- **A família e a atenção continuam como nas seções 5 a 7:** as VGGs ficam à frente das ResNets nas três receitas, e a VGG16SE segue entre as melhores (1ª em 256 com forte e Optuna, 1ª em 256 com aumento leve e Optuna e 1ª em 256 sem aumento com Optuna, 0,9550). A ordem entre as quatro variantes de cada família muda com a receita, o que reforça o peso do ruído de execução única.

**Cuidados na leitura**

- **Parada antecipada.** A paciência é de 10 épocas. Alguns treinos pararam muito cedo (checkpoint na época 12 ou antes: 5 sem aumento, 2 com forte e 1 com leve, por exemplo ResNet18SE em 128 sem aumento com Optuna, época 7, F1 0,8390). Parte do efeito de uma receita pode ser, na verdade, uma parada precoce por ruído, e não a receita em si. Seria preciso repetir com outras sementes ou paciência maior para separar as duas coisas.
- **Seleção na validação.** O checkpoint, a época de parada e os hiperparâmetros do Optuna foram escolhidos na validação. O F1 de validação fica ligeiramente otimista, e o melhor dos 112 treinos é o máximo de 112 números com ruído, portanto o mais otimista de todos. O desempenho de generalização só aparece no teste, avaliado uma vez ao final (ainda não feito).
- **O Optuna das receitas novas veio da busca com aumento leve** (acima), então "Optuna" nas receitas sem aumento e forte quer dizer "valores do Optuna do aumento leve".
- **O desvio das tabelas é entre as 10 classes** de cada modelo, não entre sementes.

### 13.12 Figuras

`src/graficos_cnn_aumentos.py` gera em `resultados/figuras/cnn/`:

- `aumentos_1_f1_por_receita.png`: F1 de cada modelo por receita, em cada resolução e rodada.
- `aumentos_2_efeito_medio.png`: efeito médio de cada receita contra o aumento leve.
- `aumentos_3_custo.png`: custo (tempo e épocas) por receita.

Cada treino tem também as curvas (`<nome>_curvas.png`) e a matriz de confusão (`matrizes/<nome>_confusao.png`).

---

## 14. Optuna refeito com o aumento forte

Na seção 13, os treinos "Optuna" com aumento forte usaram os hiperparâmetros da busca feita com aumento **leve**. Para testar se isso atrapalhava, a **busca foi refeita com o aumento forte** (`src/optuna_cnns_aumento_forte.py`), com a mesma configuração da busca original (seção 7): uma busca por família (ResNet18 e VGG16, versão sem atenção) e por resolução, 10 tentativas, no máximo 30 épocas por tentativa, TPE com semente 42 e poda pela mediana. Depois as 8 CNNs de cada resolução foram treinadas com os melhores valores achados (16 treinos novos, `<CNN>-<R>_aug2_optf_preliminar`; `optf` = Optuna refeito com forte). Tudo na mesma A100, só treino e validação, um treino por modelo.

**Valores achados** (tentativas em `resultados/optuna/tentativas_<família>-<R>_aug2.csv`):

| Busca (aumento forte) | Melhor tentativa | Melhor F1 na busca | lr | Weight decay | Lote | Concluídas / podadas | Busca com aumento leve (seção 7) |
|---|---|---|---|---|---|---|---|
| ResNet18, 128 | 0 | 0,8967 | 8,5e-5 | 6,4e-3 | 32 | 3 / 7 | mesmos valores (tentativa 0) |
| VGG16, 128 | 0 | 0,9170 | 8,5e-5 | 6,4e-3 | 32 | 4 / 6 | outros valores (lr 2,4e-5, wd 1,7e-6) |
| ResNet18, 256 | 1 | 0,9089 | 2,4e-5 | 1,7e-6 | 32 | 3 / 7 | mesmos valores (tentativa 1) |
| VGG16, 256 | 1 | 0,9086 | 2,4e-5 | 1,7e-6 | 32 | 3 / 7 | outros valores (lr 2,8e-5, wd 1,6e-5) |

Nas quatro buscas, a melhor tentativa foi uma das duas primeiras, e as demais foram podadas ou ficaram abaixo. Como todas as buscas usam a mesma semente, as primeiras tentativas sorteiam os mesmos valores, então com 3 ou 4 tentativas concluídas a busca praticamente escolhe entre os 3 ou 4 primeiros pontos sorteados.

### 14.1 128 × 128, aumento forte, hiperparâmetros do Optuna refeito com aumento forte

| Modelo | F1 macro (Optuna forte) | F1 aumento forte com Optuna do leve | Diferença | F1 aumento forte inicial | F1 aumento leve com Optuna | Acurácia | Checkpoint | Tempo | s/época |
|---|---|---|---|---|---|---|---|---|---|
| ResNet18 | 0,9120 | 0,9120 | 0,0 pts | 0,9150 | 0,9164 | 0,9124 | 40/50 | 101 s | 2,03 |
| ResNet18SE | 0,9320 | 0,9320 | 0,0 pts | 0,8698 | 0,9372 | 0,9324 | 58/68 | 199 s | 2,92 |
| ResNet18CBAM | 0,8989 | 0,8989 | 0,0 pts | 0,8683 | 0,9349 | 0,8990 | 36/46 | 195 s | 4,23 |
| ResNet18SECBAM | 0,9068 | 0,9068 | 0,0 pts | 0,9122 | 0,9099 | 0,9076 | 47/57 | 287 s | 5,04 |
| VGG16 | 0,9170 | 0,9447 | −2,8 pts | 0,9469 | 0,9487 | 0,9171 | 26/36 | 152 s | 4,24 |
| VGG16SE | 0,9163 | 0,9355 | −1,9 pts | 0,9433 | 0,9433 | 0,9171 | 29/39 | 186 s | 4,78 |
| VGG16CBAM | 0,9304 | 0,9137 | +1,7 pts | 0,9501 | 0,9347 | 0,9305 | 59/69 | 424 s | 6,15 |
| VGG16SECBAM | 0,9255 | 0,9123 | +1,3 pts | 0,9312 | 0,9520 | 0,9267 | 44/54 | 362 s | 6,71 |

Média dos 8: **0,9174** (aumento forte com Optuna do leve 0,9195; aumento forte inicial 0,9171; aumento leve com Optuna 0,9346). Diferença média contra o Optuna do leve: −0,2 pts; só nas VGGs: −0,4 pts.

### 14.2 256 × 256, aumento forte, hiperparâmetros do Optuna refeito com aumento forte

| Modelo | F1 macro (Optuna forte) | F1 aumento forte com Optuna do leve | Diferença | F1 aumento forte inicial | F1 aumento leve com Optuna | Acurácia | Checkpoint | Tempo | s/época |
|---|---|---|---|---|---|---|---|---|---|
| ResNet18 | 0,9089 | 0,9089 | 0,0 pts | 0,9514 | 0,9356 | 0,9095 | 29/39 | 135 s | 3,47 |
| ResNet18SE | 0,9446 | 0,9446 | 0,0 pts | 0,9375 | 0,9314 | 0,9448 | 82/92 | 370 s | 4,02 |
| ResNet18CBAM | 0,9266 | 0,9266 | 0,0 pts | 0,9274 | 0,9368 | 0,9276 | 41/51 | 281 s | 5,51 |
| ResNet18SECBAM | 0,9165 | 0,9165 | 0,0 pts | 0,9357 | 0,9326 | 0,9171 | 29/39 | 243 s | 6,23 |
| VGG16 | 0,9513 | 0,9542 | −0,3 pts | 0,9422 | 0,9579 | 0,9514 | 46/56 | 809 s | 14,45 |
| VGG16SE | 0,9694 | 0,9695 | 0,0 pts | 0,9240 | 0,9645 | 0,9695 | 93/103 | 1656 s | 16,08 |
| VGG16CBAM | 0,9396 | 0,9460 | −0,6 pts | 0,9597 | 0,9590 | 0,9400 | 41/51 | 1074 s | 21,06 |
| VGG16SECBAM | 0,9354 | 0,9532 | −1,8 pts | 0,9525 | 0,9541 | 0,9362 | 29/39 | 886 s | 22,71 |

Média dos 8: **0,9365** (aumento forte com Optuna do leve 0,9399; aumento forte inicial 0,9413; aumento leve com Optuna 0,9465). Diferença média contra o Optuna do leve: −0,3 pts; só nas VGGs: −0,7 pts.

**Leitura**

- **As 8 ResNets (4 em 128 e 4 em 256) repetiram exatamente os treinos da seção 13** (mesmo F1, mesma acurácia e mesma época de checkpoint), porque a busca refeita devolveu os mesmos valores que a busca com aumento leve. É uma segunda conferência de reprodutibilidade do treino na A100.
- **Só as VGGs mudaram, e não melhoraram na média:** −0,42 ponto em 128 e −0,68 em 256 contra o Optuna da busca com aumento leve. Em 128, a VGG16 e a VGG16SE pioram (−2,8 e −1,9) e as duas com CBAM melhoram (+1,7 e +1,3); em 256, as quatro pioram ou ficam iguais (de −1,8 a −0,0). Com um treino por modelo, cada diferença cabe em ruído de inicialização.
- **O melhor treino de CNN não mudou:** VGG16SE em 256 com aumento forte e Optuna da busca com aumento leve (F1 0,9695). Com o Optuna refeito, a mesma arquitetura fez **0,9694** (acurácia 0,9695, checkpoint na época 93 de 103), com lr, weight decay e épocas diferentes. Os dois valores são indistinguíveis, e o resultado de cerca de 0,97 para essa arquitetura e receita não depende do Optuna ter sido feito com um aumento ou com o outro.
- **Conclusão:** a hipótese de que a busca com aumento leve atrapalhava o aumento forte **não se confirmou**: refazer a busca não melhorou nada. As médias do aumento forte com Optuna refeito (0,9174 em 128 e 0,9365 em 256) ficam no mesmo nível das do forte com os valores do leve (0,9195 e 0,9399) e do forte inicial (0,9171 e 0,9413), e abaixo do aumento leve com Optuna (0,9346 e 0,9465).
- **Limite:** a busca é pequena (10 tentativas, 3 ou 4 concluídas). Não foi testado se uma busca maior (mais tentativas, mais épocas por tentativa, ou lr e lote em faixas mais estreitas) acharia algo melhor.

**Custo:** os 16 treinos somaram 122,7 minutos de GPU e as quatro buscas mais os treinos levaram cerca de 3 horas de A100 (cerca de 26 compute units).
