"""Refaz a busca do Optuna das CNNs COM O AUMENTO FORTE e treina as 8 CNNs com os valores achados, numa resolução.

Nas variações de aumento (src/rodar_cnns_aumentos.py) os treinos "Optuna" das receitas sem aumento e aumento forte
usaram os hiperparâmetros da busca feita com aumento LEVE. Aqui a busca é refeita com o aumento forte (o mesmo das
MLPs: leve + recorte com zoom + variação de cor + apagamento aleatório), na mesma configuração da busca original
(notebooks/03_CNNs.ipynb, seção do Optuna): uma busca por família (ResNet18 e VGG16, sem atenção), 10 tentativas, no
máximo 30 épocas por tentativa, TPE com semente 42 e poda pela mediana. Depois, as 8 CNNs são treinadas com os melhores
valores de cada família (os de ResNet18 valem para as 4 ResNets e os de VGG16 para as 4 VGGs).

Saídas (nada do que já existe é sobrescrito):
  resultados/optuna/cnn_aug2.db                                  estudos (SQLite)
  resultados/optuna/tentativas_<família>-<R>_aug2.csv            todas as tentativas
  resultados/optuna/melhores_hiperparametros_cnn_aug2_<R>.json   melhores valores por família
  modelos/<CNN>-<R>_aug2_optf_preliminar.pt e os arquivos de resultados do treino ("optf" = Optuna refeito com forte)

Só usa treino e validação (o teste não é usado). A busca é retomável: o estudo guarda as tentativas já feitas, e treinos
já concluídos são pulados.

Uso (a partir da raiz do projeto, na VM com GPU): python src/optuna_cnns_aumento_forte.py <128|256> [raiz]
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd

RES = int(sys.argv[1])
assert RES in (128, 256), "resolução deve ser 128 ou 256"
RAIZ = Path(sys.argv[2] if len(sys.argv) > 2 else Path(__file__).resolve().parent.parent)
os.environ.setdefault("MPLBACKEND", "Agg")
os.chdir(RAIZ / "notebooks")

nb = json.loads((RAIZ / "notebooks" / "03_CNNs.ipynb").read_text(encoding="utf-8"))
nb_mlp = json.loads((RAIZ / "notebooks" / "02_mlp_preliminar.ipynb").read_text(encoding="utf-8"))


def celula(notebook, i):
    c = notebook["cells"][i]
    assert c["cell_type"] == "code", i
    return "".join(c["source"])


ns = {"__name__": "__main__"}
cel7 = celula(nb, 7)
assert "RESOLUCAO = 128 " in cel7
cel7 = cel7.replace("RESOLUCAO = 128 ", f"RESOLUCAO = {RES} ", 1)
# arquiteturas (3 e 5), configuração e semente (7), dados e aumento leve (9), laço de treino (11)
for origem, codigo in (("3", celula(nb, 3)), ("5", celula(nb, 5)), ("7", cel7), ("9", celula(nb, 9)), ("11", celula(nb, 11))):
    exec(compile(codigo, f"<celula {origem}>", "exec"), ns)

# aumento forte: a função das MLPs (célula 10 do notebook 02), aplicada depois do aumento leve
cel10 = celula(nb_mlp, 10)
forte = cel10[cel10.index("def aumentar_forte"):cel10.index("def treinar(")]
exec(compile(forte, "<aumentar_forte>", "exec"), ns)
import optuna  # noqa: E402
from optuna.pruners import MedianPruner  # noqa: E402
from optuna.samplers import TPESampler  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)
ns["optuna"] = optuna
ns["DESCRICOES_AUMENTO"]["forte"] = (
    "espelhamentos e rotações de 90°, recorte com zoom (50-100% da área), variação de cor (brilho, contraste e saturação ±20%) "
    "e apagamento aleatório (p = 0,5; 10-40% do lado), só no treino")
aumentar_leve = ns["aumentar"]


def aumentar_composto(x, g):
    return ns["aumentar_forte"](aumentar_leve(x, g), g)


ns["aumentar"] = aumentar_composto
treinar_cnn, liberar_memoria, plotar_curvas_cnn, resultado_salvo = (
    ns["treinar_cnn"], ns["liberar_memoria"], ns["plotar_curvas_cnn"], ns["resultado_salvo"])

N_TRIALS, EPOCAS_TRIAL, SEMENTE = 10, 30, ns["SEMENTE"]          # os mesmos da busca original
ESTUDOS = {"ResNet18": ns["ResNet18"], "VGG16": ns["VGG16"]}     # uma busca por família, na versão sem atenção
DIR_OPTUNA = RAIZ / "resultados" / "optuna"
DIR_OPTUNA.mkdir(parents=True, exist_ok=True)


def criar_objetivo(Modelo):
    def objetivo(trial):
        hp = dict(lr=trial.suggest_float("lr", 1e-5, 3e-3, log=True),
                  weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
                  lote=trial.suggest_categorical("lote", [32, 64, 128]))
        _, hist = treinar_cnn(Modelo, Modelo.__name__, f"{Modelo.__name__}-{RES}_aug2_trial{trial.number}", aumento="forte",
                              lr=hp["lr"], weight_decay=hp["weight_decay"], lote=hp["lote"], max_epocas=EPOCAS_TRIAL,
                              trial=trial, salvar=False)
        liberar_memoria()
        return float(hist["f1_macro_validacao"].max())
    return objetivo


def mostrar_tentativa(familia):
    def callback(estudo, t):
        valor = f"{t.value:.4f}" if t.value is not None else "  -   "
        melhor = f"{estudo.best_value:.4f}" if any(x.state.name == "COMPLETE" for x in estudo.trials) else "  -   "
        print(f"[{familia}] tentativa {t.number:2d} | {t.state.name:<8} | F1 val {valor} | melhor {melhor} | "
              f"lr {t.params['lr']:.2e}  wd {t.params['weight_decay']:.2e}  lote {t.params['lote']}", flush=True)
    return callback


# 1) busca do Optuna com aumento forte
estudos = {}
for familia, Modelo in ESTUDOS.items():
    estudo = optuna.create_study(
        study_name=f"{familia}-{RES}-aug2", direction="maximize", sampler=TPESampler(seed=SEMENTE),
        pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=5),
        storage=f"sqlite:///{(DIR_OPTUNA / 'cnn_aug2.db').as_posix()}", load_if_exists=True)
    faltam = N_TRIALS - len(estudo.trials)
    print(f"=== Estudo {familia}-{RES} (aumento forte): {len(estudo.trials)} tentativas já feitas, rodando mais {max(faltam, 0)} ===", flush=True)
    if faltam > 0:
        estudo.optimize(criar_objetivo(Modelo), n_trials=faltam, callbacks=[mostrar_tentativa(familia)])
    estudos[familia] = estudo
    estudo.trials_dataframe().to_csv(DIR_OPTUNA / f"tentativas_{familia}-{RES}_aug2.csv", index=False)

melhores = {f: dict(e.best_params) for f, e in estudos.items()}
(DIR_OPTUNA / f"melhores_hiperparametros_cnn_aug2_{RES}.json").write_text(
    json.dumps({f"{f}-{RES}": {**hp, "melhor_f1_macro_validacao": estudos[f].best_value} for f, hp in melhores.items()},
               indent=2, ensure_ascii=False), encoding="utf-8")
print("BUSCA TERMINOU:", {f: {k: (round(v, 8) if isinstance(v, float) else v) for k, v in hp.items()} for f, hp in melhores.items()}, flush=True)

# 2) as 8 CNNs com os melhores valores de cada família
MODELOS = {a: ns[a] for a in ("ResNet18", "ResNet18SE", "ResNet18CBAM", "ResNet18SECBAM",
                              "VGG16", "VGG16SE", "VGG16CBAM", "VGG16SECBAM")}
for arq, Modelo in MODELOS.items():
    familia = "ResNet18" if arq.startswith("ResNet") else "VGG16"
    hp = melhores[familia]
    nome = f"{arq}-{RES}_aug2_optf_preliminar"
    if resultado_salvo(nome) is not None:
        print(f"##### {nome}: já treinado, pulando", flush=True)
        continue
    print(f"##### {nome} | aumento forte | Optuna refeito com forte | lr {hp['lr']:.2e} wd {hp['weight_decay']:.2e} lote {hp['lote']}", flush=True)
    resumo, hist = treinar_cnn(
        Modelo, arq, nome, aumento="forte", lr=hp["lr"], weight_decay=hp["weight_decay"], lote=hp["lote"],
        max_epocas=ns["MAX_EPOCAS_CNN"], tipo="preliminar_optuna",
        extras={"familia": familia, "busca": "optuna (refeita com aumento forte)", "estudo": f"{familia}-{RES}-aug2",
                "receita_aumento": "forte"})
    plotar_curvas_cnn(hist, nome)
    liberar_memoria()
print(f"OPTUNA FORTE {RES} TERMINOU", flush=True)
