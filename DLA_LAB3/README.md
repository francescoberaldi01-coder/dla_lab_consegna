# DLA Lab 3 — Deep Reinforcement Learning

Il laboratorio sviluppa una pipeline progressiva di Deep Reinforcement Learning
su `CartPole-v1` e `LunarLander-v3`. Le implementazioni sono raccolte in moduli
riutilizzabili, mentre i notebook descrivono configurazione, addestramento,
valutazione periodica e analisi dei risultati.

## Struttura

- `professor/`: materiale originale del laboratorio;
- `src/`: reti neurali, algoritmi, valutazione, plotting e utilità comuni;
- `exercise_1_reinforce_cartpole/`: REINFORCE e valutazione periodica su CartPole;
- `exercise_2_value_baseline/`: confronto tra ritorni non standardizzati,
  standardizzati e baseline appresa `V(s)`;
- `exercise_3_1_a2c/`: A2C su CartPole e LunarLander, più una ricerca Optuna
  separata;
- `exercise_3_2_dql/`: DQN con replay buffer e target network su entrambi gli
  ambienti;
- `exercise_comparison/`: sintesi empirica e confronto tra gli algoritmi.

## Ordine di esecuzione

1. `exercise_1_reinforce_cartpole/01_reinforce_cartpole.ipynb`
2. `exercise_2_value_baseline/01_reinforce_value_baseline.ipynb`
3. `exercise_3_1_a2c/01_a2c_cartpole.ipynb`
4. `exercise_3_1_a2c/02_a2c_lunarlander.ipynb`
5. `exercise_3_1_a2c/03_a2c_lunarlander_hyperparameter_search.ipynb` *(extra)*
6. `exercise_3_2_dql/01_dql_cartpole.ipynb`
7. `exercise_3_2_dql/02_dql_lunarlander.ipynb`
8. `exercise_comparison/01_algorithms_comparison.ipynb`

Ogni notebook carica la propria configurazione YAML e può essere eseguito
indipendentemente, purché la repository sia disponibile nel percorso di lavoro.
Il notebook di confronto usa i risultati già consolidati negli output salvati.

## Metodologia comune

- seed multipli per misurare la variabilità del training;
- ambienti di training e valutazione separati;
- valutazione deterministica periodica ogni `N` episodi su `M` episodi;
- selezione in memoria del checkpoint con reward medio di valutazione migliore;
- valutazione finale su un ambiente inizializzato con un seed distinto;
- reward medio, minimo, massimo e lunghezza media degli episodi;
- rendering conclusivo come controllo qualitativo, senza aggiornamento dei pesi.

Nel Reinforcement Learning il reward di training è influenzato dall'esplorazione
e non coincide necessariamente con la qualità della policy deterministica. Per
questo le conclusioni si basano soprattutto sulle valutazioni periodiche e
finali, non sull'ultimo reward osservato durante il training.

## Risultati principali

| Ambiente | Algoritmo | Seed | Reward medio finale | Deviazione standard |
|---|---|---:|---:|---:|
| CartPole | REINFORCE senza standardizzazione | 5 | 401.2 | 106.1 |
| CartPole | REINFORCE standard | 5 | 498.6 | 3.1 |
| CartPole | REINFORCE + value baseline | 5 | 499.9 | 0.1 |
| CartPole | A2C | 5 | 402.2 | 156.0 |
| CartPole | DQN | 5 | 494.6 | 12.0 |
| LunarLander | A2C | 3 | -70.4 | 53.6 |
| LunarLander | DQN | 3 | 274.0 | 11.1 |

REINFORCE standardizzato e la variante con value baseline risolvono CartPole in
modo molto stabile. A2C raggiunge il massimo in tre seed su cinque ma mostra una
forte sensibilità all'inizializzazione; su LunarLander la configurazione di base
non raggiunge una soluzione stabile. DQN risolve CartPole e supera stabilmente
la soglia di reward 200 su LunarLander in tutti e tre i seed valutati.

## Dipendenze

Le dipendenze sono definite nei file `requirements-windows.txt` e
`requirements-mac.txt` nella root. Il Lab 3 usa in particolare PyTorch,
Gymnasium, PyGame, Box2D, Matplotlib, pandas, PyYAML e, per il notebook extra,
Optuna.

CartPole funziona interamente su CPU. LunarLander richiede le dipendenze Box2D;
su macOS possono essere necessari anche gli strumenti di compilazione di
sistema. Le configurazioni incluse forzano la CPU, scelta adeguata per le piccole
reti MLP e gli ambienti usati nel laboratorio.

## Riproducibilità

I file YAML conservano seed, budget di episodi, learning rate, fattori di sconto,
frequenza di valutazione e parametri specifici degli algoritmi. I checkpoint
migliori sono mantenuti in memoria durante ciascuna esecuzione: i notebook
salvati documentano i risultati, ma per riutilizzare un agente in una nuova
sessione è necessario rieseguire il relativo training.
