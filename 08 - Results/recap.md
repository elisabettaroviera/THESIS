# RECAP – FEATURE SELECTION & CLASSIFICATION

*(Normal vs Adjacent)*

---

# 1. Pipeline – Prove effettuate

## TRY1 – Full β-based pipeline

$$  
\mathcal{J}
\xrightarrow[\text{train-only}]{\text{Edgar }(\beta)}
\mathcal{J}_1
\xrightarrow{\text{SD}(\beta)}
\mathcal{J}_2
\xrightarrow{\text{MWU}(\beta)+\text{FDR},\ |\Delta\beta|}
\mathcal{J}_3
\xrightarrow{\text{corr prune}(\beta)}
\mathcal{J}_4
\xrightarrow{\text{Elastic Net rank}(\beta)}
\mathcal{J}_5
$$

* Feature selection interamente su β
* Classificatore: Linear SVM
* Split train/test interno
* Cross-dataset evaluation

---

## TRY2 – β → M transformation

$$
\mathcal{J}
\xrightarrow[\text{train-only}]{\text{Edgar }(\beta)}
\mathcal{J}_1
\xrightarrow{\beta \rightarrow M}
\mathcal{J}_1
\xrightarrow{\text{SD}(M)}
\mathcal{J}_2
\xrightarrow{\text{MWU}(M)+\text{FDR},\ |\Delta\beta|}
\mathcal{J}_3
\xrightarrow{\text{corr prune}(M)}
\mathcal{J}_4
\xrightarrow{\text{Elastic Net rank}(M)}
\mathcal{J}_5
$$

Differenza principale:

* ranking e classificazione in M-space
* gate biologico mantenuto su |Δβ|

---

## TRY3 – Ranking + cluster su correlazione

$$
\begin{aligned}
\mathcal{J}
&\xrightarrow[\text{train-only}]{\text{Edgar }(\beta)}
\mathcal{J}_1
\xrightarrow{\beta \rightarrow M}
\mathcal{J}_1
\xrightarrow{\text{SD}(M)}
\mathcal{J}_2
\xrightarrow{\text{MWU}(M)+\text{FDR}, |\Delta\beta|}
\mathcal{J}_3 \
&\xrightarrow{\text{ranking } r_j}
\text{top-}K
\xrightarrow{\text{corr cluster}}
\text{1 CpG per componente}
\xrightarrow{\text{Elastic Net rank}}
\mathcal{J}_5
\end{aligned}
$$

Obiettivo:

* ridurre ridondanza
* forzare diversità informativa
* aumentare stabilità cross-dataset

---

# TRY4 – mRMR-based pipeline (Relevance + Redundancy control)

$$
\mathcal{J}
\xrightarrow[\text{train-only}]{\text{Edgar }(\beta)}
\mathcal{J}_1
\xrightarrow{\beta\rightarrow M}
\mathcal{J}_1
\xrightarrow{\text{SD}(M)}
\mathcal{J}_2
\xrightarrow[\text{train-only}]{\text{MWU}(M)\ (\text{relevance})\ +\ \text{prefilter top-}P}
\widetilde{\mathcal{J}}_2
\xrightarrow[\text{train-only}]{\text{mRMR}(M)\ (\text{redundancy via }|\mathrm{corr}|)}
\mathcal{J}_3
\xrightarrow[\text{train-only}]{\text{Elastic Net rank}(M)}
\mathcal{J}_4
$$

### Logica

1. **Relevance**: MWU su M-values per catturare segnale discriminante.
2. **Prefilter top-P**: riduzione dimensionalità prima della ridondanza.
3. **mRMR**: selezione sequenziale massimizzando:
   [
   \max_j \left( \text{Rel}_j - \lambda \cdot \text{Red}_j \right)
   ]
   dove la ridondanza è misurata tramite correlazione assoluta.

Obiettivo:
forzare informazione complementare e ridurre cluster di CpG altamente correlate.

---

## TRY7 – Region-first (Island aggregation)

$$
\mathcal{J}
\xrightarrow[\text{train-only}]{\text{Edgar}}
\mathcal{J}_1
\xrightarrow{\beta\rightarrow M}
\mathcal{J}*1
\xrightarrow{\text{Agg}*{region}}
\mathcal{R}
\xrightarrow{\text{MWU+FDR}}
\mathcal{R}^\star
\xrightarrow{\text{expand to CpG}}
\mathcal{J}_2
\xrightarrow{\text{corr prune}}
\mathcal{J}_3
\xrightarrow{\text{Elastic Net}}
\mathcal{J}_4
$$

Idea:

* selezionare prima regioni (islands)
* poi riespandere a CpG
* migliorare stabilità inter-dataset

---

# 2. Risultati – Accuracy

## TRY1 – MWU + FDR (β/M)

| Train \ Test | GSE69914 | GSE225845 | GSE287331 |
| ------------ | -------- | --------- | --------- |
| **69914**    | 0.62     | 0.61      | 0.21      |
| **225845**   | 0.50     | 0.89      | 0.03      |
| **287331**   | 0.47     | 0.49      | 1.00      |

Osservazioni:

* buona performance intra-dataset
* forte crollo cross-dataset
* 287331 altamente separabile ma non generalizza

---

## TRY2 – β→M migliorato

| Train \ Test | GSE69914 | GSE225845 | GSE287331 |
| ------------ | -------- | --------- | --------- |
| **69914**    | 0.64     | 0.66      | 0.37      |
| **225845**   | 0.49     | 0.88      | 0.06      |
| **287331**   | 0.41     | 0.42      | 1.00      |

M-space migliora leggermente cross da 69914.


---

## TRY3 – Corr clustering

| Train \ Test | GSE69914 | GSE225845 | GSE287331 |
| ------------ | -------- | --------- | --------- |
| **69914**    | 0.61     | 0.67      | 0.36      |
| **225845**   | 0.48     | 0.89      | 0.06      |
| **287331**   | 0.42     | 0.41      | 1.00      |

Clustering riduce ridondanza ma non risolve instabilità cross.

---

---

## TRY4 – mRMR

| Train \ Test | GSE69914 | GSE225845 | GSE287331 |
| ------------ | -------- | --------- | --------- |
| **69914**    | 0.65     | 0.66      | 0.38      |
| **225845**   | 0.49     | 0.88      | 0.06      |
| **287331**   | 0.41     | 0.41      | 1.00      |

L’introduzione esplicita del controllo di ridondanza tramite mRMR non migliora la generalizzazione cross-dataset, indicando che l’instabilità non dipende dalla correlazione interna tra CpG ma da differenze strutturali tra coorti.

---

## TRY7 – Island-first

| Train \ Test | GSE69914 | GSE225845 | GSE287331 |
| ------------ | -------- | --------- | --------- |
| **69914**    | 0.55     | 0.56      | 0.23      |
| **225845**   | 0.53     | 0.92      | 0.10      |
| **287331**   | 0.42     | 0.34      | 1.00      |

Island migliora leggermente stabilità quando train=225845.

---

# 3. Analisi critica

### 3.1 Pattern evidente

* Ogni dataset impara struttura propria
* 287331 ha separabilità intrinseca forte
* Cross-dataset molto instabile
* Le CpG selezionate sono altamente dataset-specific

---

### 3.2 Problema reale

Il problema non è overfitting classico.
È **instabilità strutturale delle feature selezionate**.

Le firme sono diverse tra dataset.

---

# 4. Idee operative

## a) Passaggio a Island-level

Motivazione:

* Island più stabili biologicamente
* Meno rumore CpG-specific
* Migliore trasferibilità

Sto cercando la strategia ottimale:

* mean aggregation
* MWU su island
* back-expansion controllata

---

## b) Stabilità via k-fold

Obiettivo:

* misurare frequenza selezione CpG
* definire stability score
* tenere solo CpG con stabilità > τ

Formalmente:

$$
\text{stab}*j = \frac{1}{K} \sum*{k=1}^K \mathbf{1}(j \in \mathcal{S}^{(k)})
$$

---

## c) Migliorare cross-dataset

Possibili direzioni:

1. Intersection-based FS
2. Multi-dataset training (leave-one-dataset-out)
3. Penalizzazione instabilità
4. Formulazione knapsack non banale con:

   * cardinalità
   * vincolo correlazione
   * bilanciamento segno

---

# 5. Boundary Plot

Mostrare:

* PCA 2D
* hyperplane calibrato
* probabilità SVM
* evidenziare sovrapposizione cross

---

# 6. Passaggio a livello gene

Prossimo step:

* Mappare CpG selezionate → gene
* Analizzare:

  * concentrazione su pochi geni?
  * hub-like behavior?
  * pathway enrichment?

Questo serve per capire se:

* la selezione è biologicamente coerente
* o puramente statistica

---

aws s3 ls s3://cpgpt-lucascamillo-public/data/cpgcorpus/raw/GSE69914/ --request-payer requester
aws s3 ls s3://cpgpt-lucascamillo-public/data/cpgcorpus/raw/GSE225845/ --request-payer requester
aws s3 ls s3://cpgpt-lucascamillo-public/data/cpgcorpus/raw/GSE287331/ --request-payer requester
