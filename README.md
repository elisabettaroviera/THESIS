# THESIS  
## Advanced study of epigenetic mechanisms in the development of neoplasms  

### 👩‍🎓 Author & Supervision
**Author:** Elisabetta Roviera  \
**Thesis title:** *Advanced study of epigenetic mechanisms in the development of neoplasms*  \
**Supervisor:** Dr Alfredo Benso (Polytechnic University of Turin) \
**Co-supervisor:** Dr Sandro Gambino (Rivoli Hospital) \
**Institution:** Polytechnic University of Turin \
**Academic Year:** 2025-2026

---

### 🧬 Overview
This repository contains the materials, code, and references developed for the Master's thesis:  
**“Advanced study of epigenetic mechanisms in the development of neoplasms.”**

The work aims to explore the relationship between **epigenetic alterations** (such as DNA methylation patterns) and **tumorigenesis**, combining biological literature review with computational analysis.

---

### 🧠 Thesis Structure

The overall structure of the thesis is still being refined as the research evolves, but the main framework is already outlined. The project currently follows these key components:

1. **Biological and Medical Introduction** – providing the theoretical and biomedical background necessary to contextualize the problem from a clinical perspective.
2. **Dataset Description and Data Exploration** – presenting the dataset (GSE69914) and performing an exploratory analysis to understand its characteristics and potential biases.
3. **Data Pre-processing** – applying statistical and normalization techniques to ensure data quality and comparability across samples.
4. **Algorithmic Analysis** – implementing statistical and computational methods to identify biologically significant CpG sites associated with breast cancer.
5. **Work in Progress** – further model refinement, interpretation of results, and integration with biological insights will follow.

---


### 📂 Repository structure

#### **01 - Notebook**
[01 - Notebook](./01%20-%20Notebook)  
This folder contains all the Jupyter notebooks used for data processing, analysis, and visualization.  

- **[01 - Data exploration](./01%20-%20Notebook/01-data-exploration.ipynb)** → Initial data inspection and global methylation level analysis  
- **[02 - Data pre-processing](./01%20-%20Notebook/02-data-pre-processing.ipynb)** → Initial data pre-processing (work in progress...)
- *(More notebooks will be added as the analysis progresses)*

#### **02 - Paper**

[02 - Paper](./02%20-%20Paper)
This folder includes all the scientific papers and documents used to guide the analysis and support the writing of the thesis chapters.

**🧾 Notes**
All papers included are used **for academic and educational purposes only**.
Code and analyses are developed using open-access datasets and tools.

##### [**01 - Medical and Biological Information**](./02%20-%20Paper/01%20-%20Medical%20and%20Biological%20information)

Contains the main references on cancer biology and epigenetics:

0. [Medical and Biological Information (recap).pdf](./02%20-%20Paper/01%20-%20Medical%20and%20Biological%20information/Medical%20and%20Biological%20Information.pdf)
1. [A comprehensive methylome map of lineage commitment from hematopoietic progenitors – Ji.pdf](./02%20-%20Paper/01%20-%20Medical%20and%20Biological%20information/A%20comprehensive%20methylome%20map%20of%20lineage%20commitment%20from%20hematopoietic%20progenitors%20-%20Ji.pdf)
2. [BRCA Gene Changes: Cancer Risk and Genetic Testing Fact Sheet – NCI.pdf](./02%20-%20Paper/01%20-%20Medical%20and%20Biological%20information/BRCA%20Gene%20Changes_%20Cancer%20Risk%20and%20Genetic%20Testing%20Fact%20Sheet%20-%20NCI.pdf)
3. [Breast Cancer – StatPearls – NCBI Bookshelf.pdf](./02%20-%20Paper/01%20-%20Medical%20and%20Biological%20information/Breast%20Cancer%20-%20StatPearls%20-%20NCBI%20Bookshelf.pdf)
4. [Cancer Biology, Epidemiology, and Treatment in the 21st Century – Piña-Sanchez.pdf](./02%20-%20Paper/01%20-%20Medical%20and%20Biological%20information/Cancer%20Biology,%20Epidemiology,%20and%20Treatment%20in%20the%2021st%20Century%20-%20Piña-Sanchez.pdf)
5. [DNA methylation in cancer: too much but also too little – Ehrlich.pdf](./02%20-%20Paper/01%20-%20Medical%20and%20Biological%20information/DNA%20methylation%20in%20cancer%20too%20much%20but%20also%20too%20little%20-%20Ehrlich.pdf)
6. [DNA methylation patterns and epigenetic memory – Adrian Bird.pdf](./02%20-%20Paper/01%20-%20Medical%20and%20Biological%20information/DNA%20methylation%20patterns%20and%20epigenetic%20memory%20-%20Adrian%20Bird.pdf)
7. [Introduction to cancer biology – Azizi.pdf](./02%20-%20Paper/01%20-%20Medical%20and%20Biological%20information/Introduction%20to%20cancer%20biology%20-%20Azizi.pdf)
8. [The CpG Landscape of Protein Coding DNA in Vertebrates – Wilcox, Ord, Kappei, Gossmann.pdf](./02%20-%20Paper/01%20-%20Medical%20and%20Biological%20information/The%20CpG%20Landscape%20of%20Protein%20Coding%20DNA%20in%20Vertebrates%20-%20Wilcox,%20Ord,%20Kappei,%20Gossmann.pdf)
9. [The Hallmarks of Cancer – Hanahan, Weinberg.pdf](./02%20-%20Paper/01%20-%20Medical%20and%20Biological%20information/The%20Hallmarks%20of%20Cancer%20-%20Hanahan,%20Weinberg.pdf)
10. [Understanding Cancer – NIH Curriculum Supplement Series – NCBI Bookshelf.pdf](./02%20-%20Paper/01%20-%20Medical%20and%20Biological%20information/Understanding%20Cancer%20-%20NIH%20Curriculum%20Supplement%20Series%20-%20NCBI%20Bookshelf.pdf)


##### **02 - GSE67915 Application**

Papers focused on data analysis, methodology, and applications of methylation arrays (including the dataset GSE69914):

1. [GSE67915 Application.pdf](./02%20-%20Paper/02%20-%20GSE67915%20Application/GSE67915%20Application.pdf)
2. [An improved epigenetic computer to track mitotic age in normal and precancerous tissues – Zhu.pdf](./02%20-%20Paper/02%20-%20GSE67915%20Application/Zhu.pdf)
3. [An integrative pan-cancer-wide analysis epigenetic enzymes reveals universal cancer – Yang.pdf](./02%20-%20Paper/02%20-%20GSE67915%20Application/Yang.pdf)
4. [DNA methylation outliers in normal breast tissue identify field defects that are enriched in cancer – Teschendorff.pdf](./02%20-%20Paper/02%20-%20GSE67915%20Application/Teschendorff.pdf)
5. [DNA Methylation Patterns in Normal Tissue Correlate More Strongly with Breast Cancer Status than Copy-Number Variants – Gao.pdf](./02%20-%20Paper/02%20-%20GSE67915%20Application/Gao.pdf)
6. [EPISCORE: cell type deconvolution of bulk tissue DNA methylomes from single-cell RNA-Seq data – Teschendorff.pdf](./02%20-%20Paper/02%20-%20GSE67915%20Application/EPISCORE%20-%20Teschendorff.pdf)
7. [Genome-wide discovery of circulating cell-free DNA methylation signatures for triple-negative breast cancer – Gao.pdf](./02%20-%20Paper/02%20-%20GSE67915%20Application/Gao%20cfDNA.pdf)
8. [Inference of tissue relative proportions of the breast epithelial cell types – Bartlett.pdf](./02%20-%20Paper/02%20-%20GSE67915%20Application/Bartlett.pdf)
9. [Integrative analysis identifies potential DNA methylation biomarkers for pancreatic cancer – Ding.pdf](./02%20-%20Paper/02%20-%20GSE67915%20Application/Ding.pdf)
10. [Large-scale analysis of DNA methylation reveals its potential as biomarker for breast cancer – Croes.pdf](./02%20-%20Paper/02%20-%20GSE67915%20Application/Croes.pdf)
11. [Prediction of Disease Genes based on Stage-Specific Gene Regulatory Networks in Breast Cancer – Fan.pdf](./02%20-%20Paper/02%20-%20GSE67915%20Application/Fan.pdf)
12. [The integrative epigenomic–transcriptomic landscape of ER positive breast cancer – Gao.pdf](./02%20-%20Paper/02%20-%20GSE67915%20Application/ERpositive%20Gao.pdf)


##### [**03 - Data Exploration & Visualization**](./02%20-%20Paper/03%20-%20Data%20EXPLORATION%20%26%20VISUALIZATION)

Includes exploratory analyses and visualization strategies related to DNA methylation and outlier detection.

0. [Data_Visualization_Information.pdf](./02%20-%20Paper/03%20-%20Data%20EXPLORATION%20%26%20VISUALIZATION/Data_Visualization_Information.pdf)
1. [A urine-based DNA methylation assay, ProCUrE, to identify clinically significant prostate cancer – Zhao.pdf](./02%20-%20Paper/03%20-%20Data%20EXPLORATION%20%26%20VISUALIZATION/A%20urine-based%20DNA%20methylation%20assay,%20ProCUrE,%20to%20identify%20clinically%20significant%20prostate%20cancer%20-%20Zhao.pdf)
2. [Charting differentially methylated regions in cancer with Rocker-meth – Benelli, Franceschini, Magi, Romagnoli, Biagioni, Migliaccio, Malorni, Demichelis.pdf](./02%20-%20Paper/03%20-%20Data%20EXPLORATION%20%26%20VISUALIZATION/Charting%20differentially%20methylated%20regions%20in%20cancer%20with%20Rocker-meth%20%E2%80%93%20Benelli%2C%20Franceschini%2C%20Magi%2C%20Romagnoli%2C%20Biagioni%2C%20Migliaccio%2C%20Malorni%2C%20Demichelis.pdf)
3. [DNA methylation outlier burden, health, and ageing in Generation Scotland and the Lothian Birth Cohorts of 1921 and 1936 - Anne Seeboth, McCartney.pdf](./02%20-%20Paper/03%20-%20Data%20EXPLORATION%20%26%20VISUALIZATION/DNA%20methylation%20outlier%20burden%2C%20health%2C%20and%20ageing%20in%20Generation%20Scotland%20and%20the%20Lothian%20Birth%20Cohorts%20of%201921%20and%201936%20-%20Anne%20Seeboth%2C%20McCartney.pdf)
4. [SWAN Subset-quantile Within Array Normalization for Illumina Infinium HumanMethylation450 BeadChips - Maksimovic, Gordon, Oshlack .pdf](./02%20-%20Paper/03%20-%20Data%20EXPLORATION%20%26%20VISUALIZATION/SWAN%20Subset-quantile%20Within%20Array%20Normalization%20for%20Illumina%20Infinium%20HumanMethylation450%20BeadChips%20-%20Maksimovic%2C%20Gordon%2C%20Oshlack%20.pdf)






##### **04 - Data Pre-Processing**

Papers and references used to guide data normalization, preprocessing, and methylation array correction strategies.

1. [A framework for analyzing DNA methylation data – Wang.pdf](./02%20-%20Paper/04%20-%20Data%20PRE-PROCESSING/Wang.pdf)
2. [A statistical model for methylation β-values – Weinhold.pdf](./02%20-%20Paper/04%20-%20Data%20PRE-PROCESSING/Weinhold.pdf)
3. [Reducing the risk of false discovery – Naeem.pdf](./02%20-%20Paper/04%20-%20Data%20PRE-PROCESSING/Naeem.pdf)
4. [Reducing the risk (Additional 1).pdf](./02%20-%20Paper/04%20-%20Data%20PRE-PROCESSING/Additional%201%20-%20Naeem.pdf)
5. [Reducing the risk (Additional 2).pdf](./02%20-%20Paper/04%20-%20Data%20PRE-PROCESSING/Additional%202%20-%20Naeem.pdf)
6. [An evaluation using pipelines for Illumina HumanMethylation450 – Marabita.pdf](./02%20-%20Paper/04%20-%20Data%20PRE-PROCESSING/Marabita.pdf)
7. [Analysis pipelines and packages for Infinium 450k – Morris.pdf](./02%20-%20Paper/04%20-%20Data%20PRE-PROCESSING/Morris.pdf)
8. [Comparison of Beta-value and M-value methods – Du.pdf](./02%20-%20Paper/04%20-%20Data%20PRE-PROCESSING/Du.pdf)
9. [Comparison of pre-processing methodologies for Illumina 450k methylation arrays – Cazaly.pdf](./02%20-%20Paper/04%20-%20Data%20PRE-PROCESSING/Cazaly.pdf)
10. [Early detection and diagnosis of cancer with interpretable machine learning – Luo.pdf](./02%20-%20Paper/04%20-%20Data%20PRE-PROCESSING/Luo.pdf)
11. [Filtering table.csv (reference table)](./02%20-%20Paper/04%20-%20Data%20PRE-PROCESSING/filtering_table.csv)
12. [Reducing the risk of false discovery enabling identification of biologically significant genome-wide methylation status using the HumanMethylation450 array – Naeem.pdf](./02%20-%20Paper/04%20-%20Data%20PRE-PROCESSING/Reducing%20the%20risk%20of%20false%20discovery%20-%20Naeem.pdf)


---

### ⚙️ Work in progress
This project is currently **under active development**.  
Notebooks, results, and references will be updated regularly as new analyses are completed and new sections of the thesis are written.

---

### 📚 Keywords
`Epigenetics` · `DNA methylation` · `Neoplasms` · `Cancer biology` · `Bioinformatics` · `Methylation data analysis`



