############################################################
# HORVATH 2013 EPIGENETIC CLOCK — COMPLETE PIPELINE
############################################################
# This script implements the Horvath (2013) epigenetic age estimation
# from DNA methylation beta-values stored in Parquet format.
#
# MAIN STEPS:
#  1) Load beta matrix + phenotype from Parquet
#  2) Filter samples to Normal + Adjacent (labels 0 and 1)
#  3) Build initial beta matrix (samples x CpGs)
#  4) Filter CpGs to Horvath 21k panel and preserve order
#  5) Transpose to Horvath format (CpGs in rows, samples in columns)
#  6) Align probe annotation (probeAnnotation21kdatMethUsed) to data
#  7) Run Horvath StepwiseAnalysis to obtain normalized data
#  8) MANUAL STEP 4: predict DNAmAge using datMethUsedNormalized,
#     imputing missing clock CpGs with goldstandard values
#  9) Merge DNAmAge with phenotype and evaluate accuracy
# 10) Compute age acceleration metrics and print summary
#
# AUTHOR: Adapted for Parquet input and thesis workflow
# DATE: 2025
############################################################

# =========================
# USER PARAMETERS (EDIT ME)
# =========================

# Dataset-specific paths (change these for each dataset)
BETA_PARQUET_PATH  <- "C:/Users/elisa/Desktop/THESIS REPOSITORY/R/GSE69914.parquet"
PHENO_PARQUET_PATH <- "C:/Users/elisa/Desktop/THESIS REPOSITORY/R/pheno_GSE69914.parquet"

# Column in the Parquet file that contains the sample identifier
SAMPLE_ID_COL      <- "id_tissue"

# Column that encodes the tissue label (0 = Normal, 1 = Adjacent, 2 = Tumor, ...)
LABEL_COL          <- "label"

# Output prefix for result files
OUTPUT_PREFIX      <- "GSE69914"

# Horvath supplementary files (full paths or relative to working directory)
# Files from: Horvath, S. (2013). Genome Biology, 14(10), R115.
FILE_ANNOT_21K      <- "C:/Users/elisa/Desktop/THESIS REPOSITORY/R/13059_2013_3156_MOESM21_ESM.csv"
FILE_ANNOT_21K_USED <- "C:/Users/elisa/Desktop/THESIS REPOSITORY/R/13059_2013_3156_MOESM22_ESM.csv"
FILE_PREDICTOR      <- "C:/Users/elisa/Desktop/THESIS REPOSITORY/R/13059_2013_3156_MOESM23_ESM.csv"
FILE_NORMALIZATION  <- "C:/Users/elisa/Desktop/THESIS REPOSITORY/R/13059_2013_3156_MOESM24_ESM.txt"
FILE_STEPWISE       <- "C:/Users/elisa/Desktop/THESIS REPOSITORY/R/13059_2013_3156_MOESM25_ESM.txt"

############################################################
# PACKAGE SETUP (Horvath dependencies)
############################################################

cat("\n=== INSTALLING AND LOADING REQUIRED PACKAGES ===\n")

# CRAN packages
cran_pkgs <- c("arrow", "dplyr", "WGCNA")

for (pkg in cran_pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat("Installing", pkg, "from CRAN...\n")
    install.packages(pkg)
  }
}

# Bioconductor packages required by Horvath / WGCNA
if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}

bioc_pkgs <- c("impute", "preprocessCore", "GO.db", "AnnotationDbi")

for (pkg in bioc_pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat("Installing", pkg, "from Bioconductor...\n")
    BiocManager::install(pkg, ask = FALSE, update = FALSE)
  }
}

# Load required libraries
library(arrow)      # For reading Parquet files
library(dplyr)      # For data manipulation
library(WGCNA)      # Required by Horvath normalization
library(impute)     # For KNN imputation of missing values

cat("✓ All packages loaded successfully\n")

############################################################
# HORVATH FUNCTIONS & SUPPORT FILES
############################################################

cat("\n=== LOADING HORVATH SUPPORT FILES ===\n")

# Age transformation functions (from Horvath's tutorial)
# These functions convert between chronological and transformed age
trafo <- function(x, adult.age = 20) {
  x <- (x + 1) / (1 + adult.age)
  y <- ifelse(x <= 1, log(x), x - 1)
  y
}

anti.trafo <- function(x, adult.age = 20) {
  ifelse(x < 0, (1 + adult.age) * exp(x) - 1, (1 + adult.age) * x + adult.age)
}

# Read Horvath annotation and predictor files
probeAnnotation21kdatMethUsed <- read.csv(FILE_ANNOT_21K_USED)
probeAnnotation27k            <- read.csv(FILE_ANNOT_21K)
datClock                      <- read.csv(FILE_PREDICTOR)

cat("✓ Loaded", nrow(probeAnnotation21kdatMethUsed), "CpG annotations (21k used)\n")
cat("✓ Loaded", nrow(datClock) - 1, "clock CpG coefficients\n")

# Load Horvath normalization and helper functions (BMIQ etc.)
source(FILE_NORMALIZATION)
cat("✓ Horvath normalization functions loaded\n")

############################################################
# LOAD BETA MATRIX & PHENOTYPE FROM PARQUET
############################################################

cat("\n=== LOADING METHYLATION DATA FROM PARQUET ===\n")

# 1) Read beta-value table and phenotype table
beta_df  <- read_parquet(BETA_PARQUET_PATH)  |> as.data.frame()
pheno_df <- read_parquet(PHENO_PARQUET_PATH) |> as.data.frame()

cat("✓ Beta values loaded:", nrow(beta_df), "samples x", ncol(beta_df) - 2, "CpG features (approx.)\n")
cat("✓ Phenotype loaded:", nrow(pheno_df), "samples\n")

# 2) Basic checks to ensure required columns exist
if (!SAMPLE_ID_COL %in% names(beta_df)) {
  stop(paste("Column", SAMPLE_ID_COL, "not found in beta_df."))
}
if (!SAMPLE_ID_COL %in% names(pheno_df)) {
  stop(paste("Column", SAMPLE_ID_COL, "not found in pheno_df."))
}
if (!LABEL_COL %in% names(beta_df) && !LABEL_COL %in% names(pheno_df)) {
  stop(paste("Column", LABEL_COL, "not found in beta_df nor pheno_df."))
}

############################################################
# FILTER TO NORMAL + ADJACENT (label 0 and 1 ONLY)
############################################################

cat("\n=== FILTERING TO NORMAL + ADJACENT SAMPLES (label 0/1) ===\n")

# Prefer label from beta_df if present, otherwise from pheno_df
if (LABEL_COL %in% names(beta_df)) {
  label_vec <- beta_df[[LABEL_COL]]
} else {
  pheno_labels <- pheno_df[, c(SAMPLE_ID_COL, LABEL_COL), drop = FALSE]
  beta_df <- beta_df %>%
    left_join(pheno_labels, by = setNames(SAMPLE_ID_COL, SAMPLE_ID_COL))
  label_vec <- beta_df[[LABEL_COL]]
}

# Keep only Normal (0) + Adjacent (1); exclude Tumor (2) and others
keep_mask <- label_vec %in% c(0, 1)
beta_df   <- beta_df[keep_mask, , drop = FALSE]

cat("✓ Filtered to", sum(keep_mask), "samples (Normal + Adjacent only)\n")

# Also restrict pheno_df to the same samples and labels
pheno_df <- pheno_df %>%
  filter(.data[[LABEL_COL]] %in% c(0, 1),
         .data[[SAMPLE_ID_COL]] %in% beta_df[[SAMPLE_ID_COL]])

############################################################
# BUILD dat0 IN INITIAL FORMAT (samples x CpGs)
############################################################

cat("\n=== BUILDING INITIAL BETA MATRIX (dat0) ===\n")

# Identify CpG columns (names starting with "cg")
cpg_cols <- grep("^cg", names(beta_df), value = TRUE)

if (length(cpg_cols) == 0L) {
  stop("No CpG-like columns found (no columns starting with 'cg').")
}

cat("✓ Found", length(cpg_cols), "CpG columns\n")

# Beta matrix: samples in rows, CpGs in columns
beta_mat <- as.matrix(beta_df[, cpg_cols, drop = FALSE])

# dat0: first column = SampleID, others = CpGs
dat0 <- data.frame(
  SampleID = beta_df[[SAMPLE_ID_COL]],
  beta_mat,
  check.names = FALSE
)

rownames(dat0) <- dat0$SampleID

nSamples <- nrow(dat0)
nProbes  <- ncol(dat0) - 1

if (nSamples <= 0L) stop("No samples found in dat0.")
if (nProbes  <= 0L) stop("No probes found in dat0.")

cat("✓ Created dat0:", nSamples, "samples x", nProbes, "probes\n")

# Create a fresh log file (mirroring Horvath script)
if (file.exists("LogFile.txt")) file.remove("LogFile.txt")
file.create("LogFile.txt")
cat(
  paste("The methylation data set (Normal + Adjacent only) contains", nSamples,
        "samples (arrays) and", nProbes, "probes.\n"),
  file = "LogFile.txt",
  append = TRUE
)

# Helper kept for compatibility (for non-numeric data); here beta are already numeric
asnumeric1 <- function(x) as.numeric(as.character(x))

# For Parquet beta-values (already numeric), we can keep dat1 = dat0 for now
dat1 <- dat0

############################################################
# CRITICAL STEP: FILTER AND REORDER CpGs TO HORVATH PANEL
############################################################

cat("\n=== FILTERING TO HORVATH 21k CpGs (ORDER PRESERVED) ===\n")

# CpGs required by Horvath (in the exact order)
horvath_cpgs <- as.character(probeAnnotation21kdatMethUsed$Name)

cat("Total CpGs in dataset:", length(cpg_cols), "\n")
cat("CpGs needed by Horvath:", length(horvath_cpgs), "\n")

# Keep only Horvath CpGs that are present in the dataset, preserving order
available_horvath_cpgs <- horvath_cpgs[horvath_cpgs %in% cpg_cols]
cat("CpGs available in both:", length(available_horvath_cpgs), "\n")

if (length(available_horvath_cpgs) < 100) {
  stop("Too few Horvath CpGs found in your data. Check CpG naming conventions.")
}

# Extract sample IDs and filter beta matrix to Horvath CpGs
sample_ids   <- dat1$SampleID
beta_matrix  <- as.matrix(dat1[, available_horvath_cpgs, drop = FALSE])

cat("Beta matrix dimensions after Horvath filtering:", dim(beta_matrix), "\n")

############################################################
# TRANSPOSE TO HORVATH FORMAT (CpGs in rows, samples in cols)
############################################################

cat("\n=== TRANSPOSING TO HORVATH FORMAT ===\n")

beta_matrix_t <- t(beta_matrix)

# dat1: first column = ProbeID, then one column per sample
dat1 <- data.frame(
  ProbeID = available_horvath_cpgs,
  beta_matrix_t,
  check.names = FALSE,
  stringsAsFactors = FALSE
)

colnames(dat1) <- c("ProbeID", sample_ids)
rownames(dat1) <- dat1$ProbeID

cat("✓ Transposed data created\n")
cat("  Rows (CpGs):", nrow(dat1), "\n")
cat("  Columns (1 ProbeID + samples):", ncol(dat1), "\n")

############################################################
# ALIGN PROBE ANNOTATION TO DATA (ORDER-MATCHED)
############################################################

cat("\n=== ALIGNING PROBE ANNOTATION TO dat1 (CRITICAL) ===\n")
cat("Original probeAnnotation21kdatMethUsed rows:", nrow(probeAnnotation21kdatMethUsed), "\n")

cpgs_in_dat1 <- as.character(dat1$ProbeID)

probeAnnotation21kdatMethUsed <- probeAnnotation21kdatMethUsed %>%
  dplyr::filter(Name %in% cpgs_in_dat1) %>%
  dplyr::arrange(match(Name, cpgs_in_dat1))

cat("Filtered probeAnnotation21kdatMethUsed rows:", nrow(probeAnnotation21kdatMethUsed), "\n")
cat("CpGs in dat1:", length(cpgs_in_dat1), "\n")

# Consistency checks
if (nrow(probeAnnotation21kdatMethUsed) != length(cpgs_in_dat1)) {
  stop("ERROR: Mismatch between probeAnnotation21kdatMethUsed and dat1 CpGs!")
}

if (!all(probeAnnotation21kdatMethUsed$Name == cpgs_in_dat1)) {
  stop("ERROR: Order mismatch between probeAnnotation21kdatMethUsed and dat1!")
}

cat("✓ Probe annotation successfully filtered and aligned\n")

############################################################
# RUN HORVATH STEPWISE ANALYSIS (FOR NORMALIZATION ONLY)
############################################################

cat("\n=== RUNNING HORVATH STEPWISE ANALYSIS (to obtain normalized data) ===\n")
cat("This may take several minutes (BMIQ normalization is expensive)...\n")

# Flag controlling normalization (as in Horvath tutorial)
normalizeData <- TRUE

# Reproducibility
set.seed(1)

# The StepwiseAnalysis script expects:
#  - dat1 (CpGs in rows, samples in columns, first column = ProbeID)
#  - probeAnnotation21kdatMethUsed, probeAnnotation27k, datClock
#  - trafo / anti.trafo, normalizeData
# and will create objects including datMethUsedNormalized.
source(FILE_STEPWISE)

# Check that datMethUsedNormalized exists
if (!exists("datMethUsedNormalized")) {
  stop("ERROR: 'datMethUsedNormalized' was not created by StepwiseAnalysis. Check Horvath scripts.")
}

cat("✓ StepwiseAnalysis completed and normalized data available (datMethUsedNormalized)\n")

############################################################
# STEP 4 (MANUAL): PREDICT DNAmAge USING NORMALIZED DATA
############################################################

cat("\n=== STEP 4 (MANUAL): PREDICTING DNAmAge FROM CLOCK CpGs ===\n")

# Select CpGs needed for the clock
selectCpGsClock <- is.element(dimnames(datMethUsedNormalized)[[2]],
                              as.character(datClock$CpGmarker[-1]))

cat("CpGs needed by clock:", nrow(datClock) - 1, "\n")
cat("CpGs found in normalized data:", sum(selectCpGsClock), "\n")

# If some clock CpGs are missing, impute with goldstandard values
if (sum(selectCpGsClock) < nrow(datClock) - 1) {
  cat("WARNING: Not all clock CpGs are present. Imputing missing CpGs with goldstandard values...\n")
  
  # Convert to data frame if needed
  if (is.matrix(datMethUsedNormalized)) {
    datMethUsedNormalized <- as.data.frame(datMethUsedNormalized)
  }
  
  cpgs_needed    <- as.character(datClock$CpGmarker[-1])
  cpgs_available <- colnames(datMethUsedNormalized)
  missing_cpgs   <- setdiff(cpgs_needed, cpgs_available)
  
  cat("Missing CpGs:", length(missing_cpgs), "\n")
  
  n_samples_norm <- nrow(datMethUsedNormalized)
  
  for (cpg in missing_cpgs) {
    # 1) Try 21k annotation
    idx21 <- which(probeAnnotation21kdatMethUsed$Name == cpg)
    if (length(idx21) > 0) {
      gold_value <- probeAnnotation21kdatMethUsed$goldstandard2[idx21[1]]
    } else {
      # 2) Try 27k annotation
      idx27 <- which(probeAnnotation27k$Name == cpg)
      if (length(idx27) > 0) {
        gold_value <- probeAnnotation27k$goldstandard[idx27[1]]
      } else {
        # 3) Final fallback: neutral methylation level
        gold_value <- 0.5
      }
    }
    
    # Ensure a valid value
    if (is.na(gold_value) || length(gold_value) == 0) {
      gold_value <- 0.5
    }
    
    datMethUsedNormalized[[cpg]] <- rep(gold_value, n_samples_norm)
  }
  
  # Recompute selection mask after imputing missing CpGs
  selectCpGsClock <- is.element(
    dimnames(datMethUsedNormalized)[[2]],
    as.character(datClock$CpGmarker[-1])
  )
}

if (sum(selectCpGsClock) < nrow(datClock) - 1) {
  stop("ERROR: Still missing clock CpGs after imputation.")
}

# Extract clock CpGs in the correct order
datMethClock0 <- data.frame(datMethUsedNormalized[, selectCpGsClock])
datMethClock  <- data.frame(datMethClock0[as.character(datClock$CpGmarker[-1])])

cat("Clock matrix dimensions (samples x CpGs):", dim(datMethClock), "\n")

# Predict DNAmAge using anti.trafo and Horvath coefficients
predictedAge <- as.numeric(anti.trafo(
  datClock$CoefficientTraining[1] +
    as.matrix(datMethClock) %*% as.numeric(datClock$CoefficientTraining[-1])
))

cat("✓ Age prediction completed for", length(predictedAge), "samples\n")

# Map back sample IDs from dat1 (column names, excluding ProbeID)
sample_ids <- colnames(dat1)[-1]  # first column is ProbeID

if (length(sample_ids) != length(predictedAge)) {
  stop("ERROR: Number of sample IDs does not match number of predicted ages.")
}

cat("Sample IDs extracted:", length(sample_ids), "\n")

# datout: final DNAmAge table
datout <- data.frame(
  SampleID = sample_ids,
  DNAmAge  = predictedAge,
  stringsAsFactors = FALSE
)

cat("✓ Created datout with", nrow(datout), "samples\n")
cat("First few predicted ages:", head(predictedAge, 5), "\n")

############################################################
# SAVE HORVATH OUTPUT & MERGE WITH PHENOTYPE
############################################################

cat("\n=== SAVING HORVATH DNAmAge OUTPUT AND MERGING WITH PHENOTYPE ===\n")

# 1) Save raw DNAmAge output
out_csv_path <- paste0("Output_", OUTPUT_PREFIX, "_HorvathDNAmAge_normal_adj_only.csv")
write.table(
  datout,
  file      = out_csv_path,
  sep       = ",",
  row.names = FALSE,
  quote     = TRUE
)
cat("✓ Horvath DNAmAge output written to:", out_csv_path, "\n")

# 2) ID column is explicitly SampleID
HORVATH_ID_COL <- "SampleID"
cat("Using Horvath ID column:", HORVATH_ID_COL, "\n")

# 3) Merge DNAmAge into phenotype table (Normal + Adjacent only)
pheno_with_dnamage <- pheno_df %>%
  dplyr::left_join(datout, by = setNames(HORVATH_ID_COL, SAMPLE_ID_COL))

# 4) Save enriched phenotype table
pheno_out_path <- paste0("Pheno_", OUTPUT_PREFIX, "_with_HorvathDNAmAge_normal_adj_only.csv")
write.table(
  pheno_with_dnamage,
  file      = pheno_out_path,
  sep       = ",",
  row.names = FALSE,
  quote     = TRUE
)
cat("✓ Phenotype with Horvath DNAmAge written to:", pheno_out_path, "\n")

############################################################
# EVALUATION OF HORVATH DNAmAge ACCURACY
############################################################

cat("\n=== EVALUATING HORVATH CLOCK ACCURACY (Normal + Adjacent only) ===\n")

# Sanity checks on required columns
if (!"age_at_surgery" %in% colnames(pheno_with_dnamage)) {
  stop("ERROR: Column 'age_at_surgery' not found in phenotype table.")
}
if (!"DNAmAge" %in% colnames(pheno_with_dnamage)) {
  stop("ERROR: Column 'DNAmAge' not found in merged phenotype.")
}
if (!LABEL_COL %in% colnames(pheno_with_dnamage)) {
  warning(paste("Column", LABEL_COL, "not found. Cannot compute per-group accuracy."))
}

# Filter complete cases
eval_df <- pheno_with_dnamage %>%
  dplyr::filter(!is.na(age_at_surgery),
                !is.na(DNAmAge))

cat("Samples with complete data for evaluation:", nrow(eval_df), "\n")

if (nrow(eval_df) == 0) {
  stop("ERROR: No samples with both age_at_surgery and DNAmAge available.")
}
if (nrow(eval_df) < 10) {
  warning(paste0("WARNING: Very few samples (n = ", nrow(eval_df),
                 ") available for evaluation. Results may be unstable."))
}

# Evaluation metrics
RMSE <- function(y, yhat) sqrt(mean((y - yhat)^2))
MAE  <- function(y, yhat) mean(abs(y - yhat))
MAPE <- function(y, yhat) mean(abs((y - yhat) / y)) * 100

# Global metrics (Normal + Adjacent together)
y_true <- eval_df$age_at_surgery
y_pred <- eval_df$DNAmAge

global_rmse <- RMSE(y_true, y_pred)
global_mae  <- MAE(y_true, y_pred)
global_cor  <- cor(y_true, y_pred)
global_r2   <- global_cor^2
global_mape <- MAPE(y_true, y_pred)

cat("\n================ GLOBAL PERFORMANCE (NORMAL + ADJACENT) ================\n")
cat("N samples:     ", nrow(eval_df), "\n")
cat("MAE   (years): ", round(global_mae, 3), "\n")
cat("RMSE  (years): ", round(global_rmse, 3), "\n")
cat("R     (corr):  ", round(global_cor, 3), "\n")
cat("R²            : ", round(global_r2, 3), "\n")
cat("MAPE   (%):    ", round(global_mape, 3), "\n")

# Per-group metrics (0 = Normal, 1 = Adjacent)
if (LABEL_COL %in% colnames(eval_df)) {
  group_results <- eval_df %>%
    dplyr::group_by(.data[[LABEL_COL]]) %>%
    dplyr::summarise(
      n            = dplyr::n(),
      MAE_years    = MAE(age_at_surgery, DNAmAge),
      RMSE_years   = RMSE(age_at_surgery, DNAmAge),
      Correlation  = cor(age_at_surgery, DNAmAge),
      R2           = Correlation^2,
      MAPE_percent = MAPE(age_at_surgery, DNAmAge),
      .groups      = "drop"
    )
  
  cat("\n================ GROUP-SPECIFIC PERFORMANCE ================\n")
  cat("(0 = Normal tissue, 1 = Adjacent tissue)\n\n")
  print(group_results)
}

############################################################
# AGE ACCELERATION METRICS
############################################################

eval_df <- eval_df %>%
  dplyr::mutate(age_acceleration = DNAmAge - age_at_surgery)

cat("\n================ AGE ACCELERATION SUMMARY ================\n")
cat("Definition: age_acceleration = DNAmAge - chronological age\n\n")

acc_summary <- eval_df %>%
  dplyr::summarise(
    mean_acc   = mean(age_acceleration),
    sd_acc     = sd(age_acceleration),
    median_acc = median(age_acceleration),
    q1_acc     = quantile(age_acceleration, 0.25),
    q3_acc     = quantile(age_acceleration, 0.75)
  )
print(acc_summary)

if (LABEL_COL %in% colnames(eval_df)) {
  cat("\n================ AGE ACCELERATION BY GROUP ================\n")
  acc_group <- eval_df %>%
    dplyr::group_by(.data[[LABEL_COL]]) %>%
    dplyr::summarise(
      n          = dplyr::n(),
      mean_acc   = mean(age_acceleration),
      median_acc = median(age_acceleration),
      sd_acc     = sd(age_acceleration),
      IQR_acc    = IQR(age_acceleration),
      .groups    = "drop"
    )
  print(acc_group)
}

############################################################
# PIPELINE COMPLETED
############################################################

cat("\n", paste(rep("=", 70), collapse = ""), "\n", sep = "")
cat("✓ HORVATH 2013 EPIGENETIC CLOCK PIPELINE COMPLETED SUCCESSFULLY\n")
cat(paste(rep("=", 70), collapse = ""), "\n\n", sep = "")

cat("Output files generated:\n")
cat("  1.", out_csv_path, "\n")
cat("  2.", pheno_out_path, "\n\n")

cat("Summary statistics (Normal + Adjacent):\n")
cat("  - Samples analyzed:", nrow(eval_df), "\n")
cat("  - CpGs used (after Horvath filtering):", nrow(dat1), "\n")
cat("  - MAE:", round(global_mae, 2), "years\n")
cat("  - R² :", round(global_r2, 3), "\n\n")

cat("End of script.\n")
