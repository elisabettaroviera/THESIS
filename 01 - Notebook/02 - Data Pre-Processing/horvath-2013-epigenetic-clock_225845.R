############################################################
# HORVATH 2013 EPIGENETIC CLOCK — COMPLETE PIPELINE
############################################################
# This script implements the Horvath (2013) epigenetic age estimation
# from DNA methylation beta-values stored in Parquet format.
#
# KEY FEATURES:
# - Reads methylation data from Parquet files (efficient binary format)
# - Filters samples by tissue type (Normal + Adjacent only)
# - Properly formats data for Horvath's original scripts
# - Handles CpG filtering, ordering, and annotation alignment
# - Runs BMIQ normalization and age prediction
# - Evaluates accuracy against chronological age
# - Computes age acceleration metrics
#
# CRITICAL FIXES IMPLEMENTED:
# 1. Transpose data to match Horvath format (CpGs in rows, samples in cols)
# 2. Filter to only CpGs present in Horvath annotation (~21k CpGs)
# 3. Preserve exact order of CpGs to match goldstandard reference
# 4. Filter probeAnnotation21kdatMethUsed to match available CpGs
#
# AUTHOR: Adapted for Parquet input and thesis workflow
# DATE: 2025
############################################################

# =========================
# USER PARAMETERS (EDIT ME)
# =========================

# Dataset-specific paths (change these for each dataset)
BETA_PARQUET_PATH  <- "C:/Users/elisa/Desktop/THESIS REPOSITORY/R/GSE225845.parquet"
PHENO_PARQUET_PATH <- "C:/Users/elisa/Desktop/THESIS REPOSITORY/R/pheno_GSE225845.parquet"

# Column in the Parquet file that contains the sample identifier
SAMPLE_ID_COL      <- "id_tissue"

# Column that encodes the tissue label (0 = Normal, 1 = Adjacent, 2 = Tumor, ...)
LABEL_COL          <- "label"

# Output prefix for result files
OUTPUT_PREFIX      <- "GSE225845"

# Horvath supplementary files (must be in the current working directory or given with full paths)
# These files are from: Horvath, S. (2013). Genome Biology, 14(10), R115.
# Download from: https://genomebiology.biomedcentral.com/articles/10.1186/gb-2013-14-10-r115
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
# These contain CpG annotations and clock coefficients
probeAnnotation21kdatMethUsed <- read.csv(FILE_ANNOT_21K_USED)
probeAnnotation27k            <- read.csv(FILE_ANNOT_21K)
datClock                      <- read.csv(FILE_PREDICTOR)

cat("✓ Loaded", nrow(probeAnnotation21kdatMethUsed), "CpG annotations\n")
cat("✓ Loaded", nrow(datClock) - 1, "clock CpG coefficients\n")

# Load Horvath normalization and helper functions
# This script contains the BMIQ calibration function
source(FILE_NORMALIZATION)

cat("✓ Horvath normalization functions loaded\n")

############################################################
# LOAD BETA MATRIX & PHENOTYPE FROM PARQUET
############################################################

cat("\n=== LOADING METHYLATION DATA ===\n")

# 1) Read beta-value table and phenotype table
# Parquet is a columnar binary format that is much faster than CSV
beta_df  <- read_parquet(BETA_PARQUET_PATH)  |> as.data.frame()
pheno_df <- read_parquet(PHENO_PARQUET_PATH) |> as.data.frame()

cat("✓ Beta values loaded:", nrow(beta_df), "samples x", ncol(beta_df) - 2, "features\n")
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

cat("\n=== FILTERING TO NORMAL + ADJACENT SAMPLES ===\n")

# Prefer label from beta_df if present, otherwise from pheno_df
if (LABEL_COL %in% names(beta_df)) {
  label_vec <- beta_df[[LABEL_COL]]
} else {
  # Align pheno labels by SAMPLE_ID_COL
  pheno_labels <- pheno_df[, c(SAMPLE_ID_COL, LABEL_COL), drop = FALSE]
  beta_df <- beta_df %>%
    left_join(pheno_labels, by = setNames(SAMPLE_ID_COL, SAMPLE_ID_COL))
  label_vec <- beta_df[[LABEL_COL]]
}

# Keep only Normal (0) + Adjacent (1) tissue samples
# Tumor samples (label 2) are excluded for age estimation
keep_mask <- label_vec %in% c(0, 1)

beta_df  <- beta_df[keep_mask, , drop = FALSE]

cat("✓ Filtered to", sum(keep_mask), "samples (Normal + Adjacent only)\n")

# Also restrict pheno_df to the same samples and labels
pheno_df <- pheno_df %>%
  filter(.data[[LABEL_COL]] %in% c(0, 1),
         .data[[SAMPLE_ID_COL]] %in% beta_df[[SAMPLE_ID_COL]])

############################################################
# BUILD dat0 IN INITIAL FORMAT (samples x CpGs)
############################################################

cat("\n=== BUILDING INITIAL DATA MATRIX ===\n")

# Identify CpG columns (columns whose names start with "cg")
cpg_cols <- grep("^cg", names(beta_df), value = TRUE)

if (length(cpg_cols) == 0L) {
  stop("No CpG-like columns found (no columns starting with 'cg').")
}

cat("✓ Found", length(cpg_cols), "CpG columns\n")

# Build beta matrix: samples in rows, CpGs in columns
beta_mat <- as.matrix(beta_df[, cpg_cols, drop = FALSE])

# BUILD dat0: first column = SampleID, others = CpGs
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

# Create a log file (as in Horvath's tutorial)
if (file.exists("LogFile.txt")) file.remove("LogFile.txt")
file.create("LogFile.txt")
cat(
  paste("The methylation data set (Normal + Adjacent only) contains", nSamples,
        "samples (arrays) and", nProbes, "probes.\n"),
  file = "LogFile.txt",
  append = TRUE
)

# Horvath conversion helper (kept for compatibility)
asnumeric1 <- function(x) as.numeric(as.character(x))

# For Parquet beta-values (already numeric), we can skip heavy conversion
dat1 <- dat0

############################################################
# CRITICAL FIX 1: FILTER AND REORDER CpGs
############################################################
# PROBLEM: The dataset contains ~750k CpGs, but Horvath uses only ~21k
# SOLUTION: Filter to keep only CpGs in Horvath annotation, preserving order

cat("\n=== FILTERING TO HORVATH CpGs (CRITICAL STEP) ===\n")

# Get the list of CpGs that Horvath needs (in the correct order!)
horvath_cpgs <- as.character(probeAnnotation21kdatMethUsed$Name)

cat("Total CpGs in dataset:", length(cpg_cols), "\n")
cat("CpGs needed by Horvath:", length(horvath_cpgs), "\n")

# Find which Horvath CpGs are available in your data
# CRITICAL: Use horvath_cpgs order, not intersect which randomizes order!
available_horvath_cpgs <- horvath_cpgs[horvath_cpgs %in% cpg_cols]
cat("CpGs available in both:", length(available_horvath_cpgs), "\n")

if (length(available_horvath_cpgs) < 100) {
  stop("Too few Horvath CpGs found in your data. Check CpG naming conventions.")
}

# Extract sample IDs and filter beta matrix to Horvath CpGs only
sample_ids <- dat1$SampleID
beta_matrix <- as.matrix(dat1[, available_horvath_cpgs, drop = FALSE])

cat("Beta matrix dimensions after filtering:", dim(beta_matrix), "\n")

############################################################
# CRITICAL FIX 2: TRANSPOSE TO HORVATH FORMAT
############################################################
# PROBLEM: Horvath expects CpGs in rows, samples in columns
# SOLUTION: Transpose the matrix and create proper data frame

cat("\n=== TRANSPOSING TO HORVATH FORMAT ===\n")

# Transpose: CpGs in rows, samples in columns
beta_matrix_t <- t(beta_matrix)

# Create dat1 with ProbeID as FIRST COLUMN, then samples
# This is the exact format expected by Horvath's scripts
dat1 <- data.frame(
  ProbeID = available_horvath_cpgs,  # Use the ordered list directly!
  beta_matrix_t,
  check.names = FALSE,
  stringsAsFactors = FALSE
)

# Set column names: first is "ProbeID", rest are sample IDs
colnames(dat1) <- c("ProbeID", sample_ids)
rownames(dat1) <- dat1$ProbeID

cat("✓ Transposed data created\n")
cat("  Rows (CpGs):", nrow(dat1), "\n")
cat("  Columns (1 ProbeID + Samples):", ncol(dat1), "\n")
cat("  First 3 CpGs in dat1:", paste(head(dat1[,1], 3), collapse = ", "), "\n")

############################################################
# CRITICAL FIX 3: FILTER PROBE ANNOTATION
############################################################
# PROBLEM: probeAnnotation21kdatMethUsed has all 21k CpGs, but we only have 19k
# SOLUTION: Filter annotation to match exactly what's in dat1

cat("\n=== ALIGNING PROBE ANNOTATION (CRITICAL STEP) ===\n")
cat("Original probeAnnotation21kdatMethUsed rows:", nrow(probeAnnotation21kdatMethUsed), "\n")

# Get the CpGs that are actually in dat1 (in the correct order)
cpgs_in_dat1 <- as.character(dat1$ProbeID)

# Filter probeAnnotation21kdatMethUsed to keep only CpGs present in dat1
# AND preserve the order to match dat1
probeAnnotation21kdatMethUsed <- probeAnnotation21kdatMethUsed %>%
  filter(Name %in% cpgs_in_dat1) %>%
  arrange(match(Name, cpgs_in_dat1))

cat("Filtered probeAnnotation21kdatMethUsed rows:", nrow(probeAnnotation21kdatMethUsed), "\n")
cat("CpGs in dat1:", length(cpgs_in_dat1), "\n")

# Verify they match exactly
if (nrow(probeAnnotation21kdatMethUsed) != length(cpgs_in_dat1)) {
  stop("ERROR: Mismatch between probeAnnotation21kdatMethUsed and dat1 CpGs!")
}

# Verify order matches exactly (this is CRITICAL for BMIQ calibration)
if (!all(probeAnnotation21kdatMethUsed$Name == cpgs_in_dat1)) {
  stop("ERROR: Order mismatch between probeAnnotation21kdatMethUsed and dat1!")
}

cat("✓ Probe annotation successfully filtered and aligned\n")
cat("  First 3 CpGs in annotation:", paste(head(probeAnnotation21kdatMethUsed$Name, 3), collapse = ", "), "\n")
cat("  First 3 CpGs in dat1:       ", paste(head(cpgs_in_dat1, 3), collapse = ", "), "\n")

############################################################
# RUN HORVATH STEPWISE ANALYSIS
############################################################

cat("\n=== RUNNING HORVATH AGE ESTIMATION ===\n")
cat("This may take several minutes (BMIQ normalization ~8s per sample)...\n")

# This flag controls whether data are normalized (recommended)
normalizeData <- TRUE

# Set seed for reproducibility (as in the tutorial)
set.seed(1)

# Run the original StepwiseAnalysis script
# This script expects:
#   - dat1 (CpGs in rows, samples in columns, first column = ProbeID)
#   - probeAnnotation21kdatMethUsed (filtered to match dat1)
#   - probeAnnotation27k
#   - datClock
#   - trafo / anti.trafo functions
#   - normalizeData flag
# and will create an object called "datout" with DNAmAge and related fields.
source(FILE_STEPWISE)

# Verify that datout was created successfully
if (!exists("datout")) {
  stop("ERROR: Object 'datout' was not created by StepwiseAnalysis. Check Horvath scripts.")
}

cat("✓ Horvath age estimation completed successfully\n")

############################################################
# SAVE HORVATH OUTPUT & MERGE WITH PHENOTYPE
############################################################

cat("\n=== SAVING RESULTS ===\n")

# 1) Save raw Horvath output
out_csv_path <- paste0("Output_", OUTPUT_PREFIX, "_HorvathDNAmAge_normal_adj_only.csv")
write.table(
  datout,
  file      = out_csv_path,
  sep       = ",",
  row.names = FALSE,
  quote     = TRUE
)

cat("✓ Horvath DNAmAge output written to:", out_csv_path, "\n")

# 2) Robustly detect the ID column in datout
cat("\nColumns in datout:", paste(colnames(datout), collapse = ", "), "\n")

candidate_id_cols <- intersect(
  c("SampleID", "ID", "sampleID", "sample_id", "sampleName", "sample_name"),
  colnames(datout)
)

if (length(candidate_id_cols) == 0L) {
  # Fallback: create SampleID column from rownames, if any
  if (!is.null(rownames(datout)) && any(nzchar(rownames(datout)))) {
    datout$SampleID <- rownames(datout)
    HORVATH_ID_COL <- "SampleID"
  } else {
    stop("ERROR: Could not find a suitable sample ID column in datout.")
  }
} else {
  HORVATH_ID_COL <- candidate_id_cols[1]
}

cat("Using Horvath ID column:", HORVATH_ID_COL, "\n")

# 3) Merge DNAmAge into phenotype table (Normal + Adjacent only)
pheno_with_dnamage <- pheno_df %>%
  left_join(datout, by = setNames(HORVATH_ID_COL, SAMPLE_ID_COL))

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

cat("\n=== EVALUATING HORVATH CLOCK ACCURACY ===\n")

# Basic sanity checks
if (!"age_at_surgery" %in% colnames(pheno_with_dnamage)) {
  stop("ERROR: Column 'age_at_surgery' not found in phenotype table.")
}

if (!"DNAmAge" %in% colnames(pheno_with_dnamage)) {
  stop("ERROR: Column 'DNAmAge' not found in merged phenotype.")
}

if (!LABEL_COL %in% colnames(pheno_with_dnamage)) {
  warning(paste("Column", LABEL_COL, "not found. Cannot compute per-group accuracy."))
}

# Filter complete cases (within Normal + Adjacent only)
eval_df <- pheno_with_dnamage %>%
  filter(!is.na(age_at_surgery),
         !is.na(DNAmAge))

cat("Samples with complete data for evaluation:", nrow(eval_df), "\n")

# Check if we have enough samples for evaluation
if (nrow(eval_df) == 0) {
  stop("ERROR: No samples with both age_at_surgery and DNAmAge available for evaluation.")
}

if (nrow(eval_df) < 10) {
  warning("WARNING: Very few samples (n=", nrow(eval_df), ") available for evaluation. Results may be unreliable.")
}

# Define evaluation metrics
RMSE <- function(y, yhat) sqrt(mean((y - yhat)^2))
MAE  <- function(y, yhat) mean(abs(y - yhat))
MAPE <- function(y, yhat) mean(abs((y - yhat) / y)) * 100

# =========================
# GLOBAL METRICS
# =========================

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

# =========================
# PER-GROUP METRICS
# =========================

if (LABEL_COL %in% colnames(eval_df)) {
  group_results <- eval_df %>%
    group_by(.data[[LABEL_COL]]) %>%
    summarise(
      n              = n(),
      MAE_years      = MAE(age_at_surgery, DNAmAge),
      RMSE_years     = RMSE(age_at_surgery, DNAmAge),
      Correlation    = cor(age_at_surgery, DNAmAge),
      R2             = Correlation^2,
      MAPE_percent   = MAPE(age_at_surgery, DNAmAge),
      .groups        = "drop"
    )
  
  cat("\n================ GROUP-SPECIFIC PERFORMANCE ================\n")
  cat("(0 = Normal tissue, 1 = Adjacent tissue)\n\n")
  print(group_results)
}

# =========================
# AGE ACCELERATION METRICS
# =========================

eval_df <- eval_df %>%
  mutate(age_acceleration = DNAmAge - age_at_surgery)

cat("\n================ AGE ACCELERATION SUMMARY ================\n")
cat("Age acceleration = DNAmAge - Chronological Age\n")
cat("(Positive values indicate epigenetic aging is faster than chronological)\n\n")

acc_summary <- eval_df %>%
  summarise(
    mean_acc     = mean(age_acceleration),
    sd_acc       = sd(age_acceleration),
    median_acc   = median(age_acceleration),
    q1_acc       = quantile(age_acceleration, 0.25),
    q3_acc       = quantile(age_acceleration, 0.75)
  )
print(acc_summary)

if (LABEL_COL %in% colnames(eval_df)) {
  cat("\n================ AGE ACCELERATION BY GROUP ================\n")
  acc_group <- eval_df %>%
    group_by(.data[[LABEL_COL]]) %>%
    summarise(
      n               = n(),
      mean_acc        = mean(age_acceleration),
      median_acc      = median(age_acceleration),
      sd_acc          = sd(age_acceleration),
      IQR_acc         = IQR(age_acceleration),
      .groups         = "drop"
    )
  print(acc_group)
}

############################################################
# PIPELINE COMPLETED
############################################################

cat("\n" , rep("=", 70), "\n", sep = "")
cat("✓ HORVATH 2013 EPIGENETIC CLOCK PIPELINE COMPLETED SUCCESSFULLY\n")
cat(rep("=", 70), "\n\n", sep = "")

cat("Output files generated:\n")
cat("  1.", out_csv_path, "\n")
cat("  2.", pheno_out_path, "\n")
cat("  3. LogFile.txt\n\n")

cat("Summary statistics:\n")
cat("  - Samples analyzed:", nrow(eval_df), "(Normal + Adjacent)\n")
cat("  - CpGs used:", nrow(dat1), "out of", length(horvath_cpgs), "Horvath CpGs\n")
cat("  - MAE:", round(global_mae, 2), "years\n")
cat("  - R²:", round(global_r2, 3), "\n\n")

############################################################
# END OF SCRIPT
############################################################



############################################################
# STEP 4: PREDICT AGE (MANUAL - BYPASS SOURCE)
############################################################

cat("\n=== PREDICTING DNAm AGE (using existing normalized data) ===\n")

# Select CpGs needed for the clock
selectCpGsClock <- is.element(dimnames(datMethUsedNormalized)[[2]], as.character(datClock$CpGmarker[-1]))

cat("CpGs needed by clock:", dim(datClock)[[1]] - 1, "\n")
cat("CpGs found in data:", sum(selectCpGsClock), "\n")

if (sum(selectCpGsClock) < dim(datClock)[[1]] - 1) {
  cat("WARNING: Not all clock CpGs are present. Imputing missing CpGs with goldstandard values...\n")
  
  # Convert to data frame if matrix
  if (is.matrix(datMethUsedNormalized)) {
    datMethUsedNormalized <- as.data.frame(datMethUsedNormalized)
  }
  
  # Identify missing CpGs
  cpgs_needed    <- as.character(datClock$CpGmarker[-1])
  cpgs_available <- colnames(datMethUsedNormalized)
  missing_cpgs   <- setdiff(cpgs_needed, cpgs_available)
  
  cat("Missing CpGs:", length(missing_cpgs), "\n")
  
  n_samples <- nrow(datMethUsedNormalized)
  
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
        # 3) Final fallback
        gold_value <- 0.5
      }
    }
    
    # Ensure gold_value is valid
    if (is.na(gold_value) || length(gold_value) == 0) {
      gold_value <- 0.5
    }
    
    datMethUsedNormalized[[cpg]] <- rep(gold_value, n_samples)
  }
  
  # Recompute clock CpG mask now that missing CpGs were added
  selectCpGsClock <- is.element(
    dimnames(datMethUsedNormalized)[[2]],
    as.character(datClock$CpGmarker[-1])
  )
}


if (sum(selectCpGsClock) < dim(datClock)[[1]] - 1) {
  stop("ERROR: Still missing clock CpGs after imputation.")
}

# Extract clock CpGs in correct order
datMethClock0 <- data.frame(datMethUsedNormalized[, selectCpGsClock])
datMethClock <- data.frame(datMethClock0[as.character(datClock$CpGmarker[-1])])

cat("Clock matrix dimensions:", dim(datMethClock), "\n")

# Predict age
predictedAge <- as.numeric(anti.trafo(datClock$CoefficientTraining[1] + 
                                        as.matrix(datMethClock) %*% 
                                        as.numeric(datClock$CoefficientTraining[-1])))

cat("✓ Age prediction completed for", length(predictedAge), "samples\n")

# Get sample IDs from dat1 (the transposed matrix we created earlier)
sample_ids <- colnames(dat1)[-1]  # Exclude first column (ProbeID)

cat("Sample IDs extracted:", length(sample_ids), "\n")

# Create datout
datout <- data.frame(
  SampleID = sample_ids,
  DNAmAge = predictedAge,
  stringsAsFactors = FALSE
)

cat("✓ Created datout with", nrow(datout), "samples\n")
cat("First few predicted ages:", head(predictedAge, 5), "\n")

############################################################
# SAVE HORVATH OUTPUT & MERGE WITH PHENOTYPE
############################################################

cat("\n=== SAVING RESULTS ===\n")

# 1) Save raw Horvath output
out_csv_path <- paste0("Output_", OUTPUT_PREFIX, "_HorvathDNAmAge_normal_adj_only.csv")
write.table(
  datout,
  file      = out_csv_path,
  sep       = ",",
  row.names = FALSE,
  quote     = TRUE
)

cat("✓ Horvath DNAmAge output written to:", out_csv_path, "\n")

# 2) Use SampleID column
HORVATH_ID_COL <- "SampleID"

cat("Using Horvath ID column:", HORVATH_ID_COL, "\n")

# 3) Merge DNAmAge into phenotype table
pheno_with_dnamage <- pheno_df %>%
  left_join(datout, by = setNames(HORVATH_ID_COL, SAMPLE_ID_COL))

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

cat("\n=== EVALUATING HORVATH CLOCK ACCURACY ===\n")

# Basic sanity checks
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
  filter(!is.na(age_at_surgery),
         !is.na(DNAmAge))

cat("Samples with complete data for evaluation:", nrow(eval_df), "\n")

if (nrow(eval_df) == 0) {
  stop("ERROR: No samples with both age_at_surgery and DNAmAge available.")
}

if (nrow(eval_df) < 10) {
  warning("WARNING: Very few samples (n=", nrow(eval_df), ") available for evaluation.")
}

# Define evaluation metrics
RMSE <- function(y, yhat) sqrt(mean((y - yhat)^2))
MAE  <- function(y, yhat) mean(abs(y - yhat))
MAPE <- function(y, yhat) mean(abs((y - yhat) / y)) * 100

# Global metrics
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

# Per-group metrics
if (LABEL_COL %in% colnames(eval_df)) {
  group_results <- eval_df %>%
    group_by(.data[[LABEL_COL]]) %>%
    summarise(
      n              = n(),
      MAE_years      = MAE(age_at_surgery, DNAmAge),
      RMSE_years     = RMSE(age_at_surgery, DNAmAge),
      Correlation    = cor(age_at_surgery, DNAmAge),
      R2             = Correlation^2,
      MAPE_percent   = MAPE(age_at_surgery, DNAmAge),
      .groups        = "drop"
    )
  
  cat("\n================ GROUP-SPECIFIC PERFORMANCE ================\n")
  cat("(0 = Normal tissue, 1 = Adjacent tissue)\n\n")
  print(group_results)
}

# Age acceleration metrics
eval_df <- eval_df %>%
  mutate(age_acceleration = DNAmAge - age_at_surgery)

cat("\n================ AGE ACCELERATION SUMMARY ================\n")
cat("Age acceleration = DNAmAge - Chronological Age\n\n")

acc_summary <- eval_df %>%
  summarise(
    mean_acc     = mean(age_acceleration),
    sd_acc       = sd(age_acceleration),
    median_acc   = median(age_acceleration),
    q1_acc       = quantile(age_acceleration, 0.25),
    q3_acc       = quantile(age_acceleration, 0.75)
  )
print(acc_summary)

if (LABEL_COL %in% colnames(eval_df)) {
  cat("\n================ AGE ACCELERATION BY GROUP ================\n")
  acc_group <- eval_df %>%
    group_by(.data[[LABEL_COL]]) %>%
    summarise(
      n               = n(),
      mean_acc        = mean(age_acceleration),
      median_acc      = median(age_acceleration),
      sd_acc          = sd(age_acceleration),
      IQR_acc         = IQR(age_acceleration),
      .groups         = "drop"
    )
  print(acc_group)
}

############################################################
# PIPELINE COMPLETED
############################################################

cat("\n" , rep("=", 70), "\n", sep = "")
cat("✓ HORVATH 2013 EPIGENETIC CLOCK PIPELINE COMPLETED SUCCESSFULLY\n")
cat(rep("=", 70), "\n\n", sep = "")

cat("Output files generated:\n")
cat("  1.", out_csv_path, "\n")
cat("  2.", pheno_out_path, "\n\n")

cat("Summary statistics:\n")
cat("  - Samples analyzed:", nrow(eval_df), "(Normal + Adjacent)\n")
cat("  - MAE:", round(global_mae, 2), "years\n")
cat("  - R²:", round(global_r2, 3), "\n\n")

############################################################
# END OF SCRIPT
############################################################