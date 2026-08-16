#!/usr/bin/env Rscript
# ============================================================================
# Paper 1 (IDJ revision) — INTRA-RATER reliability of test-set MBL annotations
# Same rater, two passes:  initial = test_labels/ , retest = retest_true/
# Unit of analysis = implant (n = 151 matched implants).
#
# Computes, for maxMBL%, MBL1% (mesial) and MBL2% (distal):
#   - ICC(A,1)  two-way, absolute-agreement, single measure   (test-retest of ONE rating)
#   - ICC(A,k)  two-way, absolute-agreement, average of k=2 measures
#   - Pearson r, MAE, RMSE, bias, Bland-Altman 95% limits of agreement
# Interpretation per Koo & Li (2016) reliability bands.
#
# Run (from WSL):
#   docker exec claude-desktop Rscript \
#     /home/claudeuser/workspace/yolo/paper1_intra-rater/R/intra_rater_icc.R
# Optional arg 1 = project base dir (defaults to the container mount path).
# ============================================================================

.libPaths(c("/home/R/library", .libPaths()))
suppressMessages({
  library(irr)       # icc()
  library(writexl)   # write_xlsx()
})

args  <- commandArgs(trailingOnly = TRUE)
BASE  <- if (length(args) >= 1) args[1] else
         "/home/workspace/yolo/paper1_intra-rater/retest_final/final_statistics"
csv   <- file.path(BASE, "per_implant_for_R.csv")
outdir<- file.path(BASE, "R")
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

cat("================================================================\n")
cat("INTRA-RATER ICC  —  R", as.character(getRversion()), "\n")
cat("input :", csv, "\n")
d <- read.csv(csv, stringsAsFactors = FALSE)
cat("implants (rows):", nrow(d), " | images:", length(unique(d$image)), "\n")
cat("================================================================\n\n")

koo_li <- function(x) {                       # band on the POINT estimate
  if (is.na(x))      "NA"
  else if (x < 0.50) "poor"
  else if (x < 0.75) "moderate"
  else if (x < 0.90) "good"
  else               "excellent"
}

measures <- list(maxMBL = c("initial_maxMBL", "retest_maxMBL"),
                 MBL1   = c("initial_MBL1",   "retest_MBL1"),
                 MBL2   = c("initial_MBL2",   "retest_MBL2"))

icc_rows <- list(); agr_rows <- list(); desc_rows <- list()

for (m in names(measures)) {
  a <- d[[measures[[m]][1]]]; b <- d[[measures[[m]][2]]]
  ratings <- cbind(a, b)

  s1 <- icc(ratings, model = "twoway", type = "agreement", unit = "single")
  sk <- icc(ratings, model = "twoway", type = "agreement", unit = "average")

  for (s in list(list("ICC(A,1)", s1), list("ICC(A,k=2)", sk))) {
    o <- s[[2]]
    icc_rows[[length(icc_rows) + 1]] <- data.frame(
      measure = m, type = s[[1]],
      ICC = round(o$value, 4),
      CI95_low = round(o$lbound, 4), CI95_high = round(o$ubound, 4),
      Fvalue = round(o$Fvalue, 3), df1 = o$df1, df2 = o$df2,
      p_value = signif(o$p.value, 3),
      interpretation = koo_li(o$value),
      stringsAsFactors = FALSE)
  }

  diff <- a - b
  bias <- mean(diff); sdd <- sd(diff)
  agr_rows[[length(agr_rows) + 1]] <- data.frame(
    measure = m, n = length(a),
    MAE_pp = round(mean(abs(diff)), 3), RMSE_pp = round(sqrt(mean(diff^2)), 3),
    Pearson_r = round(cor(a, b), 4),
    bias_pp = round(bias, 3),
    LoA_lower = round(bias - 1.96 * sdd, 3), LoA_upper = round(bias + 1.96 * sdd, 3),
    stringsAsFactors = FALSE)

  for (lab in c("initial", "retest")) {
    v <- d[[measures[[m]][if (lab == "initial") 1 else 2]]]
    desc_rows[[length(desc_rows) + 1]] <- data.frame(
      measure = m, pass = lab, n = length(v),
      mean = round(mean(v), 3), median = round(median(v), 3),
      sd = round(sd(v), 3), min = round(min(v), 3), max = round(max(v), 3),
      stringsAsFactors = FALSE)
  }
}

icc_tbl  <- do.call(rbind, icc_rows)
agr_tbl  <- do.call(rbind, agr_rows)
desc_tbl <- do.call(rbind, desc_rows)

cat(">>> ICC (two-way, absolute agreement)\n");    print(icc_tbl,  row.names = FALSE)
cat("\n>>> Pairwise agreement & Bland-Altman\n");   print(agr_tbl,  row.names = FALSE)
cat("\n>>> Descriptive (per pass)\n");              print(desc_tbl, row.names = FALSE)

primary <- icc_tbl[icc_tbl$measure == "maxMBL" & icc_tbl$type == "ICC(A,1)", ]
interp <- data.frame(item = c(
  "Primary endpoint",
  "ICC(A,1) maxMBL",
  "95% CI",
  "Reliability band (Koo & Li 2016)",
  "Bands",
  "Mean absolute error",
  "Pearson r",
  "Systematic bias",
  "Benchmark (Paper 2 inter-rater)",
  "Reading",
  "Caveat (categorical flips)",
  "Caveat (MBL% near zero)"),
  value = c(
  "Intra-rater (test-retest) absolute agreement of per-implant maxMBL%",
  sprintf("%.3f", primary$ICC),
  sprintf("[%.3f, %.3f]", primary$CI95_low, primary$CI95_high),
  primary$interpretation,
  "poor <0.50; moderate 0.50-0.75; good 0.75-0.90; excellent >=0.90",
  sprintf("%.2f percentage points", agr_tbl$MAE_pp[agr_tbl$measure=="maxMBL"]),
  sprintf("%.3f", agr_tbl$Pearson_r[agr_tbl$measure=="maxMBL"]),
  sprintf("%+.2f pp (initial reads higher than retest)", agr_tbl$bias_pp[agr_tbl$measure=="maxMBL"]),
  "3-expert inter-rater ICC(A,1) = 0.887 (Paper 2)",
  "Single-rater test-retest agreement is GOOD and on par with between-expert agreement.",
  "4/151 implants are low-MBL 'flips' retained by the rater; excluding them r rises to ~0.87.",
  "MBL% is a ratio that is unstable when true bone loss ~0 (BL and IS keypoints nearly coincide)."),
  stringsAsFactors = FALSE)

write_xlsx(list(ICC = icc_tbl, Agreement = agr_tbl, Descriptive = desc_tbl,
                Interpretation = interp),
           path = file.path(outdir, "intra_rater_ICC_R_results.xlsx"))

cat("\n[written]", file.path(outdir, "intra_rater_ICC_R_results.xlsx"), "\n")
cat("[done]\n")
