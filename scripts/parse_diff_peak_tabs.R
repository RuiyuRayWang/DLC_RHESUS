library(tidyverse)

diff_peaks <- read_csv("results/ACA-PFC_334_Microglia_NN_diff_peaks_pos.csv")
closest_feat <- read_tsv(
  "results/ACA-PFC_334_Microglia_NN_diff_peaks_pos.closest.gencode.vM23.annotation.gene.protein_coding.bed",
  col_names = c('chr','start','end','closest_feature')
  ) |>
  unite('chr_start', chr, start, sep = ":") |>
  unite('feature name', chr_start, end, sep = "-") |>
  group_by(`feature name`) |>
  summarise(
    closest_feature = paste(closest_feature, collapse = ", "),
    .groups = "drop"
  )

diff_peaks <- left_join(diff_peaks, closest_feat, by = 'feature name')

diff_peaks |> write_csv("results/ACA-PFC_334_Microglia_NN_diff_peaks_pos_closest-feat.csv")