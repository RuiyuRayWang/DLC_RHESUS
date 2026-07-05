#!/usr/bin/env Rscript

# PD monkey behavior titration plots from selected_time_window.json files.
#
# RStudio/Posit usage:
#   1) Open this script.
#   2) Edit monkey_id/base_dir if needed.
#   3) Run the whole script.
#   4) Plots will appear in the Plots panel.

suppressPackageStartupMessages({
  library(jsonlite)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
})

# ---- User settings ------------------------------------------------------------

monkey_id <- "H1705011"
base_dir <- "results/oft_analysis"

# ---- Load daily selected_time_window.json ------------------------------------

monkey_dir <- file.path(base_dir, monkey_id)
if (!dir.exists(monkey_dir)) {
  stop(paste("Monkey directory not found:", monkey_dir), call. = FALSE)
}

session_dirs <- list.dirs(monkey_dir, full.names = TRUE, recursive = FALSE)
session_dirs <- session_dirs[grepl("/[0-9]{4}-[0-9]{2}-[0-9]{2}$", session_dirs)]
session_dirs <- sort(session_dirs)

if (length(session_dirs) == 0) {
  stop("No daily session folders found under monkey directory.", call. = FALSE)
}

rows <- list()
for (dir_path in session_dirs) {
  json_path <- file.path(dir_path, "selected_time_window.json")
  if (!file.exists(json_path)) {
    next
  }

  rec <- fromJSON(json_path, simplifyVector = TRUE)
  session_date <- basename(dir_path)
  if (!is.null(rec$session_date) && nzchar(rec$session_date)) {
    session_date <- rec$session_date
  }

  rows[[length(rows) + 1]] <- tibble(
    session_date = as.Date(session_date),
    mean_velocity_moving_cm_s = as.numeric(rec$mean_velocity_moving_cm_s),
    distance_cm = as.numeric(rec$distance_cm),
    max_velocity_cm_s = as.numeric(rec$max_velocity_cm_s),
    n_movement_bouts = as.numeric(rec$n_movement_bouts),
    mean_bout_duration_sec = as.numeric(rec$mean_bout_duration_sec),
    immobility_time_sec = as.numeric(rec$immobility_time_sec)
  )
}

if (length(rows) == 0) {
  stop("No selected_time_window.json files found for this monkey.", call. = FALSE)
}

wide_df <- bind_rows(rows) %>%
  arrange(.data$session_date)

metric_labels <- c(
  mean_velocity_moving_cm_s = "Mean velocity (moving; cm/s)",
  distance_cm = "Distance traveled (cm)",
  max_velocity_cm_s = "Max velocity (cm/s)",
  n_movement_bouts = "Movement bouts",
  mean_bout_duration_sec = "Mean bout duration (s)",
  immobility_time_sec = "Immobility time (s)"
)

long_df <- wide_df %>%
  pivot_longer(
    cols = names(metric_labels),
    names_to = "metric_key",
    values_to = "value"
  ) %>%
  mutate(metric = factor(metric_labels[.data$metric_key], levels = unname(metric_labels)))

# Build segmented x-axis with explicit "..." gap tokens between non-consecutive dates.
date_ticks <- sort(unique(wide_df$session_date))
date_labels <- format(date_ticks, "%Y-%m-%d")

x_levels <- c(date_labels[1])
segment_idx <- 1L
segment_ids <- setNames(integer(length(date_labels)), date_labels)
segment_ids[[date_labels[1]]] <- segment_idx

if (length(date_ticks) > 1) {
  for (i in 2:length(date_ticks)) {
    day_gap <- as.integer(date_ticks[i] - date_ticks[i - 1])
    if (!is.na(day_gap) && day_gap > 1) {
      x_levels <- c(x_levels, paste0("gap_", i))
      segment_idx <- segment_idx + 1L
    }
    x_levels <- c(x_levels, date_labels[i])
    segment_ids[[date_labels[i]]] <- segment_idx
  }
}

long_df <- long_df %>%
  mutate(
    date_label = format(.data$session_date, "%Y-%m-%d"),
    x_axis = factor(.data$date_label, levels = x_levels),
    segment_id = unname(segment_ids[.data$date_label])
  )

x_axis_labels <- setNames(
  ifelse(grepl("^gap_", x_levels), "...", x_levels),
  x_levels
)

# Expose data frames in global env for quick inspection in RStudio.
pd_titration_wide_df <- wide_df
pd_titration_long_df <- long_df

cat("Loaded", nrow(wide_df), "session(s) for", monkey_id, "\n")
print(wide_df)

# ---- Plot 1: combined multi-panel --------------------------------------------

combined_plot <- ggplot(long_df, aes(x = .data$x_axis, y = .data$value)) +
  geom_line(aes(group = interaction(.data$metric_key, .data$segment_id)), color = "#1f77b4", linewidth = 0.8, na.rm = TRUE) +
  geom_point(color = "#1f77b4", size = 2.3, na.rm = TRUE) +
  facet_wrap(~ metric, scales = "free_y", ncol = 2) +
  scale_x_discrete(drop = FALSE, labels = x_axis_labels) +
  labs(
    title = paste("PD Monkey Behavior Titration -", monkey_id),
    x = "Experiment date",
    y = NULL
  ) +
  theme_bw(base_size = 11) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text = element_text(face = "bold"),
    plot.title = element_text(face = "bold")
  )

print(combined_plot)

# ---- Plots 2-7: individual metrics -------------------------------------------

for (mk in names(metric_labels)) {
  sub_df <- long_df %>% filter(.data$metric_key == mk)

  p <- ggplot(sub_df, aes(x = .data$x_axis, y = .data$value)) +
    geom_line(aes(group = .data$segment_id), color = "#1f77b4", linewidth = 0.9, na.rm = TRUE) +
    geom_point(color = "#1f77b4", size = 2.5, na.rm = TRUE) +
    scale_x_discrete(drop = FALSE, labels = x_axis_labels) +
    labs(
      title = paste(metric_labels[[mk]], "-", monkey_id),
      x = "Experiment date",
      y = metric_labels[[mk]]
    ) +
    theme_bw(base_size = 11) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      plot.title = element_text(face = "bold")
    )

  print(p)
}
