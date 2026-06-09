library(tidyverse)
library(ggrepel)
library(readxl)
library(fs)

mouse_list <- read_excel("data/ZHY_GPe_2025/Mouse_list_selected.xlsx") |>
  mutate(Virus_plasmid = str_remove_all(Virus, "2/9-"))
primers <- read_excel("data/ZHY_GPe_2025/Primer_selected.xlsx")

mouse_to_primers <- left_join(
  x = mouse_list |> 
    select(Mouse_id, Virus_plasmid) |> 
    separate_rows(Virus_plasmid, sep = "; ") |>
    mutate(
      Inj_side = str_extract(Virus_plasmid, "^[LR]"),
      Virus_plasmid = str_extract(Virus_plasmid, "\\d+")
    ),
  y = primers |> 
    mutate(across(Number, as.character)),
  by = join_by(Virus_plasmid == Number)
)

# Set the directory containing effi/spec files
data_dir <- "data/ZHY_GPe_2025/"

# Get all .ome.txt files
files_spec <- dir_ls(data_dir, glob = "*_spec/results/*.ome.txt", recurse = TRUE)
files_effi <- dir_ls(data_dir, glob = "*_effi/results/*.ome.txt", recurse = TRUE)

# Only work on intersect files
files_comm <- intersect(basename(files_spec), basename(files_effi))

# Function to extract mouse ID from filename
extract_mouse_id <- function(filename) {
  # Extract the 4-digit number at the beginning
  str_extract(basename(filename), "^\\d{4}") %>% as.numeric()
}

# # Function to extract slice index from filename
# extract_inj_side <- function(filename) {
#   # Extract the injection side following slice position
#   str_extract(basename(filename), "^\\d{4}") %>% as.numeric()  # TODO
# }

# Function to extract injection side from filename
extract_inj_side <- function(filename) {
  # Extract the injection side following slice position
  str_extract(basename(filename), ".*_([A-Z]{1})\\_Maximum*", group = 1) %>% as.character()
}

# ==============================================================================

# Read specificity files and combine
data_spec <- map_dfr(files_spec, function(file) {
  if(basename(file) %in% files_comm){
    cat("Reading:", file, "\n")
    
    # Read the file
    data <- read_tsv(file, show_col_types = FALSE)
    
    # Add mouse and image IDs
    data %>%
      mutate(
        mouse_id = extract_mouse_id(file),
        inj_side = extract_inj_side(file),
        filename = basename(file)
      )
  }
})

# Count classifications by img
classification_counts_spec <- data_spec |>
  group_by(filename, Classification) |>
  summarise(count = n(), .groups = "drop") |>
  pivot_wider(names_from = Classification, values_from = count, values_fill = 0) |>
  mutate(mouse_id = extract_mouse_id(filename), inj_side = extract_inj_side(filename))

# Compute specificity
classification_counts_spec <- classification_counts_spec |>
  mutate(total = `GFP` + `GFP/Ai14`) |>
  mutate(`specificity(%)` = `GFP/Ai14` / total * 100)

# ==============================================================================

# Read efficiency files and combine
data_effi <- map_dfr(files_effi, function(file) {
  if(basename(file) %in% files_comm){
    cat("Reading:", file, "\n")
    
    # Read the file
    data <- read_tsv(file, show_col_types = FALSE)
    
    # Add mouse and image IDs
    data %>%
      mutate(
        mouse_id = extract_mouse_id(file),
        inj_side = extract_inj_side(file),
        filename = basename(file)
      )
  }
})

# Count classifications by img
classification_counts_effi <- data_effi |>
  group_by(filename, Classification) |>
  summarise(count = n(), .groups = "drop") |>
  pivot_wider(names_from = Classification, values_from = count, values_fill = 0) |>
  mutate(mouse_id = extract_mouse_id(filename), inj_side = extract_inj_side(filename))

# Compute efficiency
classification_counts_effi <- classification_counts_effi |>
  mutate(total = `Ai14` + `GFP/Ai14`) |>
  mutate(`efficiency(%)` = `GFP/Ai14` / total * 100)

# ==============================================================================

classification_counts <- left_join(
  x = classification_counts_effi |> select(filename, mouse_id, inj_side, `efficiency(%)`), 
  y = classification_counts_spec |> select(filename, mouse_id, inj_side, `specificity(%)`), 
  by = c("filename", "mouse_id", "inj_side")
  )

cls_counts_1 <- classification_counts |> 
  filter(mouse_id < 2022) |>
  mutate(across(mouse_id, as.character)) |>
  left_join(y = mouse_to_primers, by = join_by(mouse_id == Mouse_id))
cls_counts_2 <- classification_counts |> 
  filter(mouse_id >= 2022) |>
  mutate(across(mouse_id, as.character)) |>
  left_join(y = mouse_to_primers, by = join_by(mouse_id == Mouse_id, inj_side == Inj_side))

cls_counts <- bind_rows(cls_counts_1, cls_counts_2)

cls_counts |>
  ggplot(mapping = aes(x = `efficiency(%)`, y = `specificity(%)`)) +
  geom_point(mapping = aes(color = Category)) +
  xlab("Efficiency(%)") +
  ylab("Specificity(%)") +
  theme_linedraw()

cls_counts_lab <- cls_counts
cls_counts_lab$Name <- ""
idx_label <- which(cls_counts$mouse_id %in% c("1823","1371","1903","1812","1914"))

# cls_counts_lab$Name[idx_label] <- cls_counts$Name[idx_label]
cls_counts_lab$Name[idx_label] <- c("Gpr139-20_evolved_r3 (1762)", 
                                    "Isl1_evolved_r3 (1581)",
                                    "Gpr139_13-14_evolved_r9_3xCore (1797)",
                                    "Gm2115_4_evolved_r9 (1751)",
                                    "Gpr174_3_evolved_r10_3xCore (1807)")

cls_counts_lab |>
  ggplot(mapping = aes(x = `efficiency(%)`, y = `specificity(%)`, label = Name)) +
  geom_text_repel(min.segment.length = 0, seed = 42, box.padding = 0.5) +
  geom_point(color = ifelse(cls_counts_lab$Name == "", "grey70", "red")) +
  xlab("Efficiency(%)") +
  ylab("Specificity(%)") +
  theme_linedraw()

cls_counts |>
  ggplot(mapping = aes(x = `efficiency(%)`, y = `specificity(%)`)) +
  geom_point(mapping = aes(color = Category)) +
  xlab("Efficiency(%)") +
  ylab("Specificity(%)") +
  xlim(c(0,100)) +
  ylim(c(0,100))

cls_counts |>
  filter(Category == 'orig') |>
  ggplot(mapping = aes(x = `efficiency(%)`, y = `specificity(%)`)) +
  geom_point(mapping = aes(color = Category)) +
  xlab("Efficiency(%)") +
  ylab("Specificity(%)") +
  xlim(c(0,100)) +
  ylim(c(0,100))

cls_counts |>
  filter(Category == 'evo') |>
  ggplot(mapping = aes(x = `efficiency(%)`, y = `specificity(%)`)) +
  geom_point(mapping = aes(color = Category)) +
  xlab("Efficiency(%)") +
  ylab("Specificity(%)") +
  xlim(c(0,100)) +
  ylim(c(0,100))

cls_counts |>
  filter(Category == 'evo') |>
  ggplot(mapping = aes(x = `efficiency(%)`, y = `specificity(%)`)) +
  geom_point(mapping = aes(color = Category)) +
  xlab("Efficiency(%)") +
  ylab("Specificity(%)") +
  xlim(c(0,100)) +
  ylim(c(0,100))