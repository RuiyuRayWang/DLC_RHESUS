library(tidyverse)
library(ggrepel)
library(readxl)
library(fs)
library(ggpubr)

library(extrafont)
# font_import()
loadfonts(device = "postscript")

mouse_list <- read_excel("data/ZHY_GPe_2025/Mouse_list_selected.xlsx") |>
  mutate(Virus_plasmid = str_remove_all(Virus, "2/9-")) |>
  fill(everything(), .direction = "down")
primers <- read_excel("design/Primer_selected.xlsx")

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

# Only work on subject with both spec and effi profiles
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
    
    # If file is empty, create a minimal row to track it
    if(nrow(data) == 0) {
      cat("  File is empty - recording as zero counts\n")
      return(tibble(
        mouse_id = extract_mouse_id(file),
        inj_side = extract_inj_side(file),
        filename = basename(file),
        Classification = NA_character_
      ))
    }
    
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
  filter(!is.na(Classification)) |>  # Remove NA classifications before pivoting
  group_by(filename, Classification) |>
  summarise(count = n(), .groups = "drop") |>
  # Ensure all files are represented, even those with zero counts
  right_join(
    data_spec |> distinct(filename, mouse_id, inj_side),
    by = "filename"
  ) |>
  pivot_wider(names_from = Classification, values_from = count, values_fill = 0)

# Compute specificity
classification_counts_spec <- classification_counts_spec |>
  mutate(
    GFP = if(!"GFP" %in% names(classification_counts_spec)) 0 else GFP,
    `GFP/Ai14` = if(!"GFP/Ai14" %in% names(classification_counts_spec)) 0 else `GFP/Ai14`
  ) |>
  mutate(
    total = GFP + `GFP/Ai14`,
    `specificity(%)` = if_else(total == 0, 0, `GFP/Ai14` / total * 100)
  )

# ==============================================================================

# Read completeness files and combine
data_effi <- map_dfr(files_effi, function(file) {
  if(basename(file) %in% files_comm){
    cat("Reading:", file, "\n")
    
    # Read the file
    data <- read_tsv(file, show_col_types = FALSE)
    
    # If file is empty, create a minimal row to track it
    if(nrow(data) == 0) {
      cat("  File is empty - recording as zero counts\n")
      return(tibble(
        mouse_id = extract_mouse_id(file),
        inj_side = extract_inj_side(file),
        filename = basename(file),
        Classification = NA_character_
      ))
    }
    
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
  filter(!is.na(Classification)) |>  # Remove NA classifications before pivoting
  group_by(filename, Classification) |>
  summarise(count = n(), .groups = "drop") |>
  # Ensure all files are represented, even those with zero counts
  right_join(
    data_effi |> distinct(filename, mouse_id, inj_side),
    by = "filename"
  ) |>
  pivot_wider(names_from = Classification, values_from = count, values_fill = 0)

# Compute completeness
classification_counts_effi <- classification_counts_effi |>
  mutate(
    Ai14 = if(!"Ai14" %in% names(classification_counts_effi)) 0 else Ai14,
    `GFP/Ai14` = if(!"GFP/Ai14" %in% names(classification_counts_effi)) 0 else `GFP/Ai14`
  ) |>
  mutate(
    total = Ai14 + `GFP/Ai14`,
    `completeness(%)` = if_else(total == 0, 0, `GFP/Ai14` / total * 100)
  )

# ==============================================================================

classification_counts <- left_join(
  x = classification_counts_effi |> select(filename, mouse_id, inj_side, `completeness(%)`), 
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
cls_counts <- cls_counts |> relocate(`completeness(%)`, .after = `specificity(%)`)

# Save if not already
cls_counts |> write_csv(file = "data/ZHY_GPe_2025/cls_counts.csv")

# Define label texts
cls_counts_lab <- cls_counts |> 
  filter(Category == "orig") |>
  filter(!mouse_id %in% c(1871))
cls_counts_lab$Name <- ""
idx_label <- match(c("1884",  # Gpr139_20: CiE1723
                     "1887",  # Gpr174_1: CiE1726
                     "2008",  # Gpr174_6-2: CiE1733
                     "1877"  # Gpr139_6-7: CiE1716
                     # "1871",  # Adamts19
                     # "1873",  # Gm2115_4: CiE1712
                     # "1874",  # Gm2115_6: CiE1713
                     # "2014"   # Gpr174_11: CiE1739
                     ), cls_counts_lab$mouse_id)
# Refer to primer_design_gpe-pv_20250825.xlsx for enhancer index
cls_counts_lab$Name[idx_label] <- c("CiE1723",
                                    "CiE1726",
                                    "CiE1733",
                                    "CiE1716"
                                    # "Adamts19 Enh-3",
                                    # "CiE1712",
                                    # "CiE1713",
                                    # "CiE1739"
                                    )

# Color assignment
cls_counts_lab$Color <- dplyr::case_when(
  cls_counts_lab$Name %in% c("CiE1716", "CiE1723") ~ "red2",
  cls_counts_lab$Name %in% c("CiE1726", "CiE1733") ~ "green3",
  TRUE ~ "grey60"
)

# Scatter plot for 1st round screening (orig seqs)
cls_counts_lab |>
  ggplot(mapping = aes(x = `completeness(%)`, y = `specificity(%)`, label = Name)) +
  geom_text_repel(min.segment.length = 0, seed = 42, box.padding = 0.6, force = 5, size = 5) +
  # geom_point(color = ifelse(cls_counts_lab$Name == "", "gray60", "red")) +
  geom_point(color = cls_counts_lab$Color) +
  annotate("text", x = 98, y = 1.5, label = "(n=37)") +
  xlab("Completness(%)") +
  ylab("Specificity(%)") +
  xlim(c(0,108)) + ylim(c(0,70)) + 
  theme_linedraw(
    base_family = "Arial", 
    base_size = 16
    ) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(linetype = "dashed", size = .5, color = "gray80"),
    axis.text = element_text(size = 14),
  )

ggsave(filename = "scatter_spec_effi_4-labels.eps",
       path = "figures/Fig1/",
       dpi = 1000,
       width = 3.5,
       height = 3.5,
       family = "Arial")

# ==============================================================================
# Investigate orig/evo/3xCore

cls_counts_orig_evo_3xCore_use <- cls_counts |>
  filter(!cCRE %in% c("Isl1_3", "Adamts19_7", "Gpr174_14")) |>
  # filter(!mouse_id %in% c(1866,1850)) |>
  filter(!filename %in% c("2022_1_4-3_L_Maximum intensity projection.ome.txt"))  # gender mismatch
cls_counts_orig_evo_3xCore_use$Category <- factor(cls_counts_orig_evo_3xCore_use$Category,
                                                  levels = c("orig","evo","3xCore"))

orig_evo_3xCore_summary_data <- cls_counts_orig_evo_3xCore_use |>
  group_by(Category) |>
  summarise(
    mean_spec = mean(`specificity(%)`, na.rm = TRUE),
    se_spec = sd(`specificity(%)`, na.rm = TRUE) / sqrt(sum(!is.na(`specificity(%)`))),
    sd_spec = sd(`specificity(%)`, na.rm = TRUE),
    mean_comp = mean(`completeness(%)`, na.rm = TRUE),
    se_comp = sd(`completeness(%)`, na.rm = TRUE) / sqrt(sum(!is.na(`completeness(%)`))),
    sd_comp = sd(`completeness(%)`, na.rm = TRUE)
    )

# Plot Specificity
ggplot() +
  geom_violin(data = cls_counts_orig_evo_3xCore_use,
              mapping = aes(x = Category, y = `specificity(%)`, fill = Category),
              width = 0.6, linewidth = 0.8, alpha = 0.2) +
  geom_boxplot(cls_counts_orig_evo_3xCore_use,
               mapping = aes(x = Category, y = `specificity(%)`),
               width = 0.1, linewidth = 0.5, color = "gray30", alpha = .5, outliers = FALSE) +
  geom_line(data = cls_counts_orig_evo_3xCore_use, 
            mapping = aes(x = Category, y = `specificity(%)`, group = cCRE), 
            alpha = 0.1, linewidth = 0.5) +
  geom_point(data = cls_counts_orig_evo_3xCore_use, 
             mapping = aes(x = Category, y = `specificity(%)`, group = cCRE, color = cCRE), 
             alpha = 0.5, size = 1.5,
             position = position_jitterdodge(jitter.width = 0.3, jitter.height = 0, dodge.width = 0.6)) +
  scale_y_continuous(limits = c(0,100), expand = c(0,0), breaks = seq(0, 100, 25), name = "Specificity (%)") +
  scale_x_discrete(name = NULL, labels = c("Natural","Evolved","3×Core")) +
  scale_fill_manual(values = c("white","white","white")) +
  theme_linedraw(base_size = 24, base_family = "Arial") +
  theme(
    legend.position = "none"
  )

ggsave(filename = "violin_spec_orig-evo-3xCore.svg",
       path = "figures/FigS3/",
       dpi = 600,
       width = 6,
       height = 6)

# Plot Completeness
ggplot() +
  geom_violin(data = cls_counts_orig_evo_3xCore_use,
              mapping = aes(x = Category, y = `completeness(%)`, fill = Category),
              width = 0.6, linewidth = 0.8, alpha = 0.2) +
  geom_boxplot(cls_counts_orig_evo_3xCore_use,
               mapping = aes(x = Category, y = `specificity(%)`),
               width = 0.1, linewidth = 0.5, color = "gray30", alpha = .5, outliers = FALSE) +
  geom_line(data = cls_counts_orig_evo_3xCore_use, 
            mapping = aes(x = Category, y = `completeness(%)`, group = cCRE), 
            alpha = 0.1, size = 0.5) +
  geom_point(data = cls_counts_orig_evo_3xCore_use, 
             mapping = aes(x = Category, y = `completeness(%)`, group = cCRE, color = cCRE), 
             alpha = 0.5, size = 1.5,
             position = position_jitterdodge(jitter.width = 0.3, jitter.height = 0, dodge.width = 0.65)) +
  scale_y_continuous(limits = c(0,100), expand = c(0,0), breaks = seq(0, 100, 25), name = "Completeness (%)") +
  scale_x_discrete(name = NULL, labels = c("Natural","Evolved","3×Core")) +
  scale_fill_manual(values = c("white","white","white")) +
  theme_linedraw(base_size = 24, base_family = "Arial") +
  theme(
    legend.position = "none"
  )

ggsave(filename = "violin_effi_orig-evo-3xCore.svg",
       path = "figures/FigS3/",
       dpi = 600,
       width = 6,
       height = 6)

cls_counts_orig_evo_3xCore_use |>
  select(c(`specificity(%)`,`completeness(%)`,Category,cCRE)) |>
  pivot_wider(names_from = Category, values_from = c(`specificity(%)`,`completeness(%)`)) |>
  relocate(`specificity(%)_orig`, .before = `specificity(%)_evo`) |>
  relocate(`completeness(%)_orig`, .before = `completeness(%)_evo`) |>
  View()



