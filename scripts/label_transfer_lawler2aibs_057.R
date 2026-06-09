library(Seurat)
library(SeuratDisk)
library(tidyverse)

setwd('/home/luolab/PROJECTS/PAL_ENHANCER/')
lawler <- Read10X('data/Lawler_2020/processed/lawler_10x_format/')
meta_lawler <- read.csv('data/Lawler_2020/processed/lawler_10x_format/metadata.csv', row.names = 1)
aibs_str_pal <- Read10X('data/nemo_archive_processed/STR-PAL_10x_format/')
meta_aibs_str_pal <- read.csv('data/nemo_archive_processed/STR-PAL_10x_format/metadata.csv', row.names = 1)

query <- CreateSeuratObject(lawler, meta.data = meta_lawler)
reference <- CreateSeuratObject(aibs_str_pal, meta.data = meta_aibs_str_pal)

rm(lawler, aibs_str_pal)
gc()

# pre-process reference dataset
reference <- NormalizeData(reference)
reference <- FindVariableFeatures(reference)
reference <- ScaleData(reference)
reference <- RunPCA(reference)
reference <- FindNeighbors(reference, dims = 1:30)
# reference <- RunUMAP(reference, dims = 1:30)

# pre-process query dataset
query <- NormalizeData(query)
anchors <- FindTransferAnchors(
  reference = reference, 
  query = query, 
  dims = 1:30,
  reference.reduction = "pca"
)
predictions <- TransferData(
  anchorset = anchors, 
  refdata = reference$subclass, 
  k.weight = 28,
  dims = 1:30
)
query <- AddMetaData(query, metadata = predictions)

query@meta.data |> View()

# SaveH5Seurat(object = query, filename = 'data/Lawler_2020/processed/query_with_preds.h5Seurat', overwrite = T)
# SaveH5Seurat(object = reference, filename = 'data/nemo_archive_processed/ref_aibs_STR_PAL.h5Seurat', overwrite = T)  # Seurat v5 incompatible with SeuratDisk?
saveRDS(query, 'data/Lawler_2020/processed/query_with_preds.rds')
# saveRDS(reference, 'data/nemo_archive_processed/ref_aibs_STR_PAL.h5Seurat')

# ==============================================================================

reference_057 <- readRDS('data/nemo_archive_processed/reference_057.rds')

# Subsetting 057
Idents(reference) <- "subclass"
reference_057 <- subset(reference, idents = "057 NDB-SI-MA-STRv Lhx8 Gaba")

# pre-process reference dataset
reference_057 <- NormalizeData(reference_057)
reference_057 <- FindVariableFeatures(reference_057)
reference_057 <- ScaleData(reference_057)
reference_057 <- RunPCA(reference_057)
reference_057 <- FindNeighbors(reference_057, dims = 1:20)
reference_057 <- FindClusters(reference_057, method = , resolution = 0.6)
reference_057 <- RunUMAP(reference_057, dims = 1:20)

DimPlot(reference_057, label = T) | FeaturePlot(reference_057, features = "Pvalb")

# pre-process query dataset
query <- NormalizeData(query)
anchors <- FindTransferAnchors(
  reference = reference_057, 
  query = query, 
  dims = 1:30,
  reference.reduction = "pca"
)
predictions <- TransferData(
  anchorset = anchors, 
  refdata = reference_057$seurat_clusters, 
  k.weight = 28,
  dims = 1:30
)
query <- AddMetaData(query, metadata = predictions)

query@meta.data |> View()

mks.gpe_pv_pos <- FindMarkers(reference_057, ident.1 = 8)
mks.gpe_pv_pos |> View()

SaveH5Seurat(object = query, filename = 'data/Lawler_2020/processed/query_to_057_with_preds.h5Seurat', overwrite = T)
SaveH5Seurat(object = reference_057, filename = 'data/nemo_archive_processed/ref_aibs_STR_PAL_subset_057.h5Seurat', overwrite = T)

# ==============================================================================

reference <- readRDS('data/nemo_archive_processed/reference_STR_PAL.rds')

Idents(reference) <- "cluster"
mks.057.by_cluster <- FindAllMarkers(reference, only.pos = TRUE)
write_csv(mks.057.by_cluster, file = "results/STR-PAL_mks_057_by_cluster.csv")

# ==============================================================================

reference_057 <- readRDS('data/nemo_archive_processed/reference_057.rds')

DimPlot(reference_057, cells.highlight = WhichCells(reference_057, idents = '0892 NDB-SI-MA-STRv Lhx8 Gaba_4')) + 
  NoLegend() + 
  ggtitle('0892 NDB-SI-MA-STRv Lhx8 Gaba_4') + theme(title = element_text(size = 10))

top_90 <- mks.057.by_cluster |> 
  filter(cluster == "0892 NDB-SI-MA-STRv Lhx8 Gaba_4") |> 
  slice_head(n = 90) |>
  pull("gene")

FeaturePlot(reference_057, features = top_90[1:9])
FeaturePlot(reference_057, features = top_90[10:18])
FeaturePlot(reference_057, features = top_90[19:27])
FeaturePlot(reference_057, features = top_90[28:36])
FeaturePlot(reference_057, features = top_90[37:45])
FeaturePlot(reference_057, features = top_90[46:54])
FeaturePlot(reference_057, features = top_90[55:63])
FeaturePlot(reference_057, features = top_90[64:72])
FeaturePlot(reference_057, features = top_90[73:81])
FeaturePlot(reference_057, features = top_90[82:90])
mks_use <- c('Pvalb', 'Gpr174', 'Gpr139', "Adamts19", "P2ry10b", 'Npas1', 'Isl1', 'Lhx6', 'Lhx8')
FeaturePlot(reference_057, features = mks_use)
FeaturePlot(reference_057, features = 'Gm2115')

# Try to investigate what other cell markers may be
cells.top_left <- CellSelector(DimPlot(reference_057))
mks.057.top_left <- FindMarkers(reference_057, ident.1 = cells.top_left)

cells.mid_left <- CellSelector(DimPlot(reference_057))
mks.057.mid_left <- FindMarkers(reference_057, ident.1 = cells.mid_left)

cells.left <- CellSelector(DimPlot(reference_057))
mks.057.left <- FindMarkers(reference_057, ident.1 = cells.left)

