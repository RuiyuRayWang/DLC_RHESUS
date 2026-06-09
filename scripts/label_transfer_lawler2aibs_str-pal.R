# Re-analysis of GSE157359 to elucidate the identity of GPe-Pvalb population in AIBS taxonomy
library(Seurat)
library(SeuratDisk)

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

saveRDS(query, 'data/Lawler_2020/processed/query_with_preds.rds')
saveRDS(reference, 'data/nemo_archive_processed/reference_STR_PAL.rds')


