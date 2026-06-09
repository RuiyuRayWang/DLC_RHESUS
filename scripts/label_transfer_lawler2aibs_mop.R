library(Seurat)
library(SeuratDisk)

setwd('/home/luolab/PROJECTS/PAL_ENHANCER/')
lawler <- Read10X('data/Lawler_2020/processed/lawler_10x_format/')
meta_lawler <- read.csv('data/Lawler_2020/processed/lawler_10x_format/metadata.csv', row.names = 1)
aibs_mop <- Read10X('data/nemo_archive_processed/MOp_10x_format/')
meta_aibs_mop <- read.csv('data/nemo_archive_processed/MOp_10x_format/metadata.csv', row.names = 1)

query <- CreateSeuratObject(lawler, meta.data = meta_lawler)
reference <- CreateSeuratObject(aibs_mop, meta.data = meta_aibs_mop)

# pre-process reference dataset
reference <- NormalizeData(reference)
reference <- FindVariableFeatures(reference)
reference <- ScaleData(reference)
reference <- RunPCA(reference)
reference <- FindNeighbors(reference, dims = 1:30)
reference <- RunUMAP(reference, dims = 1:30)

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
  k.weight = 25,
  dims = 1:30
)
query <- AddMetaData(query, metadata = predictions)

