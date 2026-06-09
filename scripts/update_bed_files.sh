#!/bin/bash
# Script to add gene names to the 4th column of BED files

for file in analysis/visibleData_NDB_*.bed; do
    # Extract gene name from filename
    gene_name=$(basename "$file" .bed | sed 's/visibleData_NDB_//')
    
    # Create a temporary file with modified content
    awk -v gene="$gene_name" 'BEGIN{OFS="\t"} {$4 = gene "_" $4; print}' "$file" > "${file}.tmp"
    
    # Replace original file with modified content
    mv "${file}.tmp" "$file"
    
    echo "Updated $file with gene name: $gene_name"
done
