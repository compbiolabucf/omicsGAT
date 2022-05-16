# omicsGAT
omicsGAT is a graph attention network based framework for cancer subtype analysis. It performs the task of classification or clustering of patient/cell samples based on the gene expression. It strives to secure important information while discarding the rest by assigning different attention coefficients to the neighbors of a sample in a network/graph.

## Sample data
BRCA data for classification: https://drive.google.com/drive/folders/1wEOhmtMVt-S-2mKTqsQVXD1Chnxqn80T?usp=sharing
BLCA data for bulk RNA-seq clustering: https://drive.google.com/drive/folders/163vmubMxpg2yd22IMNK9skcQk03frQlX?usp=sharing
single-cell RNA-seq data for clustering: https://drive.google.com/drive/folders/1wv8eHr3G93GEOO2YDoCg2IRytNf6RS_1?usp=sharing

## Code descriptions



The 'main.py' file can be run to train and evaluate the omicsGAT model for classification or clustering. 
The 'task' type can be selected using command line option '--task'. Default task is set to be 'classification'. If 'clustering' is selected as the task then the clustering type can be selected between 'bulk' and 'single_cell' using the '--clustering_type' option.
