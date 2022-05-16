# omicsGAT
omicsGAT is a graph attention network based framework for cancer subtype analysis. It performs the task of classification or clustering of patient/cell samples based on the gene expression. It strives to secure important information while discarding the rest by assigning different attention coefficients to the neighbors of a sample in a network/graph.

## Code descriptions
**main.py** file can be run to train and evaluate the omicsGAT model for classification or clustering. This file takes the inputs and calls the other functions from different files based on the task and other options selected.\
**classification.py** file runs the classification task. It splits the input dataset into train, test and validation set. Then it calls function from the **cls_model.py** file to create the omicsGAT classification model, trains it and print out the AUROC score along with loss for the test set.\
Similarly, **clustering.py** file runs clustering (bulk RNA-seq or scRNA-seq). It calls function from **clustering_model.py** file to create the omicsGAT clustering model, train it and then prints out the NMI and ARI score of the model.\
**layers.py** contains code of a single layer of omicsGAT and is common to both classification and clustering.
**cls_utils.py** and **clustering_utils.py** contains some extra functions used for classification and clustering respectively.

## Sample data
BRCA data for classification: https://drive.google.com/drive/folders/1wEOhmtMVt-S-2mKTqsQVXD1Chnxqn80T?usp=sharing \
BLCA data for bulk RNA-seq clustering: https://drive.google.com/drive/folders/163vmubMxpg2yd22IMNK9skcQk03frQlX?usp=sharing \
Single-cell RNA-seq data for clustering: https://drive.google.com/drive/folders/1wv8eHr3G93GEOO2YDoCg2IRytNf6RS_1?usp=sharing

## Input data format

The 'task' type can be selected using command line option '--task'. Default task is set to be 'classification'. If 'clustering' is selected as the task then the clustering type can be selected between 'bulk' and 'single_cell' using the '--clustering_type' option.
