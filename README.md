# omicsGAT
omicsGAT is a graph attention network based framework for cancer subtype analysis. It performs the task of classification or clustering of patient/cell samples based on the gene expression. It strives to secure important information while discarding the rest by assigning different attention coefficients to the neighbors of a sample in a network/graph.

## Workflow
![alt text](https://github.com/compbiolabucf/omicsGAT/blob/main/omicsGAT_workflow.png)


## Required Python libraries
- Python (>= 3.9.7)
- Pytorch (>= 1.11) [with cudatoolkit (>= 11.3.1) if cuda is used]
- scikit-learn (>= 1.0.2)
- scipy (>= 1.7.1)
- pandas (>= 1.3.4)
- numpy (>= 1.20.3)

## Code descriptions
**main.py** file can be run to train and evaluate the omicsGAT model for classification or clustering. This file takes the inputs and calls the other functions from different files based on the task and other options selected.\
**classification.py** file runs the classification task. It splits the input dataset into train, test and validation set. Then it calls function from the **cls_model.py** file to create the omicsGAT classification model, trains it and print out the AUROC score along with loss for the test set.\
Similarly, **clustering.py** file runs clustering (bulk RNA-seq or scRNA-seq). It calls function from **clustering_model.py** file to create the omicsGAT clustering model, train it and then prints out the NMI and ARI score of the model.\
**layers.py** contains code of a single layer of omicsGAT and is common to both classification and clustering.
**cls_utils.py** and **clustering_utils.py** contains some extra functions used for classification and clustering respectively. The user has to change the file path in the 'load_data()' function of these files in order use the respective dataset for classification or clustering.

## Sample data
BRCA data for classification: https://drive.google.com/drive/folders/1wEOhmtMVt-S-2mKTqsQVXD1Chnxqn80T?usp=sharing \
BLCA data for bulk RNA-seq clustering: https://drive.google.com/drive/folders/163vmubMxpg2yd22IMNK9skcQk03frQlX?usp=sharing \
Single-cell RNA-seq data for clustering: https://drive.google.com/drive/folders/1wv8eHr3G93GEOO2YDoCg2IRytNf6RS_1?usp=sharing \
If used, these datasets should be placed in the same directory as that of the codes ('.py' files). Note that, these datasets are already preprocessed. As mentioned in the paper, the features of the BRCA data comprises of gene expression selected using correlation whereas, features of BLCA and scRNA data comprises of PCA components. Therefore, no preprocessing needs to be done for them.

## Input data format
All input data should be provided in '.csv' format and placed in the directory specified in the 'load_data()' function of **cls_utils.py** or **clustering_utils.py** file for the respective task. Input for a specific task consists of the feature matrix, binary adjacency matrix and label data.\
The feature matrix is of the shape _sample_\*_feature_. The features can be gene-expression/ PCA components/ any other feature. An example feature matrix consisting of three samples and four features is given below:
|         | Feature1 | Feature2 | Feature3 | Feature4 |
|---------|----------|----------|----------|----------|
| Sample1 |     -    |     -    |     -    |     -    |
| Sample2 |     -    |     -    |     -    |     -    |
| Sample3 |     -    |     -    |     -    |     -    |

The binary adjacency matrix is of shape _sample_\*_sample_ representing if a connection is present between the corresponding two samples. An adjacency matrix for the example given above can be:
|         | Sample1 | Sample2 | Sample3 |
|---------|---------|---------|---------|
| Sample1 |    -    |    -    |    -    |
| Sample2 |    -    |    -    |    -    |
| Sample3 |    -    |    -    |    -    |

The label data can be a series vector of shape _sample_\*1. A label vector for the above exmample can be:
|         | Label |
|---------|-------|
| Sample1 |   -   |
| Sample2 |   -   |
| Sample3 |   -   |

The classification class label should be binary (i.e. 'Positive', 'Negative). For example, for 'ER', 'Sample1' could be 'Positive' or 'Negative'. For the clastering part, integer multi-label can be used depending on the number of clusters. For the sample bulk RNA-seq dataset (BLCA), '0,1,2,3,4' labels are being used.

## Running the code
The omicsGAT model can be run using the command line interface. User only needs to run the 'main.py' script. A number of options or flags are available to modify the model or traiing process. \
'--task' : Using this option one can select between 'classification' and 'clustering' tasks. Default is 'classification'.\
'--selection' : For the 'classification' task, if multiple labels are present ('ER', 'TN' etc), this option lets the user select the intended label. Default is set to 'ER'. If there is a single label, this option should be set to 'NULL'.\
'--clustering_type' : If 'clustering' is selected as task, then the type of clustering ('bulk' or 'single_cell') can be selected using this option. Default is 'bulk'.\
'--nb_heads' and '--embed' : Provides the number of heads and the embedding size of each head for the model. Default is set to 8 for both of them.\
'--nb_clusters' : Selects the number of clusters for the stratification task. Default is 5.\
'--clustering_affn' and '--clustering_dist' : Hyperparameters used for the hierarchical clustering from _scikit-learn_ library. Default is set to 'manhattan' and 'aveerage' respectively.\
The other options can be used to modify the training process of the model. 

**Command examples**\
An example for running the classification task: python main.py --task classification --selection ER\
An example for running bulk RNA-seq clustering: python main.py --task clustering --clustering_type bulk --nb_clusters 5 --nb_heads 64 --embed 64\
An example for running scRNA-seq clustering: python main.py --task clustering --clustering_type single_cell --nb_clusters 6 --cluster_affn cosine
