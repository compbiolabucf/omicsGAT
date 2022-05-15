# omicsGAT
Graph Attention Network for Cancer Subtype Analysis

Sample data available at https://drive.google.com/drive/folders/1iOSTRpxlaAdnuw30Vpgr_In7cO-QmXa4?usp=sharing

The 'main.py' file can be run to train and evaluate the omicsGAT model for classification or clustering. 
The 'task' type can be selected using command line option '--task'. Default task is set to be 'classification'. If 'clustering' is selected as the task then the clustering type can be selected between 'bulk' and 'single_cell' using the '--clustering_type' option.
