# GeoGCL
This code describes the process of GeoGCL


## Required Packages

# Generate Node embeddings
pip install pytorch-metric-learning


pip install torch_geometric


pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-$(python -c 'import torch; print(torch.__version__)').html

All required packages can be installed directly from the `.ipynb` file — simply run the first three installation cells before executing the rest of the notebook.
A dummy graph: Graph.pkl is provided.

**Run the PNAConv.ipynb file to generate the node embeddings**


Open the `.ipynb` file and run all code cells sequentially from top to bottom to execute the complete code. The node embedding will be stored in the file : embedding_fairness_mul.pkl


All other required files are present in this directory.

# Finding top k geographical inclusive influencers 

Run the code inside the folder Seed_finder as : **python seed_nodes.py 10 0.4** to find top 10 geographical inclusive influencer

Here, 10 is the number of top k inclusive influencers, and 0.4 is the threshold beta.

