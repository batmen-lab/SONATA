# sonata
Source code for **SONATA: Disambiguated manifold alignment of single-cell data**  

**Updated on Nov. 1, 2024**: We have added more comprehensive tests for 5 baseline methods, which can be found in the *src/run_baselines* folder. We're also working on the new version of SONATA—coming soon! 

## Requirements
Dependencies for **SONATA** are recorded in *requirements.txt*.  

## Data
All datasets are available at [this link](https://drive.google.com/drive/folders/1DKDP2eSfWODHiFqmn2GQY4m-sNda5seg).

## Baseline Performance
All baseline method tests are implemented in the folder *src/run_baselines*. To run a test, use the following commands:
```python
cd src
python run_baselines/run_unioncom.py --dataset t_branch
```


## Examples
Jupyter notebooks to replicate the results from the manuscript are available under folder *examples*:  
- Simulation datasets
    - no ambiguous: [simulation_swiss_roll.ipynb](https://github.com/batmen-lab/SONATA/blob/main/examples/simulation_swiss_roll.ipynb)
    - all ambiguous: [simulation_circle.ipynb](https://github.com/batmen-lab/SONATA/blob/main/examples/simulation_circle.ipynb)
    - partial ambiguous: [simulation_t_branch.ipynb](https://github.com/batmen-lab/SONATA/blob/main/examples/simulation_t_branch.ipynb), [simulation_benz.ipynb](https://github.com/batmen-lab/SONATA/blob/main/examples/simulation_benz.ipynb), [simulation_cross.ipynb](https://github.com/batmen-lab/SONATA/blob/main/examples/simulation_cross.ipynb)
- Real bio datasets: [scGEM.ipynb](https://github.com/batmen-lab/SONATA/blob/main/examples/scGEM.ipynb), [SNARE-seq.ipynb](https://github.com/batmen-lab/SONATA/blob/main/examples/SNARE-seq.ipynb)

## Basic Use
Input for SONATA: *data* in form of numpy arrays/matrices, where the rows correspond to samples and columns correspond to features.
```python
import sonata
sn = sonata.sonata(kmin=10, sigma=0.1, t=0.1)
alter_mappings = sn.alter_mapping(data)
```

### Required parameters for sonata
- **k**: Number of neighbors to be used when constructing kNN graphs. Default=10. The number of neighbors k should be suffciently large to connect the corresponding k-NN graph   
- **sigma**: Bandwidth parameter for cell-wise ambiguity (Aij). Default=0.1.
- **t**: A threshold to ascertain the ambiguity status of individual cells before clustering them into groups. Default=0.1, with lower values resulting in stricter ambiguity classification.

### Optional parameters:
- **kmode**: Determine whether to use a connectivity graph (adjacency matrix of 1s/0s based on whether nodes are connected) or a distance graph (adjacency matrix entries weighted by distances between nodes). Default="distance"
- **kmetric**: Sets the metric to use while constructing nearest neighbor graphs. some possible choices are "euclidean", "correlation". Default= "euclidean".
- **kmax**: Maximum value of knn when constructing geodesic distance matrix. Default=200.
- **percnt_thres**: The percentile of the data distribution used in the calculation of the “virtual” cell. Default=95.
- **eval_knn**: Evaluate whether the alternative alignment distorts the data manifold by changing the mutual nearest neighbors of cells. Default=False.