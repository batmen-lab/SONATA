# SONATA
Source code for **Securing diagonal integration of multimodal single-cell data against ambiguous mapping**  

![SONATA](images/overview.jpg)

## Requirements
Dependencies for **SONATA** are recorded in *requirements.txt*.  

## Data
The datasets used in this project are available for download at the following link: [data](https://drive.google.com/drive/folders/1YWvcBaJ-yj76OjkcMz8cfKchKwuJmkGV?usp=sharing).  

Then organize the project as follows:

```
project_root/
├── src/
│   ├── examples/
│   │   ├── baselines/
│   │   ├── cfgs/
│   │   ├── noise_scale.ipynb
│   │   ├── simulation_t_branch.ipynb
│   │   └── ...
│   ├── run_baselines/
│   ├── utils/
│   └── sonata.py
├── examples/
│   ├── cfgs/
│   ├── simulation_t_branch.ipynb
│   └── ...
├── data/
│   ├── t_branch/
│   └── ...
├── results/
│   ├── sonata_pipeline
│   ├── ├── t_branch/
│   └── └── ...
├── README.md
└── requirements.txt
```

## Baseline Performance
We demonstrate that artificial integrations resulting from ambiguous mapping in diagonal data integration are widespread yet surprisingly overlooked, occurring across all mainstream diagonal integration methods.
The following notebooks show the performance cases of baseline methods on various ambiguous datasets:
 - t_branch: [t_branch.ipynb](https://github.com/batmen-lab/SONATA/blob/main/examples/baselines/t_branch.ipynb)
 - scGEM: [scGEM.ipynb](https://github.com/batmen-lab/SONATA/blob/main/examples/baselines/scGEM.ipynb)
 - SNARE: [SNARE.ipynb](https://github.com/batmen-lab/SONATA/blob/main/examples/baselines/SNARE.ipynb)
 - scNMT: [scNMT.ipynb](https://github.com/batmen-lab/SONATA/blob/main/examples/baselines/scNMT.ipynb)

To quantify the ambiguity in these cases, we report label transfer accuracy and average FOSCTTM metrics in our manuscript. All baseline method tests are implemented in the folder *src/run_baselines*. To run a test, use the following commands:
```python
cd src
python run_baselines/run_unioncom.py --dataset t_branch
```
We argue that artificial integrations are more harmful than failed integrations because, while failed integrations can be qualitatively recognized, artificial integrations are difficult to detect and can mislead users into pursuing hypotheses based on erroneous results.

## SONATA Examples
Jupyter notebooks to replicate the SONATA results from the manuscript are available under folder *examples*:  
- Simulation datasets
    - partial ambiguous: [simulation_t_branch.ipynb](https://github.com/batmen-lab/SONATA/blob/main/src/examples/simulation_t_branch.ipynb), [simulation_y_branch.ipynb](https://github.com/batmen-lab/SONATA/blob/main/examples/simulation_y_branch.ipynb), [simulation_x_branch.ipynb](https://github.com/batmen-lab/SONATA/blob/main/src/examples/simulation_x_branch.ipynb)
    - no ambiguous: [simulation_decay_path.ipynb](https://github.com/batmen-lab/SONATA/blob/main/src/examples/simulation_decay_path.ipynb)
- Real biology datasets
    - scGEM: [scGEM.ipynb](https://github.com/batmen-lab/SONATA/blob/main/src/examples/scGEM.ipynb)
    - SNARE: [SNARE.ipynb](https://github.com/batmen-lab/SONATA/blob/main/src/examples/SNARE.ipynb)
    - scNMT: [scNMT.ipynb](https://github.com/batmen-lab/SONATA/blob/main/src/examples/scNMT.ipynb)


## Basic Use
```python
import sonata
sn = sonata.sonata(noise_scale=0.2)
DiagnoseResult = sn.diagnose(data)

# Get the indices of cells identified as ambiguous
ambiguous_idx = DiagnoseResult.ambiguous_idx
# Get the corresponding ambiguous group labels for those cells
ambiguous_labels = DiagnoseResult.ambiguous_labels
```

Input for SONATA: 
 - **parameters**:
    - **noise_scale**: 
        The scale of gaussian noise added to generate variational versions of the manifold. Default: 0.2.
    - **n_neighbor**: 
        Number of neighbors when constructing noise manifold. Default: 10.  
    - **mode**:
        Mode for constructing the graph. Options: "connectivity" or "distance". Default: "connectivity".
    - **metric**:
        Metric to use for distance computation. Default: "correlation".
    - **e**:
        Coefficient of the entropic regularization term in the objective function of OT formulation. Default: 1e-3.
    - **repeat**:
        Number of iterations for alignment. Default: 10.
    - **n_cluster**: 
        Number of cell groups used in hierarchical clustering to achieve a smooth and efficient spline fit. Recommended: n_cluster <= $\sqrt{n\_samples}$. Default: 20.
    - **pval_thres**: 
        P-value threshold for ambiguous group pair detection. Default: 1e-2.         
    - **scalableOT**:
        If True, uses the scalable version of OT. Default: False.
    - **scale_sample_rate**:
        The sample rate for the scalable version of OT. Default: 0.1.
    - **verbose**:
        If True, prints the progress of the algorithm. Default: True.

 - **data**: A NumPy array or matrix where rows correspond to samples and columns correspond to features.

Output for SONATA: 
- An object of SimpleNamespace containing the following attributes:
    - ambiguous_labels: A numpy array of ambiguous group labels for ambiguous samples.
    - ambiguous_idx: A numpy array of indices of ambiguous samples.
    - cannot_links: A list of ambiguous sample pairs.

### Guidence on how to decide parameter "noise_scale"
Please refer to notebook: [noise_scale.ipynb](https://github.com/batmen-lab/SONATA/blob/main/src/examples/simulation_t_branch.ipynb).

### Scalable SONATA
To support large-scale datasets, we offer a more efficient yet equally effective optimal transport algorithm that significantly improves the scalability of SONATA. You can enable this scalable mode by simply setting scalableOT=True:
```python
import sonata
sn = sonata.sonata(noise_scale=0.2, scalableOT=True)
DiagnoseResult = sn.diagnose(data)
``` 


## Major Updates
- **Jun. 11, 2025**: Added Quantized Gromov–Wasserstein to enhance the scalability of SONATA for large datasets.
- **Nov. 2, 2024**: We have released the source code for new version of SONATA.
- **Nov. 1, 2024**: We have added more comprehensive tests for 5 baseline methods, which can be found in the *src/run_baselines* folder. We're also working on the new version of SONATA—coming soon! 