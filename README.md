# SONATA
Source code for **SONATA: Disambiguated manifold alignment of single-cell data**  

![SONATA](images/overview.jpg)

## Requirements
Dependencies for **SONATA** are recorded in *requirements.txt*.  

## Data
Datasets are available at [this link](https://drive.google.com/drive/folders/1YWvcBaJ-yj76OjkcMz8cfKchKwuJmkGV?usp=sharing).

## Baseline Performance
All baseline method tests are implemented in the folder *src/run_baselines*. To run a test, use the following commands:
```python
cd src
python run_baselines/run_unioncom.py --dataset t_branch
```

## Examples
Jupyter notebooks to replicate the results from the manuscript are available under folder *examples*:  
- Simulation datasets
    - partial ambiguous: [simulation_t_branch.ipynb](https://github.com/batmen-lab/SONATA/blob/main/examples/simulation_t_branch.ipynb)

**Updated on Nov. 2, 2024**: More examples are coming soon!

## Basic Use
Input for SONATA: *data* in form of numpy arrays/matrices, where the rows correspond to samples and columns correspond to features.
```python
import sonata
sn = sonata.sonata(params)
ambiguous_labels, ambiguous_idx = sn.diagnose(data)
```
For an example, please refer to the cfg file under folder *examples/cfgs*.


## Major Updates
- **Nov. 1, 2024**: We have added more comprehensive tests for 5 baseline methods, which can be found in the *src/run_baselines* folder. We're also working on the new version of SONATA—coming soon! 