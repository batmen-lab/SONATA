# sonata
Source code for **SONATA: Disambiguated manifold alignment of single-cell data**

## Requirements
Dependencies for **SONATA** are recorded in *requirements.txt*. We downloaded Manifold aligner **SCOT** from its [official github](https://github.com/rsinghlab/SCOT).

## Data
Datasets are available at [this link]().

## Examples
Jupyter notebooks to replicate the results from the manuscript are available under folder *examples*:  
- Simulation datasets
    - no ambiguous: [simulation_swiss_roll.ipynb](https://github.com/batmen-lab/SONATA/blob/main/examples/simulation_swiss_roll.ipynb)
    - all ambiguous: [simulation_circle.ipynb](https://github.com/batmen-lab/SONATA/blob/main/examples/simulation_circle.ipynb)
    - partial ambiguous: [simulation_t_branch.ipynb](https://github.com/batmen-lab/SONATA/blob/main/examples/simulation_t_branch.ipynb), [simulation_benz.ipynb](https://github.com/batmen-lab/SONATA/blob/main/examples/simulation_benz.ipynb), [simulation_cross.ipynb](https://github.com/batmen-lab/SONATA/blob/main/examples/simulation_cross.ipynb)
- Real bio datasets: [scGEM.ipynb](https://github.com/batmen-lab/SONATA/blob/main/examples/scGEM.ipynb)

## Basic Use
```python
import sonata
sn = sonata.sonata(args)
alter_mappings = sn.alter_mapping(data)
```
