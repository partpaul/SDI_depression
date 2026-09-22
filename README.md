# SDI in patients with major depressive disorder
Project shortname: SDI_depression

- Maintainers: Enrico Glerean, Paula Partanen.
- Other contributors: Teemu Ruokolainen

## How to set-up
To use this code, git clone it in your local folder. The python code has a few dependecies and they are listed in the file `environment.yaml`.
You will also need to have a working installation of FSL before running certain scripts.

To create a python environment for the project

```
cd code
mamba env create -f environment.yaml -p ./env
# wait...
source activate ./env
```

## Steps on how to reproduce the results
Each file is named after a step for reproducibility purposes. Please refer to the comments in each file for further details.
