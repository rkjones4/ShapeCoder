#  ShapeCoder: Discovering Abstractions for Visual Programs from Unstructured Primitives 

By [R. Kenny Jones](https://rkjones4.github.io/), [Paul Guerrero](https://paulguerrero.net/), [Niloy J. Mitra](http://www0.cs.ucl.ac.uk/staff/n.mitra/), and [Daniel Ritchie](https://dritchie.github.io/)

![image](https://rkjones4.github.io/img/shapecoder/sc_teaser.png)
 
## About the paper

[Project Page](https://rkjones4.github.io/shapecoder.html)

[Paper](https://rkjones4.github.io/pdf/shapeCoder.pdf)

[Supplemental](https://rkjones4.github.io/pdf/shapeCoder_supplemental.pdf)

Presented at [Siggraph 2023](https://s2023.siggraph.org/).


## Bibtex
```
@article{jones2023ShapeCoder
  author = {Jones, R. Kenny and Guerrero, Paul and Mitra, Niloy J. and Ritchie, Daniel},
  title = {ShapeCoder: Discovering Abstractions for Visual Programs from Unstructured Primitives},
  year = {2023},
  issue_date = {August 2023},
  address = {New York, NY, USA},
  volume = {42},
  number = {4},
  journal={ACM Transactions on Graphics (TOG), Siggraph 2023},
  articleno = {49},
}
```

# ShapeCoder Results

We provide discovered libraries and programs for the chair, table, and storage domains as described in our paper, which can be download from this [link](https://drive.google.com/file/d/1CmuNsR06fRt0l-zTjnQ2m_HtUbIxa0n4/view?usp=sharing).

# Running ShapeCoder

## Setup intstructions

This code was tested on Ubuntu 20.04, an NVIDIA 3090 GPU, python 3.0, pytorch 1.9, and cuda 11.1. You must have [rust](https://www.rust-lang.org/tools/install) installed to enable the e-graph refactor operation.

You can take the following steps to set up the conda environment:

```
cd code
conda env create -f env.yml
conda activate shapecoder
cd egg
maturin develop
```

To check this was succesful, from code, try running:

> python3 extract.py

If you get an error message, then something has gone wrong in building the refactor operation.

## Starting a new run

code/shapecode.py is the main entrypoint for the shapecoder method. Please run all commands from the code/ directory.

To start training a new run, you can use a command like:

> python3 shapecoder.py -en {exp_name} -dn {domain} -ds {num_shapes} -dp {path_to_shapes}

*path_to_shapes* is relative to s3d_lang/prog_data. We include three options: PN_chair, PN_table, PN_storage, that each definie primitive-based manufactured shapes from PartNet annotations.

*num_shapes* is the number of shapes the method will run over. We recommend setting this to 400, or between 100 and 1000 more generally. Some parameters in shapecoder.py might need to be updated, if num_shapes changes beyond this range.

*domain* is the domain to run shapecoder over. The provided options are s2d or s3d, for either 2d or 3d shapes. For s2d, shapecoder will sample shapes from a grammar, described in s2d_lang/simp_chair_grammar.py, so no -dp argument needs to be provided.

*exp_name* will be name of experiment, all outputs will be saved in abs_output/{exp_name}.

For instance, to run shapecoder over chairs, you can run:

> python3 shapecoder.py -en sc_pn_chairs -dn s3d -ds 400 -dp PN_chair_1k

For each round, the library of discovered abstraction functions will be written to abs_output/{exp_name}/round_{round}/end_lib.txt, while the inferred program dataset will be written to abs_output/{exp_name}/round_{round}/end_data.txt

See the argument lists in shapecoder.py for additional method hyperparameters that can be adjusted. 

## Visualizing results

To visualize how the discovered abstractions from a run are used, you can use the following command:

> python3 vis_res.py {path_to_results} {num_vis_per_abs} {out_dir}

This will load the library and dataset from *path_to_results* (the saved results of a new run), which should have both a lib.pkl and data.pkl file. It will save *num_vis_per_abs* example uses of each abstraction into *out_dir*. Example outputs of this script are provided in the "ShapeCoder Results" section link.

## Directory structure

**code** has logic for shapecoder method. shapecoder.py is the main entrypoint. Please see code/README for further information on repository structure

**s3d_lang** defines domain-specific language for 3D shapes, along with visualizer.

**s2d_lang** defines domain-specific language for 2D shapes, and a simple grammar for sampling chair-like structures.
