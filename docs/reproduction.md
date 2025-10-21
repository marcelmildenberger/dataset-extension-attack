# Reproduce our Experiments
These are the steps required to reproduce the results we reported in our paper.

**Note:** Several parts of the attack, most importantly hyperparameter optimization, encoding, embedding and
alignment, involve randomness. It is thus extremely unlikely that you are able to
perfectly reproduce our results. However, the overall difference in results should be
negligible.

**Another Note:** Re-Running all experiments will take a considerable amount of time. Depending on your
system specification you might face runtimes in excess of a week.
This is due to the large number of parameter combinations.
___
### System Details

The experiments were run on a virtual machine with the following specification:

- Ubuntu 24.04 LTS
- 20 cores of an AMD EPYC 9254
- NVIDIA GeForce RTX 3090 Ti, 24 GB VRAM
- 176 GB of RAM
- 3 TB HDD space

___
### Obtain Datasets
Make sure that you have all required datasets in the  `./data` directory.
The code expects the following files to be present:

```
fakename_1k.tsv     fakename_2k.tsv     fakename_5k.tsv     fakename_10k.tsv 
fakename_20k.tsv    fakename_50k.tsv    euro_person.tsv     titanic_full.tsv
```

Remember to [prepare](../readme.md) the dataset so it fits the correct file format.
___
### Run the Benchmarks
To reproduce the results we reported in our paper, you may simply run

``python3 experiment_setup.py``

Note: Set the flag in the global config to enable the graph-matching attack or to generate synthetic data splits.
___
### Reproduce Plots
Once the benchmark is complete, you can generate the result plots used in our paper.
Simply generate the plots by running

``python3 extract_nepal_results.py``

and then run the ``analysis.ipnb`` notebook