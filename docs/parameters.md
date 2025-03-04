# Parameters for Graph Matching Attack
___
It is necessary to decide on a number of parameters before running the Graph Matching Attack.
The choice of parameter values can significantly impact attack performance, both in
terms of attack duration and success rate.

We have tried to come up with reasonable defaults that proved useful in our experiments. However,
your specific experiment might work better with other values.

The tables below describe the individual parameters, along with their default values
and references to further information.

The ``run``-Method in ``gma.py`` expects four dictionaries as arguments that specify
the parameters for different stages of the attack.
``gma.py`` (Line 667 onwards) as well as the benchmarking scripts already contain the required
dictionaries, which are filled with default values. You may edit the values freely.
___
## Global Configuration
**Argument Name:** ``GLOBAL_CONFIG``

| Parameter Name | Description                                                                                                                                                                                                             | Default                   | Reference   |
|----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------|-------------|
| Data           | Dataset to run the attack on.                                                                                                                                                                                           | "./data/titanic_full.tsv" |             |
| Overlap        | The share of overlapping records between the attacker's and the victim's data. Must be >0 and <=1.                                                                                                                      | 1                         | Chapter 5   |
| DropFrom       | Which dataset should records be dropped from to achieve the desired overlap? One of "Eve" (Attacker), "Alice" (Victim) or "Both".                                                                                       | "Alice"                   | Chapter 5   |
| DevMode        | If True, similarity graphs (edgelists) and serialized embedding models are stored in the ``./dev/`` directory.                                                                                                          | False                     |             |
| BenchMode      | If True, performance metrics and parameter details are stored as tab-separated values in ``./data/benchmark.tsv``                                                                                                       | False                     |             |
| Verbose        | If True, prints detailed status messages                                                                                                                                                                                | True                      |             |
| MatchingMetric | Similarity metric to be computed on aligned embeddings during bipartite graph matching. Must be available in [scikit learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html). | "cosine"                  | Chapter 4.3 |
| Matching       | Matching algorithm for bipartite graph matching. Must be "MinWeight", "Stable", "Symmetric" or "NearestNeighbor".                                                                                                       | "MinWeight"               | Chapter 4.3 |
| Workers        | Number of cores used in multiprocessing. If >= 1 defines the number of cores. If -1, all available cores are used. If < -1, all available cores except (value-1) ones are used.                                         | -1                        |             |
| StoreAliceEncs | Stores a pickled dictionary containing UIDs as keys and encodings as values in ``./data/encoded/`` for Alice's (victim) dataset.                                                                                        | False                     |             |
| StoreEveEncs   | Stores a pickled dictionary containing UIDs as keys and encodings as values in ``./data/encoded/`` for Eve's (attacker) dataset.                                                                                        | False                     |             |

___
## Encoding Configuration
**Argument Name:** ``ENC_CONFIG``

| Parameter Name | Description                                                                                                        | Default                     | Reference   |
|----------------|--------------------------------------------------------------------------------------------------------------------|-----------------------------|-------------|
| AliceAlgo      | Algorithm used for encoding Alice's data. One of "BloomFilter", "TabMinHash", "TwoStepHash" or None (No Encoding). | "TwoStepHash"               | Appendix A  |
| AliceSecret    | Secret (seed for hash function selection/salt) used when encoding Alice's data. Can be String or Integer.          | "SuperSecretSalt1337"       | Appendix A  |
| AliceN         | Size of N-grams used for encoding Alice's data.                                                                    | 2                           |             |
| AliceMetric    | Similarity metric to be computed during similarity graph generation on Alice's data                                | "dice"                      | Chapter 3.1 |
| EveAlgo        | Algorithm used for encoding Eve's data. One of "BloomFilter", "TabMinHash", "TwoStepHash" or None (No Encoding).   | None                        | Appendix A  |
| EveSecret      | Secret (seed for hash function selection/salt) used when encoding Eve's data. Can be String or Integer.            | "ATotallyDifferentString42" | Appendix A  |
| EveN           | Size of N-grams used for encoding Eve's data.                                                                      | 2                           |             |
| EveMetric      | Similarity metric to be computed during similarity graph generation on Eve's data                                  | "dice"                      | Chapter 3.1 |

**Additional Parameters for Bloom Filter Encoding**

| Parameter Name | Description                                                                                              | Default | Reference                                                                     |
|----------------|----------------------------------------------------------------------------------------------------------|---------|-------------------------------------------------------------------------------|
| AliceBFLength  | Length of the Bloom Filters created for Alice's data. Must be a power of 2.                              | 1024    | Appendix A                                                                    |
| AliceBits      | Number of hash functions to populate the Bloom Filter, i.e. bits per N-Gram                              | 10      | Appendix A                                                                    |
| AliceDiffuse   | If True, adds diffusion layer to Bloom Filter encoding                                                   | False   | [Armknecht et al.](https://petsymposium.org/popets/2023/popets-2023-0054.php) |
| AliceT         | Diffusion parameter t, i.e. number of bit positions in Alice's encodings to be XORed when creating ELDs. | 10      | [Armknecht et al.](https://petsymposium.org/popets/2023/popets-2023-0054.php) |
| AliceEldLength | Length of the ELD, i.e. BF with applied diffusion, for Alice's encodings.                                | 1024    | [Armknecht et al.](https://petsymposium.org/popets/2023/popets-2023-0054.php) |
| EveBFLength    | Length of the Bloom Filters created for Eve's data. Must be a power of 2.                                | 1024    | Appendix A                                                                    |
| EveBits        | Number of hash functions to populate the Bloom Filter, i.e. bits per N-Gram                              | 10      | Appendix A                                                                    |
| EveDiffuse     | If True, adds diffusion layer to Bloom Filter encoding                                                   | False   | [Armknecht et al.](https://petsymposium.org/popets/2023/popets-2023-0054.php) |
| EveT           | Diffusion parameter t, i.e. number of bit positions in Eve's encodings to be XORed when creating ELDs.   | 10      | [Armknecht et al.](https://petsymposium.org/popets/2023/popets-2023-0054.php) |
| EveEldLength   | Length of the ELD, i.e. BF with applied diffusion, for Eve's encodings.                                  | 1024    | [Armknecht et al.](https://petsymposium.org/popets/2023/popets-2023-0054.php) |

**Additional Parameters Tabulation MinHash Encoding**

| Parameter Name | Description                                                                                                                 | Default | Reference                                                                               |
|----------------|-----------------------------------------------------------------------------------------------------------------------------|---------|-----------------------------------------------------------------------------------------|
| AliceNHash     | Number of (tabulation-based) hash functions to use during MinHashing of Alice's data.                                       | 1024    | [Smith](https://www.sciencedirect.com/science/article/pii/S2214212616301405?via%3Dihub) |
| AliceNHashBits | Number of bits to be generated per hash function during MinHashing of Alice's data. Must be 8, 16, 32 or 64.                | 64      | [Smith](https://www.sciencedirect.com/science/article/pii/S2214212616301405?via%3Dihub) |
| AliceNSubKeys  | Number of sub-keys to be generated from the initial 64-bit hash during MinHashing of Alice's data. Must be a divisor of 64. | 8       | [Smith](https://www.sciencedirect.com/science/article/pii/S2214212616301405?via%3Dihub) |
| Alice1BitHash  | If True, applies LSB hashing, i.e. returns only the least significant bit of the MinHash results.                           | True    | [Smith](https://www.sciencedirect.com/science/article/pii/S2214212616301405?via%3Dihub) |
| EveNHash       | Number of (tabulation-based) hash functions to use during MinHashing of Eve's data.                                         | 1024    | [Smith](https://www.sciencedirect.com/science/article/pii/S2214212616301405?via%3Dihub) |
| EveNHashBits   | Number of bits to be generated per hash function during MinHashing of Eve's data. Must be 8, 16, 32 or 64.                  | 64      | [Smith](https://www.sciencedirect.com/science/article/pii/S2214212616301405?via%3Dihub) |
| EveNSubKeys    | Number of sub-keys to be generated from the initial 64-bit hash during MinHashing of Eve's data. Must be a divisor of 64.   | 8       | [Smith](https://www.sciencedirect.com/science/article/pii/S2214212616301405?via%3Dihub) |
| Eve1BitHash    | If True, applies LSB hashing, i.e. returns only the least significant bit of the MinHash results.                           | True    | [Smith](https://www.sciencedirect.com/science/article/pii/S2214212616301405?via%3Dihub) |


**Additional Parameters for Two-Step-Hash Encoding**

| Parameter Name | Description                                                                                                                                             | Default | Reference                                                                          |
|----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|---------|------------------------------------------------------------------------------------|
| AliceNHashFunc | Number of hash function, i.e. number of bits per N-gram, to use when populating the intermediate BFs on Alice's data                                    | 10      | [Ranbaduge et al.](https://link.springer.com/chapter/10.1007/978-3-030-47436-2_11) |
| AliceNHashCol  | Number of columns (length) of intermediate BFs computed on Alice's data                                                                                 | 1000    | [Ranbaduge et al.](https://link.springer.com/chapter/10.1007/978-3-030-47436-2_11) |
| AliceRandMode  | Algorithm to be used for column-wise hashing of Alice's intermediate encodings. Either "PNG" for a pseudo-random number generator or "SHA" for SHA-256. | "PNG"   | [Ranbaduge et al.](https://link.springer.com/chapter/10.1007/978-3-030-47436-2_11) |
| EveNHashFunc   | Number of hash function, i.e. number of bits per N-gram, to use when populating the intermediate BFs on Eve's data                                      | 10      | [Ranbaduge et al.](https://link.springer.com/chapter/10.1007/978-3-030-47436-2_11) |
| EveNHashCol    | Number of columns (length) of intermediate BFs computed on Eve's data                                                                                   | 1000    | [Ranbaduge et al.](https://link.springer.com/chapter/10.1007/978-3-030-47436-2_11) |
| EveRandMode    | Algorithm to be used for column-wise hashing of Eve's intermediate encodings. Either "PNG" for a pseudo-random number generator or "SHA" for SHA-256.   | "PNG"   | [Ranbaduge et al.](https://link.springer.com/chapter/10.1007/978-3-030-47436-2_11) |

___
## Embedding Configuration
**Argument Name:** ``EMB_CONFIG``

| Parameter Name  | Description                                                                                                      | Default    | Reference                                                                                                                       |
|-----------------|------------------------------------------------------------------------------------------------------------------|------------|---------------------------------------------------------------------------------------------------------------------------------|
| AliceAlgo       | Algorithm to use for embedding Alice's data. Must be "Node2Vec" or "NetMF".                                      | "Node2Vec" |                                                                                                                                 |
| AliceQuantile   | Drop edges with *AliceQuantile* lowest edgeweights in Alice's similarity graph. Must be >=0  (keep all) and < 1. | 0.9        | Chapter 5                                                                                                                       |
| AliceDiscretize | If True, sets all edgeweights in Alice's similarity graph to 1.                                                  | False      |                                                                                                                                 |
| AliceDim        | Dimensionality of Alice's embeddings.                                                                            | 128        | [Grover and Leskovec (Node2Vec)](https://arxiv.org/abs/1607.00653) <br/> [Qiu et al. (NetMF)](https://arxiv.org/abs/1710.02971) |
| AliceContext    | Context size to use when calculating Alice's embeddings                                                          | 10         | [Grover and Leskovec (Node2Vec)](https://arxiv.org/abs/1607.00653) <br/> [Qiu et al. (NetMF)](https://arxiv.org/abs/1710.02971) |
| AliceNegative   | Number of negative samples during training of Alice's embeddings (NetMF only).                                   | 1          | [Grover and Leskovec (Node2Vec)](https://arxiv.org/abs/1607.00653) <br/> [Qiu et al. (NetMF)](https://arxiv.org/abs/1710.02971) |
| Alice Normalize | If True, normalize Alice's embeddings (NetMF only).                                                              | False      | [Qiu et al. (NetMF)](https://arxiv.org/abs/1710.02971)                                                                          |
| EveAlgo         | Algorithm to use for embedding Eve's data. Must be "Node2Vec" or "NetMF".                                        | "Node2Vec" |                                                                                                                                 |
| EveQuantile     | Drop edges with *EveQuantile* lowest edgeweights in Eve's similarity graph. Must be >=0  (keep all) and < 1.     | 0.9        | Chapter 5                                                                                                                       |
| EveDiscretize   | If True, sets all edgeweights in Eve's similarity graph to 1.                                                    | False      |                                                                                                                                 |
| EveDim          | Dimensionality of Eve's embeddings.                                                                              | 128        | [Grover and Leskovec (Node2Vec)](https://arxiv.org/abs/1607.00653) <br/> [Qiu et al. (NetMF)](https://arxiv.org/abs/1710.02971) |
| EveContext      | Context size to use when calculating Eve's embeddings                                                            | 10         | [Grover and Leskovec (Node2Vec)](https://arxiv.org/abs/1607.00653) <br/> [Qiu et al. (NetMF)](https://arxiv.org/abs/1710.02971) |
| EveNegative     | Number of negative samples during training of Eve's embeddings (NetMF only).                                     | 1          | [Grover and Leskovec (Node2Vec)](https://arxiv.org/abs/1607.00653) <br/> [Qiu et al. (NetMF)](https://arxiv.org/abs/1710.02971) |
| Eve Normalize   | If True, normalize Eve's embeddings (NetMF only).                                                                | False      | [Qiu et al.](https://arxiv.org/abs/1710.02971)                                                                                  |

**Additional Parameters for Node2Vec Embedding**

| Parameter Name | Description                                                                                      | Default | Reference                                               |
|----------------|--------------------------------------------------------------------------------------------------|---------|---------------------------------------------------------|
| AliceWalkLen   | Length of the random walks performed on Alice's similarity graph                                 | 100     | [Grover and Leskovec](https://arxiv.org/abs/1607.00653) |
| AliceNWalks    | Number of random walks performed per node in Alice's similarity graph                            | 20      | [Grover and Leskovec](https://arxiv.org/abs/1607.00653) |
| AliceP         | "Return Parameter" for governing random walks performed on Alice's similarity graph              | 250     | [Grover and Leskovec](https://arxiv.org/abs/1607.00653) |
| AliceQ         | "In-Out-Parameter" for governing random walks performed on Alice's similarity graph              | 300     | [Grover and Leskovec](https://arxiv.org/abs/1607.00653) |
| AliceEpochs    | Number of epochs Alice's embeddings are trained for                                              | 5       | [Grover and Leskovec](https://arxiv.org/abs/1607.00653) |
| AliceSeed      | Seed to initialize the (pseudo) random number generators used for calculating Alice's embeddings | 42      |                                                         |
| EveWalkLen     | Length of the random walks performed on Eve's similarity graph                                   | 100     | [Grover and Leskovec](https://arxiv.org/abs/1607.00653) |
| EveNWalks      | Number of random walks performed per node in Eve's similarity graph                              | 20      | [Grover and Leskovec](https://arxiv.org/abs/1607.00653) |
| EveP           | "Return Parameter" for governing random walks performed on Eve's similarity graph                | 250     | [Grover and Leskovec](https://arxiv.org/abs/1607.00653) |
| EveQ           | "In-Out-Parameter" for governing random walks performed on Eve's similarity graph                | 300     | [Grover and Leskovec](https://arxiv.org/abs/1607.00653) |
| EveEpochs      | Number of epochs Eve's embeddings are trained for                                                | 5       | [Grover and Leskovec](https://arxiv.org/abs/1607.00653) |
| EveSeed        | Seed to initialize the (pseudo) random number generators used for calculating Eve's embeddings   | 42      |                                                         |

___
## Alignment Configuration
**Argument Name:** ``ALIGN_CONFIG``

| Parameter Name | Description                                                                                                                                                                                                                                                                                                       | Default             | Reference                                        |
|----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------|--------------------------------------------------|
| RegWS          | Regularization Parameter for Sinkhorn solver                                                                                                                                                                                                                                                                      | max(0.1, Overlap/2) | [Grave et al.](https://arxiv.org/abs/1805.11222) |
| RegInit        | Regularization Parameter for convex initialization                                                                                                                                                                                                                                                                | 1                   | [Grave et al.](https://arxiv.org/abs/1805.11222) |
| Batchsize      | Batchsize for Wasserstein Procrustes. If <=1 interpreted as share of overall data, if >1 interpreted as number of examples                                                                                                                                                                                        | 1                   | [Grave et al.](https://arxiv.org/abs/1805.11222) |
| LR             | Learning rate for optimization                                                                                                                                                                                                                                                                                    | 200                 | [Grave et al.](https://arxiv.org/abs/1805.11222) |
| LRDecay        | Learning rate decay. LR is multiplied by this factor after every epoch.                                                                                                                                                                                                                                           | 1                   |                                                  |
| NIterInit      | Number of iterations during convex initialization                                                                                                                                                                                                                                                                 | 5                   | [Grave et al.](https://arxiv.org/abs/1805.11222) |
| NIterWS        | Number of iterations per epoch of optimization                                                                                                                                                                                                                                                                    | 100                 | [Grave et al.](https://arxiv.org/abs/1805.11222) |
| NEpochWS       | Number of optimization epochs                                                                                                                                                                                                                                                                                     | 100                 | [Grave et al.](https://arxiv.org/abs/1805.11222) |
| Sqrt           | If True, compute alignment on the square root of embeddings                                                                                                                                                                                                                                                       | True                | [Grave et al.](https://arxiv.org/abs/1805.11222) |
| EarlyStopping  | Terminate optimization if loss hasn't improved for this many epochs                                                                                                                                                                                                                                               | 10                  | Chapter 5                                        |
| Selection      | Algorithm to select the records used for alignment. If "GroundTruth", then matrices are constructed using ground truth, i.e. embeddings representing the same records are stored in the same rows. If "Random", random subsampling is performed. If None, a random permutation of all available records are used. | None                |                                                  | 
| MaxLoad        | Specifies the number of elements to use for alignment if *Selection* is not None. Must be smaller than the number of records in the smaller dataset.                                                                                                                                                              | None                |                                                  |
| Wasserstein    | If True, uses unsupervised Wasserstein Procrustes for alignment. If False, uses supervised closed-form Procrustes.                                                                                                                                                                                                | True                |                                                  |

___
## Blocking Configuration
**Argument Name:** ``BLOCKING_CONFIG``

**Blocking is only performed in the re-implementation of Vidanage et al.'s attack.
Thus, the `BLOCKING_CONFIG` parameter is not present otherwise**


| Parameter Name  | Description                                | Default | Reference                                            |
|-----------------|--------------------------------------------|---------|------------------------------------------------------|
| Disable         | If true, then the blocking step is skipped | False   |                                                      |
| PlainSampleSize | The length of the min-hash bands.          | 4       | [Broder](https://doi.org/10.1109/SEQUEN.1997.666900) |
| PlainNumSamples | The number of LSH bands.                   | 50      | [Broder](https://doi.org/10.1109/SEQUEN.1997.666900) |
| AliceRandomSeed | Random Seed used for blocking Alice's data | 42      |                                                      |
| EveRandomSeed   | Random Seed used for blocking Eve's data   | 17      |                                                      |
