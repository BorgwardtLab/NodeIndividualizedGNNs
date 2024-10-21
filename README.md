# Node Individualized GNNs

This repository implements the models and the experiments in the _NeurIPS 2024_ paper "On the Expressivity and Sample Complexity of Node-Individualized Graph Neural Networks". The paper is available [here](https://nips.cc/virtual/2024/poster/96388).

### Description

Graph neural networks (GNNs) employing message passing for graph classification are inherently limited by the expressive power of the Weisfeiler-Lehman (WL) test for graph isomorphism. Node individualization schemes, which assign unique identifiers to nodes (e.g., by adding random noise to features), are a common approach for achieving universal expressiveness. Here, we address the expressivity and the sample complexity of GNNs endowed with node individualization schemes.

In particular, we provide a novel individualization scheme, **TinhoferW**, that ensures universal expressivity while maintaining a low sample complexity. You can find the PyG transform implementing it in ```models/tinhofer.py```.

Moreover, we provide a novel model architecture **EGONN** for the subgraph identification (i.e., subgraph isomorphism) task. You can find the PyG model implementing it in ```models/egonn.py``` and examples of its usage in ```patterns/```.

### Citing our work

> Paolo Pellizzoni, Till Schulz, Dexiong Chen and Karsten Borgwardt. _On the Expressivity and Sample Complexity of Node-Individualized Graph Neural Networks_, in NeurIPS, 2024.

### Usage

Run ```source s``` to load modules 

- expressivity/ contains the experiments of Section 5.1 
- vc-dim/ contains the experiments of Section 5.2
- patterns/ contains the experiments of Section 5.3
- covering/ contains the experiments of Section 5.4
