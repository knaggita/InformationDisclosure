# Revealing Positive and Negative Role Models to Help People Make Good Decisions

Code to accompany the paper "Revealing Positive and Negative Role Models to Help People Make Good Decisions"

## Overview

TL;DR: We present various strategies for revealing positive and negative targets to help people make good decisions.

## Contents and layout

This repository includes the original datasets, generated bipartite graphs, and implementation code for different strategies clearly. 
```
.
├── Fairness/
├── coverageModel/
├── data/
├── graphs/
├── graphsGen/
├── interventionModel/
├── learningSetting/
├── scripts/
├── standardModel/
└── README.md
```
Below are the specifics of the repository files in correspondence to the paper sections.
* `Fairness/`: Python notebooks for fairness results (Section 5, Appendix G.3).
* `coverageModel/`: Python notebooks for computing the radius coverage model results (Section 5, Appendix G.5).
* `data/`: The initial binary classification datasets (csv files) used in the experiments: Adult, Student-math, Student-portuguese, and Productivity (Appendix G.1).
* `graphs/`: The generated graphs with the kNN and threshold methods (Appendix G.1).
* `graphsGen/`: Python notebooks for generating the graphs (Appendix G.1).
* `interventionModel/`: Python notebooks for computing the targeted intervention model results (Section 5, Appendix G.4).
* `learningSetting/`: Python notebooks for computing the learning setting results (Section 5, Appendix G.6).
* `scripts/`: Scripts for analyzing the results, the algorithms and generating the graphs
* `standardModel/`: Python notebooks for computing the standard model results (Section 5, Appendix G.2).
* `requirements.txt`: A list of some packages or dependencies needed to run the project.

## Getting started

### Prerequisites

* Python 3.11.5
* Recommended: Use anaconda or miniconda from [here.](https://www.anaconda.com/docs/getting-started/anaconda/install)

### Basic usage

1. Clone the repository:

   ```bash
   git clone https://github.com/knaggita/InformationDisclosure.git
   cd InformationDisclosure
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

   
## Citation

If you find this work useful in your research, please cite our paper:

```bibtex
@article{avrim2026revealpos,
	title        = {{Revealing Positive and Negative Role Models to Help People Make Good Decisions}},
	author       = {Avrim Blum and Keziah Naggita and Matthew Walter and Jingyan Wang},
	year         = 2026,
	url          = {https://},
	note         = {}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


## Contact

For questions or collaborations, please contact us [here](mailto:knaggita@ttic.edu).



