This is the code accompanying the paper:

  [On (assessing) the fairness of risk score models](https://arxiv.org/pdf/2302.08851)  
  Petersen, Ganz, Holm, Feragen  
  ACM Conference on Fairness, Accountability, and Transparency (FAccT ’23)
	
**Main scripts**:
- `model_fitting.py` fits a logistic regression and an xgboost classification model to the dataset.
- `analyze_models.py` performs the fairness analysis described in the paper and produces figure 3 in the paper.
- `auc_ranking_paper_examples.py` implements the example shown in and produces figure 1 in the paper.
- `calibration_bias_analysis.py` implements the calibration bias analyses described in the paper and produces figure 2.

The first two scripts require the Catalan juvenile recidivism dataset provided by the Centre for Legal Studies and Specialised Training (CEJFE) within the Department of Justice of the Government of Catalonia, first analyized by [Tolan et al. (2019)](https://doi.org/10.1145/3322640.3326705).
The dataset can be downloaded [here](https://cejfe.gencat.cat/en/recerca/opendata/jjuvenil/reincidencia-justicia-menors/index.html); use [the preprocessing](https://github.com/elisabethzinck/Fairness-oriented-interpretability-of-predictive-algorithms/blob/main/src/data/cleaning-catalan-juvenile-recidivism-data.py) provided and described by [Fuglsang-Damgaard and Zink (2022)](http://fairmed.compute.dtu.dk/files/theses/Fairness-oriented%20interpretability%20of%20predictive%20algorithms%20[Fuglsang-Damgaard,%20Zinck]%20(2022).pdf).

To set up the required packages, do `conda env create -f environment.yml` (if using Anaconda) or `pip install -r requirements.txt` (if using pip).

**Implemented general functionality** (besides the above main scripts):
- Implementation of the debiased calibration error metric we describe in the paper, which is based on the method by [Kumar et al. (2019)](https://proceedings.neurips.cc/paper/2019/file/f8c0c968632845cd133308b1a494967f-Paper.pdf): `get_unbiased_calibration_rmse` in `calibration.py`.
- Unified implementations of the standard (sample size-biased, like we discuss in the paper) expected calibration error (ECE) and adaptive calibration error (ACE) metrics, both with fixed and automatically determined bin counts: `ece` in `metrics.py`.
- LOESS-based calibration diagrams (with bootstrap-based uncertainty quantification), as proposed by [Austin and Steyerberg (2013)](https://doi.org/10.1002/sim.5941) and as shown in Figs 3 and 4 in our paper: `rel_diag` in `analyze_models.py`.

---

[Eike Petersen](e-pet.github.io), [Technical University of Denmark](dtu.dk), [DTU Compute](compute.dtu.dk), [Section Visual Computing](https://www.compute.dtu.dk/english/research/research-sections/visual-computing), 2023.  
Created as part of the project [Bias and fairness in medicine](fairmed.compute.dtu.dk), funded by the Independent Research Fund Denmark (DFF).
