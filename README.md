# Thesis

**Keywords:** *counting processes*, *functional data analysis*, *survival analysis*, *deep learning*


This repository contains the code of the thesis: **Modelling of the clinical history of patients and introduction via functional covariates of its effects in Cox and deep learning survival models**.

The scope of this thesis was to model the history of patients in the framework of counting processes and to use this information in classical and state-of-the-art survival models.


## Contents

- *preprocess* : prepare the dataset to fit the stochastic processes representing the history of the patients.
- *fit_compensators* :   build the compensators of the counting processes.
- *fpca*: summarise compensators through functional principal component analysis scores
- *survival_process*: build the dataset with fpca scores, fit and compare survival models (Cox, deepHit, DRSA)