# Thesis

**Keywords:** *counting processes*, *functional data analysis*, *survival analysis*, *deep learning*


This repository contains the code of the thesis: **Information extraction from counting processes: an application to survival time modelling**.

The scope of this thesis was to model the history of patients in the framework of counting processes and to use this information in classical and state-of-the-art survival models.


## Contents

- *preprocess* : prepare the dataset to fit the stochastic processes representing the history of the patients.
- *fit_compensators* :   build the compensators of the counting processes.
- *fpca*: summarise compensators through functional principal component analysis scores
- *survival_process*: build the dataset with fpca scores, fit and compare survival models (Cox, deepHit, DRSA)