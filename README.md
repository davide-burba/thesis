# Thesis

**Keywords:** *counting processes*, *functional data analysis*, *survival analysis*, *deep learning*


This repository contains the code of the thesis: **Performing Survival Analysis via Functional Cox-type Regression and a Machine Learning approach: an application to Heart Failure patients**.

The text is available [here](https://www.politesi.polimi.it/retrieve/393101/2019_07_Burba.pdf).

The goal of this thesis was to model the history of patients in the framework of counting processes, to use this information in classical and state-of-the-art survival models and to compare them.


## Contents

- *preprocess* : prepare the dataset to fit the stochastic processes representing the history of the patients.
- *fit_compensators* :   build the compensators of the counting processes.
- *fpca*: summarise compensators through functional principal component analysis scores
- *survival_process*: build the dataset with fpca scores, fit and compare survival models (Cox, deepHit, DRSA)
