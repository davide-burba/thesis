# Thesis

This repository contains the code of the thesis: **Modelling of survival time in HeartFailure patients via functional covariates in Cox and deep learning models**

**Keywords:** *counting processes*,*functional data analysis*,*survival analysis*,*deep learning*

The scope of this thesis was to model the history of patients in the framework of counting processes and to use this information in survival models.


## Contents

- *preprocess* : prepare the dataset to fit the stochastic processes representing the history of the patients.
- *fit_compensator* :   build the compensators of the counting processes.
- *fpca*: summarise compensators through functional principal component analysis scores
- *survival analysis*: build the dataset with fpca scores, fit and compare survival models (Cox, deepHit, DRSA)