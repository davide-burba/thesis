
library(splines)


#' Evaluate cumulative Hazard in a grid of points; returns a dataframe in long format
compute_cumulative_hazard = function(model,sel_df,smoothed_baseline,times,verbose = FALSE){
  # Evaluate Lambda on a grid (NB: Lambda0_fun(t) == Lambda0s_value[t+1])
  Lambda0s_value = .Lambda0_fun(times,smoothed_baseline)
  # Compute constant at times coefficients
  patient_coefficients = .compute_coefficients_ck(sel_df,model, verbose)
  # Compute daily deltas of cumulative Hazard and sum it up
  cumulative_hazard = .compute_cumulative_hazard(patient_coefficients,Lambda0s_value, verbose)
  return(cumulative_hazard)
}


#' compute coefficient ck at each interval for each patient (maybe it should be vectorized)
.compute_coefficients_ck = function(sel_df,model, verbose = FALSE){
  if(verbose){print('Computing coefficients ck')}
  # take coefficients beta
  name_coefficients = names(model$coefficients)
  coefficients = model$coefficients
  
  patient_coefficients = NULL
  patient_ids = unique(sel_df$id)
  pb <- txtProgressBar(min = 0, max = length(patient_ids), style = 3)
  for(i in 1:length(patient_ids)){
    setTxtProgressBar(pb, i)
    patient_id = patient_ids[i]
    patient_df = sel_df[id == patient_id,]
    ck = c()
    for(k in patient_df$Nm){
      beta_times_xik = sum(patient_df[, ..name_coefficients][k+1]*coefficients)
      ck = c(ck,exp(beta_times_xik))
    }
    patient_coefficients = rbind(patient_coefficients,data.frame(id = patient_id,k = patient_df$Nm, tk = patient_df$start,ck))
  }
  cat("\n")
  return(data.table(patient_coefficients))
}


#' compute daily deltas of cumulative Hazard and it sums it up; returns a dataframe in long format  (maybe it should be vectorized)
.compute_cumulative_hazard = function(patient_coefficients,Lambda0s_value, verbose = FALSE){
  if(verbose){print('Computing cumulative Hazard on the grid')}
  cumulative_hazard = NULL
  patient_ids = unique(patient_coefficients$id)
  pb <- txtProgressBar(min = 0, max = length(patient_ids), style = 3)
  for(i in 1:length(patient_ids)){
    setTxtProgressBar(pb, i)
    patient_id = patient_ids[i]
    
    # treat apart interval [-0.5,0)
    ck = patient_coefficients[id == patient_id & tk < 0]$ck
    deltas = c(ck*Lambda0s_value[1])
    # compute deltas for each day
    for(t in 1:max(times)){
      tmp = patient_coefficients[id == patient_id & tk < t]$ck
      ck = tmp[length(tmp)]
      deltas = c(deltas,ck*(Lambda0s_value[t+1] - Lambda0s_value[t]))# NB: this corresponds to Lambda0s(t) - Lambda0s(t-1)
    }
    cumulative_hazard = rbind(cumulative_hazard,data.frame(id = patient_id,time = times,cumhaz = cumsum(deltas)))
  }
  cat("\n")
  return(cumulative_hazard)
}


# Lambda_0
#      t  = evaluation points
.Lambda0_fun <- function(t,smoothed_baseline){
  return(.basis(t, smoothed_baseline$knots) %*% smoothed_baseline$coef)
}


########################################################################################
### basis: returns the evaluations of a b-spline basis on a vector of points x
###
### Inputs:
###     x:  vector of abscissa values for evaluation
### knots:  knots of the basis
###   deg:  polynomial degree of the basis
###
### Outputs:
###         a matrix with the evaluations in values of x (rows) of the B-splines basis
###         functions (columns)
########################################################################################
.basis <- function(x, knots, deg=2){
  return(bs(x,knots=knots[2:(length(knots)-1)],degree=deg,Boundary.knots=c(knots[1],knots[length(knots)]),intercept=TRUE))
}
