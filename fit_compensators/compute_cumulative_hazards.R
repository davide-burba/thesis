
library(splines)


#' Evaluate cumulative Hazard in a grid of points; returns a dataframe in long format
compute_cumulative_hazard = function(model,
                                     sel_df,
                                     smoothed_baseline,
                                     times,
                                     verbose = FALSE){
  # Evaluate Lambda on a grid (NB: Lambda0_fun(t) == Lambda0s_value[t+1])
  Lambda0s_value = .Lambda0_fun(times,smoothed_baseline)
  # Compute constant at times coefficients
  patient_coefficients = .compute_coefficients_ck(sel_df,model, verbose)
  # Compute daily deltas of cumulative Hazard and sum it up
  cumulative_hazard = .compute_cumulative_hazard(patient_coefficients,Lambda0s_value, verbose)
  return(cumulative_hazard)
}


#' Compute coefficient ck = exp{beta*x_i(t_k)} at each interval for each patient 
.compute_coefficients_ck = function(sel_df,model,verbose = FALSE){
  if(verbose){print('Computing coefficients ck')}
  # take coefficients beta
  name_coefficients = names(model$coefficients)
  coefficients = model$coefficients
  
  patient_coefficients = NULL
  patient_ids = unique(sel_df$id)
  pb <- txtProgressBar(min = 0, max = length(patient_ids), style = 3)
  # compute coefficients for each individual 
  for(i in 1:length(patient_ids)){
    setTxtProgressBar(pb, i)
    patient_id = patient_ids[i]
    patient_df = sel_df[id == patient_id,]
    ck = c()
    # compute coefficients for each time interval (start,stop]
    for(k in patient_df$Nm){
      beta_times_xik = sum(patient_df[, ..name_coefficients][k+1]*coefficients)
      ck = c(ck,exp(beta_times_xik))
    }
    patient_coefficients = rbind(patient_coefficients,data.frame(id = patient_id,
                                                                 k = patient_df$Nm,
                                                                 tk = patient_df$start,
                                                                 ck))
  }
  cat("\n")
  return(data.table(patient_coefficients))
}


#' Compute daily deltas of cumulative Hazard and it sums it up; returns a dataframe in long format 
.compute_cumulative_hazard = function(patient_coefficients,Lambda0s_value,verbose = FALSE){
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
      # deltas: Lambda0s(t) - Lambda0s(t-1)
      deltas = c(deltas,ck*(Lambda0s_value[t+1] - Lambda0s_value[t]))
    }
    cumulative_hazard = rbind(cumulative_hazard,data.frame(id = patient_id,
                                                           time = times,
                                                           cumhaz = cumsum(deltas)))
  }
  cat("\n")
  return(cumulative_hazard)
}


# Evaluate smoothed baseline cumulative hazard
.Lambda0_fun <- function(t,smoothed_baseline){
  return(.basis(t, smoothed_baseline$knots) %*% smoothed_baseline$coef)
}


# Evaluate a b-spline basis on a vector of points x
.basis <- function(x, knots, deg=2){
  return(bs(x,
            knots=knots[2:(length(knots)-1)],
            degree=deg,
            Boundary.knots=c(knots[1],knots[length(knots)]),
            intercept=TRUE))
}
