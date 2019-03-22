
library(data.table)


#' Add censoring due to days of follow-up (status=1 means uncensored)
#' NB: qt_prest_sum is set = -1 (not meaningful)
add_censored_observations = function(sel_df, verbose = FALSE, follow_up_time = 365.5){
  if(verbose){print('Adding censored observations (due to follow up)')}
  # set status uncensored observations
  sel_df[,'status' := 1]
  # compute censored observation for each patient (default censorship at 365.5 days)
  censored_observations = data.table()
  patient_ids = unique(sel_df$id)
  pb <- txtProgressBar(min = 0, max = length(patient_ids), style = 3)
  for(i in 1:length(patient_ids)){
    setTxtProgressBar(pb, i)
    patient_id = patient_ids[i]
    tmp = sel_df[sel_df$id == patient_id,]
    patient_id_censorship = data.table(
      'id' = patient_id,
      'time_event'=make_difftime('days' = follow_up_time),# censorship time (default = 365.5 days)
      'age_in' = tmp[1,][['age_in']],
      'sex' = tmp[1,][['sex']],
      'qt_prest_sum' = -1, # not meaningful
      'status' = 0
    )
    censored_observations = rbind(censored_observations,patient_id_censorship)
  }
  # put censored and uncensored observations togheter
  sel_df = rbind(sel_df,censored_observations)
  # sort by id,time_event
  setorderv(sel_df, c('id','time_event')) 
  
  return(sel_df)
}
