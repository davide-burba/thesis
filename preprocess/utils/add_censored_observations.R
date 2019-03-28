
require(data.table)
require(lubridate)


#' Add censoring due to days of follow-up (status=1 means uncensored)
#' NB: mark is set = -1 (not meaningful)
add_censored_observations = function(events_df, verbose = FALSE, follow_up_time = 365.5){
  if(verbose){print('Adding censored observations (due to follow up)')}
  # set status uncensored observations
  events_df[,'status' := 1]
  # select constant features
  const_features = colnames(events_df)[!colnames(events_df) %in% c('id','time_event','mark')]
  # set censorship time (default = 365.5 days)
  censorship_time = make_difftime('days' = follow_up_time)
  # compute censored observation for each patient (default censorship at 365.5 days)
  censored_observations = data.table()
  patient_ids = unique(events_df$id)
  pb <- txtProgressBar(min = 0, max = length(patient_ids), style = 3)
  for(i in 1:length(patient_ids)){
    setTxtProgressBar(pb, i)
    patient_id = patient_ids[i]
    tmp = events_df[events_df$id == patient_id,]
    # copy fixed features from the first observation of the patient
    patient_id_censorship = tmp[1,]
    # modify time, mark, status
    patient_id_censorship[,time_event := censorship_time] 
    patient_id_censorship[,mark := -1] # Note: mark is not meaningful
    patient_id_censorship[,status := 0] 
    # store it
    censored_observations = rbind(censored_observations,patient_id_censorship)
  }
  cat("\n")
  # put censored and uncensored observations togheter
  events_df = rbind(events_df,censored_observations)
  # sort by id,time_event
  setorderv(events_df, c('id','time_event')) 
  
  return(events_df)
}
