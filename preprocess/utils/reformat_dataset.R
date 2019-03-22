
#' reformat_dataset: add the following new columns
#' start,stop: interval times as requested by survival::coxph
#' Nm: number of events before the considered one for each patient N_i(t-)
#' sum_past_qt_prest: sum of the past qt_prest_sum for each patient
reformat_dataset = function(sel_df,verbose = FALSE){
  if(verbose){print('Adding new columns: start,stop, Nm (N_i(t-)), sum_past_qt_prest')}
  start = stop = Nm = sum_past_qt_prest = c()
  patient_ids = unique(sel_df$id)
  pb <- txtProgressBar(min = 0, max = length(patient_ids), style = 3)
  for(i in 1:length(patient_ids)){
    setTxtProgressBar(pb, i)
    patient_id = patient_ids[i]
    tmp = sel_df[sel_df$id == patient_id,]
    
    # times (start,stop]
    events_time = tmp[,'time_event']
    events_time = as.numeric(unlist(events_time))
    stop = c(stop,events_time)
    start = c(start,c(-0.5,events_time[-length(events_time)]))
    # N_i(t-)
    Nm = c(Nm,0:(length(events_time)-1))
    # sum_past_qt_prest
    qt_prest_patient = tmp[,'qt_prest_sum']
    sum_qt_prest_patient = cumsum(as.numeric(unlist(qt_prest_patient)))
    sum_past_qt_prest_patient = c(0,sum_qt_prest_patient[-length(sum_qt_prest_patient)])
    sum_past_qt_prest = c(sum_past_qt_prest, sum_past_qt_prest_patient)
  }
  sel_df[,'start' := start]
  sel_df[,'stop' := stop]
  sel_df[,'Nm' := Nm]
  sel_df[,'sum_past_qt_prest' := sum_past_qt_prest]
  
  # Keep only useful features
  features = c('id','start','stop','status','sex','age_in','Nm','sum_past_qt_prest')
  sel_df = sel_df[,..features]
  
  return(sel_df)
}