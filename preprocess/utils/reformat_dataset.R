
#' reformat_dataset: add the following new columns
#' - start,stop: interval times as requested by survival::coxph
#' - Nm: number of events before the considered one for each patient N_i(t-)
#' - y: sum of the past marks for each patient
reformat_dataset = function(events_df,verbose = FALSE){
  if(verbose){print('Adding new columns: start,stop, Nm [N_i(t-)], y [sum of past marks]')}
  start = stop = Nm = y = c()
  patient_ids = unique(events_df$id)
  pb <- txtProgressBar(min = 0, max = length(patient_ids), style = 3)
  for(i in 1:length(patient_ids)){
    setTxtProgressBar(pb, i)
    patient_id = patient_ids[i]
    tmp = events_df[events_df$id == patient_id,]
    
    # times (start,stop]
    events_time = tmp[,'time_event']
    events_time = as.numeric(unlist(events_time))
    stop = c(stop,events_time)
    start = c(start,c(-0.5,events_time[-length(events_time)]))
    # N_i(t-)
    Nm = c(Nm,0:(length(events_time)-1))
    # y
    y_patient = tmp[,'mark']
    sum_y_patient = cumsum(as.numeric(unlist(y_patient)))
    y_patient = c(0,sum_y_patient[-length(sum_y_patient)])
    y = c(y, y_patient)
  }
  cat("\n")
  events_df[,'start' := start]
  events_df[,'stop' := stop]
  events_df[,'Nm' := Nm]
  events_df[,'y' := y]
  
  # Keep only relevant features (discard mark to avoid confusion)
  const_features = colnames(events_df)[!colnames(events_df) %in% c('id','time_event','status','mark','start','stop','Nm','y')]
  features = c('id','start','stop','status',const_features,'Nm','y')
  events_df = events_df[,..features]
  
  return(events_df)
}


#' Include selected patients which did not have events in the follow-up (only one censored observation)
include_patients_without_events = function(events_df,sel_df,const_features,verbose = FALSE){
  selected_patients = unique(sel_df$id)
  patients_with_events = unique(events_df$id)
  patient_without_events = selected_patients[!selected_patients %in% patients_with_events]
  if(verbose){
    percentage_without_events = round(100*length(patient_without_events)/length(selected_patients),1)
    print(paste('Patients not experiencing events: ',percentage_without_events,'%'))
  }
  # set non-trivial features
  tmp = sel_df[id %in% patient_without_events]
  tmp = tmp[,lapply(.SD, first), by = c('id'), .SDcols = const_features]
  # set trivial features
  time_start = min(events_df$start)
  time_stop = max(events_df$stop)
  tmp[,'start':=time_start]
  tmp[,'stop':=time_stop]
  tmp[,'status':=0]
  tmp[,'Nm':=0]
  tmp[,'y':=0]
  # fix naming and order
  tmp[,'id':=id]
  features = colnames(events_df)
  tmp = tmp[,..features]
  # put togheter
  events_df = rbind(events_df,tmp)
  setorderv(events_df, c('id','start')) 
  return(events_df)
}