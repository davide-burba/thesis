
require(data.table)


# Add age_in column: age at data_rif_ev feature
add_age_in = function(df,verbose = FALSE){
  if(verbose){print('Adding age_in column (age at t=0, i.e. at dismission from first hospitalization)')}
  age_in = c()
  patient_ids = unique(df$COD_REG)
  pb <- txtProgressBar(min = 0, max = length(patient_ids), style = 3)
  for(i in 1:length(patient_ids)){
    setTxtProgressBar(pb, i)
    patient_id = patient_ids[i]
    tmp = df[COD_REG == patient_id]
    patient_age_in = tmp[data_rif_ev == data_prest,'eta_Min'][[1]][1]
    age_in = c(age_in,rep(patient_age_in,dim(tmp)[1]))
  }
  df[,'age_in' := age_in]
  
  return(df)
}


# Add time_event column (days from date of first discharge for HF event)
add_time_from_first_discharge = function(df,verbose = FALSE){
  if(verbose){print('Adding time_event column (days from date of first discharge for HF event)')}
  sel_df[,'time_event' := sel_df$data_prest - sel_df$data_rif_ev]
  return(sel_df)
}


# Select, rename features
select_rename_features = function(sel_df){
  # Keep only useful columns 
  columns_to_keep = c('COD_REG','time_event',"age_in",'SESSO','qt_prest_Sum')
  sel_df = sel_df[,..columns_to_keep]
  # Let's be more international 
  colnames(sel_df) = c('id','time_event','age_in','sex','qt_prest_sum')
  return(sel_df)
}


# Group togheter the concurrent events for same patient (MPR i.e. consider sum of qt_prest_sum)
group_concurrent_events = function(sel_df,verbose = FALSE){
  if(verbose){print('Grouping togheter the concurrent events for same patient MPR i.e. consider sum of qt_prest_sum)')}
  sel_df = sel_df[,list(
    'sex' = first(sex),
    'age_in' = first(age_in),
    'qt_prest_sum' = sum(qt_prest_sum) # MPR approach
  ),by=c("id","time_event")]
  
  return(sel_df)
}