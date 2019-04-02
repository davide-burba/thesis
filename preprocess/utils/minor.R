
require(data.table)


# Rename features
rename_features = function(df){
  # Let's be more international 
  setnames(df, 
           old = c('COD_REG','SESSO','qt_prest_Sum'), 
           new = c('id','sex','qt_prest_sum'))
  return(df)
}


# Add age_in column: age at data_rif_ev feature
add_age_in = function(df,verbose = FALSE){
  if(verbose){print('Adding age_in column (age at t=0, i.e. at dismission from first hospitalization)')}
  age_in = c()
  patient_ids = unique(df$id)
  pb <- txtProgressBar(min = 0, max = length(patient_ids), style = 3)
  for(i in 1:length(patient_ids)){
    setTxtProgressBar(pb, i)
    patient_id = patient_ids[i]
    tmp = df[id == patient_id]
    patient_age_in = tmp[data_rif_ev == data_prest,'eta_Min'][[1]][1]
    age_in = c(age_in,rep(patient_age_in,dim(tmp)[1]))
  }
  cat("\n")
  df[,'age_in' := age_in]
  return(df)
}


#' Add time_event column (days from date of first discharge for HF event)
add_time_from_first_discharge = function(df,verbose = FALSE){
  if(verbose){print('Adding time_event column (days from date of first discharge for HF event)')}
  df[,'time_event' := df$data_prest - df$data_rif_ev]
  return(df)
}


#' Change name mark column to 'mark' (so we can easily use the same code for different marks) and select features
set_mark_and_variables = function(events_df,mark,constant_variables, fill_NA = FALSE){
  setnames(events_df, 
           old = mark, 
           new = 'mark')
  variables = c('id','time_event','mark',constant_variables)
  events_df = events_df[,..variables]
  if(fill_NA){
    # fill NA marks with median
    indexes = is.na(events_df$mark)
    value = median(events_df$mark,na.rm = TRUE)
    events_df[indexes,'mark'] = value
    print(paste('Filled',round(sum(indexes)/dim(events_df)[1]*100,3),'% NAs in mark with median = ',value))
  }
  return(events_df)
}


# Group togheter the concurrent events for same patient
group_concurrent_events = function(events_df,group = sum,verbose = FALSE){
  old_length = dim(events_df)[1]
  # select constant features
  const_features = colnames(events_df)[!colnames(events_df) %in% c('id','time_event','mark')]
  # merge const features and marks
  grouped_const_features = events_df[ , lapply(.SD, first), by = c('id','time_event'), .SDcols = const_features]
  grouped_mark = events_df[, .(mark= sum(mark)), by = c('id','time_event')]
  # put everything togheter
  events_df = merge(grouped_const_features, grouped_mark,  by = c('id','time_event'))
  if(verbose){
    new_length = dim(events_df)[1]
    print(paste('Grouping togheter the concurrent events for same patient; discarded',
                          round((old_length-new_length)/old_length*100,1),'% of the events'))
  }
  return(events_df)
}


# Select features
#select_features = function(sel_df){
#  # Keep only useful columns 
#  columns_to_keep = c('id','time_event',"age_in",'SESSO','qt_prest_Sum')
#  sel_df = sel_df[,..columns_to_keep]
#  return(sel_df)
#}


