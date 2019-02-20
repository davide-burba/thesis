
require(lubridate)

#' keep only useful columns 
keep_only_useful_columns = function(df){
  columns_to_keep = c(
    "COD_REG",
    "data_rif_ev",
    "SESSO",
    "data_studio_out",
    "desc_studio_out",
    "data_prest",
    "tipo_prest", 
    "class_prest",
    "qt_prest_Sum",      
    "val_prest_Sum",
    "qt_prest_NMiss",
    "val_prest_NMiss"
  )
  return(df[,columns_to_keep])
}


#' Select rows with drug purchases events.
keep_only_pharmacological_events = function(df, verbose = FALSE){
  sel_df = df[df$tipo_prest == 30,]
  if(verbose){
    print(paste('keep_only_pharmacological_events: selected',dim(sel_df)[1],'rows out of',dim(df)[1]))
  }
  return(sel_df)
}


#' Select rows relative to ACE drugs:
keep_only_ACE_drugs = function(df, verbose = FALSE){
  indexes = sort(c(grep('C09A', df$class_prest),grep('C09B', df$class_prest) ))
  sel_df = df[indexes,]
  if(verbose){
    print(paste('keep_only_ACE_drugs: selected',dim(sel_df)[1],'rows out of',dim(df)[1]))
  }
  return(sel_df)
}


#' Select patients with the first discharge before a specified date
keep_patients_first_discharge_before_date = function(df, max_date_first_discharge = '2011-12-31',verbose = FALSE){
  sel_df = df[df$data_rif_ev < max_date_first_discharge,]
  if(verbose){
    print(paste('keep_patients_first_discharge_before_date: selected',dim(sel_df)[1],'rows out of',dim(df)[1]))
  }
  return(sel_df)
}


#' Select patient with the first discharge before a specified date
keep_patients_first_discharge_before_date = function(df, max_date_first_discharge = '2011-12-31',verbose = FALSE){
  sel_df = df[df$data_rif_ev < max_date_first_discharge,]
  if(verbose){
    print(paste('keep_patients_first_discharge_before_date: selected',dim(sel_df)[1],'rows out of',dim(df)[1]))
  }
  return(sel_df)
}


#' Keep only patients survived a minimum period after first discharge
keep_patients_survived_minimum_period = function(df, months_follow_up = 12, verbose = FALSE){
  months_follow_up_later = df$data_rif_ev %m+% months(months_follow_up)
  sel_df = df[df$data_studio_out > months_follow_up_later,]
  if(verbose){
    print(paste('keep_patients_survived_minimum_period',dim(sel_df)[1],'rows out of',dim(df)[1]))
  }
  return(sel_df)
}

#' Keep only events in follow-up period
keep_only_follow_up_events = function(df,months_follow_up = 12, verbose = FALSE){
  start_follow_up = df$data_rif_ev
  end_follow_up = start_follow_up %m+% months(months_follow_up)
  sel_df = df[df$data_prest<end_follow_up,] # there are no-events before start_follow_up
  if(verbose){
    print(paste('keep_only_follow_up_events',dim(sel_df)[1],'rows out of',dim(df)[1]))
  }
  return(sel_df)
}



