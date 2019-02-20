
require(lubridate)
library(data.table)

#' keep only useful columns 
keep_only_useful_columns = function(df){
  columns_to_keep = c(
    "COD_REG",
    "data_rif_ev",
    "SESSO",
    "eta_Min",
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
  sel_df = df[which(df$tipo_prest == 30),]
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
  sel_df = df[which(df$data_rif_ev < max_date_first_discharge),]
  if(verbose){
    print(paste('keep_patients_first_discharge_before_date: selected',dim(sel_df)[1],'rows out of',dim(df)[1]))
  }
  return(sel_df)
}


#' Keep only patients survived a minimum period after first discharge
keep_patients_survived_minimum_period = function(df, months_follow_up = 12, verbose = FALSE){
  months_follow_up_later = df$data_rif_ev %m+% months(months_follow_up)
  sel_df = df[which(df$data_studio_out > months_follow_up_later),]
  if(verbose){
    print(paste('keep_patients_survived_minimum_period',dim(sel_df)[1],'rows out of',dim(df)[1]))
  }
  return(sel_df)
}


#' Keep only events in follow-up period
keep_only_follow_up_events = function(df,months_follow_up = 12, verbose = FALSE){
  start_follow_up = df$data_rif_ev
  end_follow_up = start_follow_up %m+% months(months_follow_up)
  sel_df = df[which(df$data_prest<end_follow_up),] # there are no-events before start_follow_up
  if(verbose){
    print(paste('keep_only_follow_up_events',dim(sel_df)[1],'rows out of',dim(df)[1]))
  }
  return(sel_df)
}


#' Add 'time' column: relative time (0 equal to day of ismission first hospitalization)
add_relative_time = function(df){
  df['time'] = df['data_prest'] - df['data_rif_ev'] + 1
  return(df)
} 


#' Compute the realized counting process, returns a new dataframe
compute_counting_processes = function(df,day_max = 366, months_follow_up = 12){
  new_df = data.frame(unique(df$COD_REG))
  colnames(new_df)[1] = 'patient'
  new_df = .add_fixed_variables(new_df,df,months_follow_up)
  new_df['pharma_process'] = c()
  for(i in 1:dim(new_df)[1]){
    patient = new_df[i,'patient']
    patient_events = df[which(df$COD_REG == patient),]
    pharmacological_process = .get_pharmacological_process_one_patient(patient_events,day_max)
    new_df[i,'pharma_process'][[1]] = list(pharmacological_process)
  }
  return(new_df)
}


#' Compute pharmacological counting process for one patient
.get_pharmacological_process_one_patient = function(patient_events,day_max){
  pharmacological_process = rep(0,day_max)
  for (index in 1:dim(patient_events)[1]){
    t = patient_events[index,'time']
    jump = patient_events[index,'qt_prest_Sum']
    pharmacological_process[t:day_max] = pharmacological_process[t:day_max] + jump
  }
  pharmacological_process = c(0,pharmacological_process)
  return(pharmacological_process)
}


#' Add fixed variables to dataset in new format
.add_fixed_variables = function(new_df,df,months_follow_up){
  new_df['sex'] = c()
  new_df['min_age'] = c()
  new_df['outcome'] = c()
  new_df['survival_time'] = c()
  for(i in 1:dim(new_df)[1]){
    patient = new_df[i,'patient']
    sex = df[df$COD_REG==patient,'SESSO'][1]
    min_age = df[df$COD_REG==patient,'eta_Min'][1]
    outcome = df[df$COD_REG==patient,'desc_studio_out'][1]
    data_studio_out = df[df$COD_REG==patient,'data_studio_out'][1]
    data_end_follow_up = df[df$COD_REG==patient,'data_rif_ev'][1] %m+% months(months_follow_up)
    survival_time = data_studio_out - data_end_follow_up
    new_df[i,'sex'] = sex
    new_df[i,'min_age'] = min_age
    new_df[i,'outcome'] = outcome
    new_df[i,'survival_time'] = survival_time
  }
  return(new_df)
}



#' Melt the dataframe returned by compute_counting_processes; useful for ggplot
melt_process_patient = function(processes){
  melted_process = list()
  i = 1
  for(patient in processes$patient){
    tmp = processes[processes$patient == patient,]
    sex = tmp$sex
    min_age = tmp$min_age
    outcome = tmp$outcome
    survival_time = tmp$survival_time
    pharma_process = tmp$pharma_process[[1]]
    
    pat_data = data.frame(pharma_process)
    pat_data['time'] = 1:length(pharma_process) - 1
    pat_data['patient'] = patient
    pat_data['sex'] = sex
    pat_data['min_age'] = min_age
    pat_data['outcome'] = outcome
    pat_data['survival_time'] = survival_time
    
    melted_process[[i]] = pat_data
    i = i+1
  }
  melted_process = rbindlist(melted_process)
  return(melted_process)
}

