
require(lubridate)


#' Keep only patients survived a minimum period after first discharge
keep_patients_survived_minimum_period = function(df, months_follow_up = 12, verbose = FALSE){
  months_follow_up_later = df$data_rif_ev %m+% months(months_follow_up)
  sel_df = df[which(df$data_studio_out > months_follow_up_later),]
  if(verbose){
    n_patients_selected = length(unique(sel_df$id))
    n_patients = length(unique(df$id))
    perc_discarded = round(100*(n_patients - n_patients_selected)/n_patients,2)
    print(paste('keep_patients_survived_minimum_period: discarded patients died in the first', months_follow_up,
                'months, corresponding to the ',perc_discarded,'% of the total number of patients.'))
  }
  return(sel_df)
}


#' Select patient with the first discharge before a specified date
#keep_patients_first_discharge_before_date = function(df, max_date_first_discharge = '2011-12-31',verbose = FALSE){
#  sel_df = df[which(df$data_rif_ev < max_date_first_discharge),]
#  if(verbose){
#    print(paste('keep_patients_first_discharge_before_date: selected',dim(sel_df)[1],'rows out of',dim(df)[1]))
#  }
#  return(sel_df)
#}



