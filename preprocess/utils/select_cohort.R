
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





