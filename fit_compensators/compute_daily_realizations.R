

# compute dataframe of daily realizations in long format 
compute_daily_realizations = function(sel_df,times = c(0:365)){
  patient_ids = unique(sel_df$id)
  
  daily_realizations = NULL
  pb <- txtProgressBar(min = 0, max = length(patient_ids), style = 3)
  for(i in 1:length(patient_ids)){
    setTxtProgressBar(pb, i)
    patient_id = patient_ids[i]
    tmp = sel_df[id==patient_id]
    Nt = c()
    for(t in times){
      nt = tmp[start<=t,Nm]
      nt = nt[length(nt)]
      Nt = c(Nt,nt)
    }
    daily_realizations = rbind(daily_realizations,
                               data.frame(patient_id,time = times, Nt))
  }
  cat("\n")
  return(daily_realizations)
}  

