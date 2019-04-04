
require(lubridate)


#' Select rows with events of desired type.
keep_only_type_events = function(sel_df, type_event, verbose = FALSE){
  if(type_event == 'ACE'){ 
    indexes = sort(c(grep('C09A', sel_df$class_prest),grep('C09B', sel_df$class_prest) ))
    events_df = sel_df[indexes,]
  }else if(type_event == 'ATC_beta_blockers'){
    indexes = c(grep('C07', sel_df$class_prest))
    events_df = sel_df[indexes,]
  }else if(type_event == 'ATC_anti_aldosteronics'){
    indexes = sort(c(grep('C03D', sel_df$class_prest),grep('C03E', sel_df$class_prest) ))
    events_df = sel_df[indexes,]
  }else if(type_event == 'hospitalisation'){
    events_df = sel_df[which(sel_df$tipo_prest == 41),]
  }else{
    print('Unknown type_event')
    return(NULL)
  }
  
  if(verbose){
    print(paste('keep_only_type_events: selected',dim(events_df)[1],'rows out of',dim(sel_df)[1]))
  }
  return(events_df)
}


#' Keep only events in follow-up period
keep_only_follow_up_events = function(sel_df,months_follow_up = 12, verbose = FALSE, drop_events_time_0 = FALSE){
  start_follow_up = sel_df$data_rif_ev
  end_follow_up = start_follow_up %m+% months(months_follow_up)
  events_df = sel_df[which(sel_df$data_prest<end_follow_up),] # there are no-events before start_follow_up
  if(drop_events_time_0){
    events_df = events_df[time_event > 0]
  }
  if(verbose){
    print(paste('keep_only_follow_up_events: selected',dim(events_df)[1],'rows out of',dim(sel_df)[1],', (',length(unique(events_df$id)),' patients)'))
  }
  return(events_df)
}

