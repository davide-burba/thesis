
require(lubridate)


#' Select rows with events of desired type.
keep_only_type_events = function(df, type_event, verbose = FALSE){
  if(type_event == 'pharmacological'){ 
    sel_df = df[which(df$tipo_prest == 30),]
  }else if(type_event == 'hospitalization'){
    sel_df = df[which(df$tipo_prest == 41),]# TMP!!!! check!!!!
  }
  
  if(verbose){
    print(paste('keep_only_type_events: selected',dim(sel_df)[1],'rows out of',dim(df)[1]))
  }
  return(sel_df)
}


#' Select rows relative to desired class_prest:
keep_only_class_events = function(df, class_event, verbose = FALSE){
  if(class_event == 'ACE_drugs'){ 
    indexes = sort(c(grep('C09A', df$class_prest),grep('C09B', df$class_prest) ))
    sel_df = df[indexes,]
  }
  if(verbose){
    print(paste('keep_only_class_events: selected',dim(sel_df)[1],'rows out of',dim(df)[1]))
  }
  return(sel_df)
}


#' Keep only events in follow-up period
keep_only_follow_up_events = function(df,months_follow_up = 12, verbose = FALSE){
  start_follow_up = df$data_rif_ev
  end_follow_up = start_follow_up %m+% months(months_follow_up)
  sel_df = df[which(df$data_prest<end_follow_up),] # there are no-events before start_follow_up
  if(verbose){
    print(paste('keep_only_follow_up_events: selected',dim(sel_df)[1],'rows out of',dim(df)[1],', (',length(unique(sel_df$id)),' patients)'))
  }
  return(sel_df)
}

