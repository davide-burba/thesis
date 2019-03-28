
path = '/Users/davide/Documents/universita/tesi/src/preprocess'
setwd(path)

library(magrittr)
library(data.table)
library(survival)
source('utils/minor.R')
source('utils/select_cohort.R')
source('utils/select_events.R')
source('utils/add_censored_observations.R')
source('utils/reformat_dataset.R')


# Load data
load('../../data/dati_HF_sample.RData')
# Rename dataset
df = bla
rm(bla)


### PARAMETERS
max_date_first_discharge = '2011-12-31'
months_follow_up = 12
verbose = TRUE
# ACE drugs process: mark and features 
ACE_mark = 'qt_prest_sum'
ACE_constant_variables = c('sex','age_in')


### Select cohort patients
# Select events relative to patients who survived the monyhs_follow_up period
# Add age_in feature: age at dismission from first hospitalization (i.e. t=0)
# Add time_event column (days from date of first discharge for HF event)
sel_df = df %>% 
  rename_features %>%
  keep_patients_survived_minimum_period(months_follow_up,verbose) %>%
  add_age_in(verbose) %>%
  add_time_from_first_discharge(verbose)


if(verbose){print('*********************** Preprocessing ACE dataset ***********************')}
### ACE drugs: select events
# Select pharmacological events
# Select only ACE DRUGS
# Keep only follow-up events
ACE_df = sel_df %>% 
  keep_only_type_events('pharmacological',verbose) %>%  # In this case this step is useless (included in next one)
  keep_only_class_events('ACE_drugs',verbose) %>%
  keep_only_follow_up_events(months_follow_up,verbose)
  
### ACE drugs: prepare dataset
# Group togheter the concurrent events for same patient (0.9% of the cases,  MPR i.e. consider sum of qt_prest_sum)
# Add censoring due to follow-up (status=1 means uncensored)
# Reformat dataset in format as requested by survival::coxph (new columns: start,stop,Nm,sum_past_qt_prest)
# Include selected patients which did not have ACE events in the follow-up (only one censored observation)
ACE_df = ACE_df %>%
  set_mark_and_variables(ACE_mark,ACE_constant_variables) %>%
  group_concurrent_events(sum,verbose) %>%
  add_censored_observations(verbose) %>%
  reformat_dataset(verbose) %>%
  include_patients_without_events(sel_df,ACE_constant_variables,verbose) # NB: check if it's correct to do it! Not done in previous version

if(verbose){
  print('Head preprocessed dataset:')
  print(head(ACE_df))
}

# save 
#write.csv(sel_df,file = '../../data/preprocessed_data_2.csv',row.names = FALSE)
save(ACE_df, file = '../../data/preprocessed_data_ACE_2.RData')

