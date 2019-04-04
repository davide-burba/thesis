
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
# set mark and features 
ACE_mark = beta_mark = aldosteronics_mark = hospitalisation_mark = 'qt_prest_sum'
ACE_constant_variables = 
  beta_constant_variables = 
  aldosteronics_constant_variables =
  hospitalisation_constant_variables =c('sex','age_in')


### Select cohort patients
# Select events relative to patients who survived the monyhs_follow_up period
# Add age_in feature: age at dismission from first hospitalisation (i.e. t=0)
# Add time_event column (days from date of first discharge for HF event)
sel_df = df %>% 
  rename_features %>%
  keep_patients_survived_minimum_period(months_follow_up,verbose) %>%
  add_age_in(verbose) %>%
  add_time_from_first_discharge(verbose)


if(verbose){cat('\n');print('*********************** Preprocessing ACE dataset ***********************')}
### ACE drugs: select events and prepare dataset
# Select ACE DRUGS pharmacological events
# Keep only follow-up events
# Group togheter the concurrent events for same patient (0.9% of the cases,  MPR i.e. consider sum of qt_prest_sum)
# Add censoring due to follow-up (status=1 means uncensored)
# Reformat dataset in format as requested by survival::coxph (new columns: start,stop,Nm,y)
# Include selected patients which did not have ACE events in the follow-up (only one censored observation)
ACE_df = sel_df %>% 
  keep_only_type_events('ACE',verbose) %>%  
  keep_only_follow_up_events(months_follow_up,verbose) %>%
  set_mark_and_variables(ACE_mark,ACE_constant_variables, fill_NA = TRUE) %>%
  group_concurrent_events(sum,verbose) %>%
  add_censored_observations(verbose) %>%
  reformat_dataset(verbose) %>%
  include_patients_without_events(sel_df,ACE_constant_variables,verbose)


if(verbose){cat('\n');print('*********************** Preprocessing beta dataset ***********************')}
### beta drugs: select events and prepare dataset
# Select beta DRUGS pharmacological events
# Keep only follow-up events
# Group togheter the concurrent events for same patient (0.2% of the cases,consider sum of qt_prest_sum)
# Add censoring due to follow-up (status=1 means uncensored)
# Reformat dataset in format as requested by survival::coxph (new columns: start,stop,Nm,y)
# Include selected patients which did not have beta events in the follow-up (only one censored observation)
beta_df = sel_df %>% 
  keep_only_type_events('ATC_beta_blockers',verbose) %>%  
  keep_only_follow_up_events(months_follow_up,verbose) %>%
  set_mark_and_variables(beta_mark,beta_constant_variables, fill_NA = TRUE) %>%
  group_concurrent_events(sum,verbose) %>%
  add_censored_observations(verbose) %>%
  reformat_dataset(verbose) %>%
  include_patients_without_events(sel_df,beta_constant_variables,verbose)


if(verbose){cat('\n');print('*********************** Preprocessing aldosteronics dataset ***********************')}
### aldosteronics drugs: select events and prepare dataset
# Select aldosteronics DRUGS pharmacological events
# Keep only follow-up events
# Group togheter the concurrent events for same patient (0.3% of the cases,consider sum of qt_prest_sum)
# Add censoring due to follow-up (status=1 means uncensored)
# Reformat dataset in format as requested by survival::coxph (new columns: start,stop,Nm,y)
# Include selected patients which did not have aldosteronics events in the follow-up (only one censored observation)
aldosteronics_df = sel_df %>% 
  keep_only_type_events('ATC_anti_aldosteronics',verbose) %>%  
  keep_only_follow_up_events(months_follow_up,verbose) %>%
  set_mark_and_variables(aldosteronics_mark,aldosteronics_constant_variables, fill_NA = TRUE) %>%
  group_concurrent_events(sum,verbose) %>%
  add_censored_observations(verbose) %>%
  reformat_dataset(verbose) %>%
  include_patients_without_events(sel_df,aldosteronics_constant_variables,verbose)


if(verbose){cat('\n');print('*********************** Preprocessing hospitalisation dataset ***********************')}
### hospitalisation times: select events and prepare dataset
# Select hospitalisation  events
# Keep only follow-up events (all patients have event at time 0, we drop it)
# Group togheter the concurrent events for same patient (0.1% of the cases,consider max of y)
# Add censoring due to follow-up (status=1 means uncensored)
# Reformat dataset in format as requested by survival::coxph (new columns: start,stop,Nm,y)
# Include selected patients which did not have hospitalisation events in the follow-up (only one censored observation)
hospitalisation_df = sel_df %>% 
  keep_only_type_events('hospitalisation',verbose) %>%  
  keep_only_follow_up_events(months_follow_up,verbose, drop_events_time_0 = TRUE) %>%
  set_mark_and_variables(hospitalisation_mark,hospitalisation_constant_variables, fill_NA = TRUE) %>%
  group_concurrent_events(max,verbose) %>%
  add_censored_observations(verbose) %>%
  reformat_dataset(verbose) %>%
  include_patients_without_events(sel_df,hospitalisation_constant_variables,verbose)


### save 
save(ACE_df, file = '../../data/preprocessed_data_ACE.RData')
save(beta_df, file = '../../data/preprocessed_data_beta.RData')
save(aldosteronics_df, file = '../../data/preprocessed_data_aldosteronics.RData')
save(hospitalisation_df, file = '../../data/preprocessed_data_hospitalisation.RData')





