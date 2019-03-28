
path = '/Users/davide/Documents/universita/tesi/src/preprocess'
setwd(path)

library(magrittr)
library(data.table)
library(survival)
source('utils/minor.R')
source('utils/select_cohort.R')
source('utils/add_censored_observations.R')
source('utils/reformat_dataset.R')


# Load data
load('../../data/dati_HF_sample.RData')
# Rename dataset
df = bla
rm(bla)


# PARAMETERS
verbose = TRUE
max_date_first_discharge = '2011-12-31'
months_follow_up = 12


### Add age_in feature: age at dismission from first hospitalization (i.e. t=0)
df = df %>%
  add_age_in(verbose)

### Select cohort patients (and events)
# We select events for ACE drugs that survived at least one year and we consider 
# events in this period (therefore discarding also patients that survived more than
# one year but without events in the first year)
sel_df = df %>% 
  keep_only_type_events('pharmacological',verbose) %>%  
  keep_only_class_events('ACE_drugs',verbose) %>%
  keep_patients_first_discharge_before_date(max_date_first_discharge,verbose) %>% # not necessary (included in next step)
  keep_patients_survived_minimum_period(months_follow_up,verbose) %>%
  keep_only_follow_up_events(months_follow_up,verbose)

### Minors
# Add time_event column (days from date of first discharge for HF event)
# Select useful features, fix naming
# Group togheter the concurrent events for same patient (0.7% of the cases,  MPR i.e. consider sum of qt_prest_sum)
sel_df = sel_df %>% 
  add_time_from_first_discharge(verbose) %>% 
  select_rename_features %>%
  group_concurrent_events(verbose)

### Add censoring due to follow-up (status=1 means uncensored)
sel_df = sel_df %>%
  add_censored_observations(verbose)
 
### Reformat dataset. It adds the following new columns:
# - start,stop: interval times as requested by survival::coxph
# - Nm: number of events before the considered one for each patient N_i(t-)
# - sum_past_qt_prest: sum of the past qt_prest_sum for each patient
sel_df = sel_df %>%
  reformat_dataset(verbose)


if(verbose){
  print('Head preprocessed dataset:')
  print(head(sel_df))
}
# save (both RData and csv)
write.csv(sel_df,file = '../../data/preprocessed_data.csv',row.names = FALSE)
save(sel_df, file = '../../data/preprocessed_data.RData')

