
#setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
path = '/Users/davide/Documents/universita/tesi/src/preprocess'
setwd(path)

library(magrittr)
source('utils.R')

# load data
load('../../data/dati_HF_sample.RData')
df = data.frame(bla)
rm(bla)

# parameters
verbose = TRUE
max_date_first_discharge = '2011-12-31'
months_follow_up = 12
day_max = 366

# clean
df = df %>%
  keep_only_useful_columns

# select cohort patients
sel_df = df %>% 
  keep_only_pharmacological_events(verbose) %>%
  keep_only_ACE_drugs(verbose) %>%
  keep_patients_first_discharge_before_date(max_date_first_discharge,verbose) %>%
  keep_patients_survived_minimum_period(months_follow_up,verbose) %>%
  keep_only_follow_up_events(months_follow_up,verbose)

# add relative time
sel_df = add_relative_time(sel_df)

# compute counting processes
processes = compute_counting_processes(sel_df,day_max,months_follow_up)
melted_processes = melt_process_patient(processes)



