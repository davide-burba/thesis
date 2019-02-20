
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(magrittr)
source('utils.R')

# load data
load('../../data/dati_HF_sample.RData')
df = bla
rm(bla)

# parameters
verbose = TRUE
max_date_first_discharge = '2011-12-31'
months_follow_up = 12

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


# have a look at some patients
library(ggplot2)
selected_patient = sample(unique(sel_df$COD_REG),5)
tmp = sel_df[sel_df$COD_REG %in% selected_patient,]
ggplot(tmp) + geom_line(aes(x=data_prest, y=qt_prest_Sum, color=factor(COD_REG)))




