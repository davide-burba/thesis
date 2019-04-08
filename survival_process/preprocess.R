
path = '/Users/davide/Documents/universita/tesi/src/survival_process'
setwd(path)

require(data.table)
require(lubridate)
source('../preprocess/utils/minor.R')

# Load original data and fpca scores
load('../../data/dati_HF_sample.RData')
load('../../data/score_fpca_ACE.RData')
load('../../data/score_fpca_aldosteronics.RData')
load('../../data/score_fpca_beta.RData')
load('../../data/score_fpca_hospitalisation.RData')


# Rename original dataset and features
df = bla
rm(bla)
df = rename_features(df)

### Reformat data; one row for each selected patient with features [id,sex,age_in,time_event,status]:
# select rows relative to already selected patients
# add time_event column (time after 1 year after first hospitalization)
# add age_in feature (at first hospitalization) and add one year (time of follow-up)
# group by patient
df = df[id %in% unique(score_ACE$id)]
df[,'time_event' := data_studio_out - (data_rif_ev + make_difftime('days' = 366))]
df = add_age_in(df,verbose = TRUE)
df[,'age_in' := age_in+1]
new_df = df[,list(
  'sex' = first(sex),
  'age_in' = first(age_in),
  'time_event' = first(time_event),
  'status' = first(desc_studio_out)
),by=c("id")]

# minors
new_df[,status:=ifelse(status== 'DECEDUTO',1,0)]
new_df[,time_event:=as.numeric(new_df$time_event)]


### Add fpca scores
setorderv(new_df, c('id')) 
setorderv(score_ACE, c('id'))
#setorderv(score_aldosteronics, c('id'))
setorderv(score_beta, c('id')) 
setorderv(score_hospitalisation, c('id'))


new_df[,'ACE_PC1' := score_ACE$PC1]
new_df[,'ACE_PC2' := score_ACE$PC2]
new_df[,'aldosteronics_PC1' := score_aldosteronics$PC1]
new_df[,'aldosteronics_PC2' := score_aldosteronics$PC2]
new_df[,'beta_PC1' := score_beta$PC1]
new_df[,'beta_PC2' := score_beta$PC2]
new_df[,'hospitalisation_PC1' := score_hospitalisation$PC1]
new_df[,'hospitalisation_PC2' := score_hospitalisation$PC2]


# select 30% of data set for testing purpose
set.seed(185)
test_ids = sample(new_df$id,round(0.3*length(new_df$id)))
print(paste('Selected',length(test_ids),'patiets for test set'))

# split test/train
test_df = new_df[id %in% test_ids]
new_df = new_df[!id %in% test_ids]

save(new_df, file = '../../data/main_process_preprocessed_data.RData')
save(test_df, file = '../../data/main_process_preprocessed_data_test.RData')
write.csv(new_df, file = '../../data/main_process_preprocessed_data.csv',row.names=FALSE)
write.csv(test_df, file = '../../data/main_process_preprocessed_data_test.csv',row.names=FALSE)

