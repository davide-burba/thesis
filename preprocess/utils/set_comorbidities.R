

comorbidities = c(
  'metastatic',
  'chf dementia',
  'renal',
  'wtloss',
  'hemiplegia',
  'alcohol',
  'tumor',
  'arrhythmia',
  'pulmonarydz',
  'coagulopathy',
  'compdiabetes',
  'anemia',
  'electrolytes',
  'liver',
  'pvd',
  'psychosis',
  'pulmcirc',
  'hivaids',
  'hypertension'
)

tmp = c(
  #"n_rec", 
  "metastatic_romano",
  "chf_romano",
  "dementia_romano",
  "renal_elixhauser",
  "wtloss_elixhauser",
  "hemiplegia_romano",
  "alcohol_elixhauser",
  "tumor_romano",
  "arrhythmia_elixhauser",
  "pulmonarydz_romano",
  "coagulopathy_elixhauser",
  "compdiabetes_elixhauser",
  "anemia_elixhauser",
  "electrolytes_elixhauser",
  "liver_elixhauser",
  "pvd_elixhauser",
  "psychosis_elixhauser",
  "pulmcirc_elixhauser",
  "hivaids_romano",
  "hypertension_elixhauser",
  "metastatic_romano_inric",
  "chf_romano_inric",
  "dementia_romano_inric",
  "renal_elixhauser_inric",
  "wtloss_elixhauser_inric",
  "hemiplegia_romano_inric",
  "alcohol_elixhauser_inric",
  "tumor_romano_inric",
  "arrhythmia_elixhauser_inric",
  "pulmonarydz_romano_inric",
  "coagulopathy_elixhauser_inric",
  "compdiabetes_elixhauser_inric",
  "anemia_elixhauser_inric",
  "electrolytes_elixhauser_inric",
  "liver_elixhauser_inric",
  "pvd_elixhauser_inric",
  "psychosis_elixhauser_inric",
  "pulmcirc_elixhauser_inric",
  "hivaids_romano_inric",
  "hypertension_elixhauser_inric",
  "dec_intra",
  "flag_ti",
  "flag_cardiochir",
  "flag_ICD",
  "flag_SHOCK",
  "flag_CABG",
  "flag_PTCA"
)


for (v in tmp){
  print(paste(
    v,
    sum(is.na(df[,..v]))/dim(df)[1], 
    dim( df[eval(as.name(v)) == 1,])[1]/dim(df)[1]
    ))
}
