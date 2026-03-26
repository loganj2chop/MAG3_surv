# MAG3 Survival Model

### Clinical Risk
Clinical risk scores were derived using a Random Survival Forest trained on four features extracted from MAG3 renal scintigraphy and demographic data: differential unit volume, differential total volume, T½ half-life, and age. The outcome was time to urological complication, with censoring indicated by group membership. Feature engineering computed the absolute laterality difference in both total and per-unit renal volumes between the two kidneys. Sex and hydronephrosis grade were encoded numerically. The model was trained and evaluated using 2-fold stratified cross-validation, and out-of-fold risk scores were concatenated across folds to produce a single risk score per patient spanning the full cohort. These scores were merged back to patient record identifiers and used as the clinical modality input to the ensemble. 

### PMRI (images) Risk

pMRI Drainage Model
Dynamic contrast-enhanced pMRI drainage curves were extracted as time-series features representing renal clearance over the scan acquisition window. Features were loaded directly as columns from the processed feature matrix, with patient identifier, group label, and time-to-event retained separately. The outcome was time to urological complication with binary censoring. A Random Survival Forest with 1,600 estimators was trained and evaluated using 2-fold stratified cross-validation, and out-of-fold risk scores were assigned to each patient from the fold in which they served as the held-out set. Risk scores were merged back to patient identifiers and saved for ensemble integration.
