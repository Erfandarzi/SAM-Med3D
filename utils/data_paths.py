img_datas = ["data/train/brain_lesion/Task502_BCHUNC","data/train/brain_lesion/Task501_HIE"]
all_datasets=["Task502_BCHUNC","HIE_smalldiffuse"]

all_classes = [
'COVID_lesion',
'adrenal',
'adrenal_gland_left',
'adrenal_gland_right',
'airway',
'aorta',
'autochthon_left',
'autochthon_right',
'bilateral_optic_nerves',
'bilateral_parotid_glands',
'bilateral_submandibular_glands',
'bladder',
'bone',
'brain',
'brain_lesion',
'brainstem',
'buckle_rib_fracture',
'caudate_left',
'caudate_right',
'cerebellum',
'cerebral_microbleed',
'cerebrospinal_fluid',
'clavicula_left',
'clavicula_right',
'cocygis',
'colon',
'colon_cancer_primaries',
'deep_gray_matter',
'displaced_rib_fracture',
'duodenum',
'edema',
'enhancing_tumor',
'esophagus',
'external_cerebrospinal_fluid',
'face',
'femur_left',
'femur_right',
'gallbladder',
'gluteus_maximus_left',
'gluteus_maximus_right',
'gluteus_medius_left',
'gluteus_medius_right',
'gluteus_minimus_left',
'gluteus_minimus_right',
'gray_matter',
'head_of_femur_left',
'head_of_femur_right',
'heart',
'heart_ascending_aorta',
'heart_atrium_left',
'heart_atrium_left_scars',
'heart_atrium_right',
'heart_blood_pool',
'heart_left_atrium_blood_cavity',
'heart_left_ventricle_blood_cavity',
'heart_left_ventricular_myocardium',
'heart_myocardium',
'heart_myocardium_left',
'heart_right_atrium_blood_cavity',
'heart_right_ventricle_blood_cavity',
'heart_ventricle_left',
'heart_ventricle_right',
'hepatic_tumor',
'hepatic_vessels',
'hip_left',
'hip_right',
'hippocampus_anterior',
'hippocampus_posterior',
'humerus_left',
'humerus_right',
'iliac_artery_left',
'iliac_artery_right',
'iliac_vena_left',
'iliac_vena_right',
'iliopsoas_left',
'iliopsoas_right',
'inferior_vena_cava',
'intestine',
'ischemic_stroke_lesion',
'kidney',
'kidney_cyst',
'kidney_left',
'kidney_right',
'kidney_tumor',
'left_eye',
'left_inner_ear',
'left_lens',
'left_mandible',
'left_middle_ear',
'left_optical_nerve',
'left_parotid_gland',
'left_temporal_lobes',
'left_temporomandibular_joint',
'left_ventricular_blood_pool',
'left_ventricular_myocardial_edema',
'left_ventricular_myocardial_scars',
'left_ventricular_normal_myocardium',
'liver',
'liver_tumor',
'lumbar_vertebra',
'lung',
'lung_cancer',
'lung_infections',
'lung_left',
'lung_lower_lobe_left',
'lung_lower_lobe_right',
'lung_middle_lobe_right',
'lung_node',
'lung_right',
'lung_upper_lobe_left',
'lung_upper_lobe_right',
'lung_vessel',
'mandible',
'matter_tracts',
'multiple_sclerosis_lesion',
'myocardial_infarction',
'nasopharynx_cancer',
'no_reflow',
'non_displaced_rib_fracture',
'non_enhancing_tumor',
'optic_chiasm',
'other_pathology',
'pancreas',
'pancreatic_tumor_mass',
'pituitary',
'portal_vein_and_splenic_vein',
'prostate',
'prostate_and_uterus',
'prostate_peripheral_zone',
'prostate_transition_zone',
'pulmonary_artery',
'rectum',
'renal_artery',
'renal_vein',
'rib_left_1',
'rib_left_10',
'rib_left_11',
'rib_left_12',
'rib_left_2',
'rib_left_3',
'rib_left_4',
'rib_left_5',
'rib_left_6',
'rib_left_7',
'rib_left_8',
'rib_left_9',
'rib_right_1',
'rib_right_10',
'rib_right_11',
'rib_right_12',
'rib_right_2',
'rib_right_3',
'rib_right_4',
'rib_right_5',
'rib_right_6',
'rib_right_7',
'rib_right_8',
'rib_right_9',
'right_eye',
'right_inner_ear',
'right_lens',
'right_mandible',
'right_middle_ear',
'right_optical_nerve',
'right_parotid_gland',
'right_temporal_lobes',
'right_temporomandibular_joint',
'right_ventricular_blood_pool',
'sacrum',
'scapula_left',
'scapula_right',
'segmental_rib_fracture',
'small_bowel',
'spinal_cord',
'spleen',
'stomach',
'trachea',
'unidentified_rib_fracture',
'urinary_bladder',
'uterus',
'ventricles',
'vertebrae_C1',
'vertebrae_C2',
'vertebrae_C3',
'vertebrae_C4',
'vertebrae_C5',
'vertebrae_C6',
'vertebrae_C7',
'vertebrae_L1',
'vertebrae_L2',
'vertebrae_L3',
'vertebrae_L4',
'vertebrae_L5',
'vertebrae_L6',
'vertebrae_T1',
'vertebrae_T10',
'vertebrae_T11',
'vertebrae_T12',
'vertebrae_T13',
'vertebrae_T2',
'vertebrae_T3',
'vertebrae_T4',
'vertebrae_T5',
'vertebrae_T6',
'vertebrae_T7',
'vertebrae_T8',
'vertebrae_T9',
'white_matter',
'white_matter_hyperintensity',
]
