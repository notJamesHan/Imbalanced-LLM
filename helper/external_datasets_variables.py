########################################################################################################################
# heart
########################################################################################################################
# Used descriptions from: https://www.kaggle.com/code/azizozmen/heart-failure-predict-8-classification-techniques
chest_paint_types_list = {'TA': 'typical angina', 'ATA': 'atypical angina', 'NAP': 'non-anginal pain', 'ASY': 'asymptomatic'}
rest_ecg_results = {
    'Normal': 'normal',
    'ST': 'ST-T wave abnormality',
    'LVH': 'probable or definite left ventricular hypertrophy'
}
st_slopes = {'Up': 'upsloping', 'Flat': 'flat', 'Down': 'downsloping'}
template_config_heart = {
    'pre': {
        'Sex': lambda x: 'male' if x == 'M' else 'female',
        'ChestPainType': lambda x: chest_paint_types_list[x],
        'FastingBS': lambda x: 'yes' if x == 1 else 'no',
        'ExerciseAngina': lambda x: 'yes' if x == 'Y' else 'no',
        'ST_Slope': lambda x: st_slopes[x],
        'RestingECG': lambda x: rest_ecg_results[x]
    }
}

template_heart = 'The Age of the patient is ${Age}. ' \
                 'The Sex of the patient is ${Sex}. ' \
                 'The Chest pain type is ${ChestPainType}. ' \
                 'The Resting blood pressure is ${RestingBP}. ' \
                 'The Serum cholesterol is ${Cholesterol}. ' \
                 'The Fasting blood sugar > 120 mg/dl is ${FastingBS}. ' \
                 'The Resting electrocardiogram results is ${RestingECG}. ' \
                 'The Maximum heart rate achieved is ${MaxHR}. ' \
                 'The Exercise-induced angina is ${ExerciseAngina}. ' \
                 'The ST depression induced by exercise relative to rest is ${Oldpeak}. ' \
                 'The Slope of the peak exercise ST segment is ${ST_Slope}.'