########################################################################################################################
# stroke
########################################################################################################################
# Used descriptions from: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data

work_type_lists = {
    'children': 'is a child',
    'Govt_job': 'has a government job',
    'Never_worked': 'never worked at a job',
    'Private': 'has a private job',
    'Self-employed': 'is self employed'
}
smoking_status_lists = {
    'formerly smoked': 'has formerly smoked',
    'never smoked': 'has never smoked',
    'smokes': 'smokes',
    'Unknown': 'smoking status is not available',
}

template_config_stroke = {
    'pre': {
        'hypertension': lambda x: 'have' if x == 1 else 'does not have',
        'heart_disease': lambda x: 'have' if x == 1 else 'does not have',
        'ever_married': lambda x: 'has married' if x == "Yes" else 'never married',
        'work_type': lambda x: work_type_lists[x],
        'Residence_type': lambda x: 'rural' if x == "Rural" else 'urban',
        'smoking_status': lambda x: smoking_status_lists[x],
    }
}

# doesn\'t have, have
template_stroke = 'The Age of the patient is ${age}. ' \
                 'The Gender of the patient is ${gender}. ' \
                 'The patient ${hypertension} hypertension. ' \
                 'The patient ${heart_disease} heart disease. ' \
                 'The patient ${ever_married}. ' \
                 'The patient ${work_type}. ' \
                 'The patient resides in a ${Residence_type} area. ' \
                 'The average glucose level in blood is ${avg_glucose_level}. ' \
                 'The body mass index is ${bmi}. ' \
                 'The patient ${smoking_status}. ' \

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