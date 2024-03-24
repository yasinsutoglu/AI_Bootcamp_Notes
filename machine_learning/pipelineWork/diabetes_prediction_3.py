################################################
# End-to-End Diabetes Machine Learning Pipeline III
################################################

import joblib
import pandas as pd

# yeni veri girişlerinin olduğu DB'imiz
df = pd.read_csv("datasets/diabetes.csv")

random_user = df.sample(1, random_state=45)

new_model = joblib.load("voting_clf.pkl")

new_model.predict(random_user) # yeni veri girişi sonrası hata alabiliriz

# Yeni veri gelmesi sonrası tekrar data-preprocessing ihtiyacı olur
from AI_Bootcamp_Sections.machine_learning.pipelineWork.diabetes_pipeline_2 import diabetes_data_prep

X, y = diabetes_data_prep(df)

random_user = X.sample(1, random_state=50)

new_model = joblib.load("voting_clf.pkl")

new_model.predict(random_user)
