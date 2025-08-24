import pandas as pd 

# Test pandas functionalities on the csv file
# https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset?resource=download - check this to learn about manipulating heart data

# Feature 1 - Probability of heart disease (I can still use 3 functions in this and be done with it)
# Use pandas to clean data set - remove duplicate data ? remove some columns ?
# Logistic regression
# Probability of heart disease via input

############################################################################################################
# Feature 2 - Check what parameters are not healthy
# Use formulas for the max heart-rate

# Max heart-rate by age formula:
# simple: 220-age
# more accurate Tanaka formula: MHR = 208 - (0.7*age)

# Use pandas categorical data for range ?
############################################################################################################

# Feature 3

print("Test pandas on the heart.csv file")
heart_data = pd.read_csv("heart.csv", sep=",")

print("Patients with max heart rate above the recommended")
critical_hr = heart_data.loc[heart_data["thalach"]>208-(0.7*heart_data["age"])]
print(critical_hr.loc[:, ["age", "thalach", "thal"]])
print("\n % of patients with max heart rate above rec")
print(len(critical_hr.index)/len(heart_data.index) * 100, "%")
print("\n % of patients with max heart rate above rec and a heart problem")
critical_hr2 = critical_hr.loc[critical_hr["thal"]!=0]
print(len(critical_hr2.index)/len(critical_hr.index)*100, "%")

print(len(heart_data.index))

print(len(heart_data.index), len(set(heart_data.index)))

a = [1,2,34,3,3,5,7,8,2,3,9,34]
print(len(a), len(set(a)))