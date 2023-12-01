import pickle
import json
import numpy as np
# from config import MODEL_FILE_PATH, JSON_FILE_PATH


class MedicalInsurance():

    def __init__(self, age, sex, bmi, children, smoker, region):
        self.age =  age
        self.sex = sex
        self.bmi = bmi
        self.children = children
        self.smoker = smoker
        self.region = 'region_' + region

    def load_model(self):
        with open(r'project_app\Linear_model.pkl', 'rb') as f:
            self.model = pickle.load(f)

        with open(r'project_app\project_data.json','r') as f:
            self.project_data = json.load(f)

    def get_predicted_charges(self):
        self.load_model()

        test_array = np.zeros(len(self.project_data['columns']))
        test_array[0] = self.age
        test_array[1] = self.project_data['sex'][self.sex]
        test_array[2] = self.bmi
        test_array[3] = self.children
        test_array[4] = self.project_data['smoker'][self.smoker]

        region_index = self.project_data['columns'].index(self.region)
        test_array[region_index] = 1
        print('Test Array :',test_array)

        predicted_charges = self.model.predict([test_array])
        print(f'Charges for Medical Insurance are : Rs. {round(predicted_charges[0],2)}') 
        return predicted_charges

 

if __name__=='__main__':
    age = 24
    sex = 'male'
    bmi = 26.5
    children = 2
    smoker = 'no'
    region = 'southwest'
    med_ins= MedicalInsurance(age, sex, bmi, children, smoker, region)
    med_ins.get_predicted_charges()

# print(config.JSON_FILE_PATH)










