import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configuración de CORS
origins = [
    "http://localhost",
    "http://localhost:4200",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from sklearn.preprocessing import MinMaxScaler
from kdtree import KDTreeDS, KnnQuery, BallQuery
from patient import Patient
import os

app=FastAPI()

origins = [
    "http://localhost",
    "http://localhost:4200",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

df = pd.read_csv('data.csv')

index = df['index'].astype(str)
levels = df['Level']
age = df['Age']

features = df[['Air Pollution', 'Alcohol use', 'Dust Allergy', 'OccuPational Hazards', 'Genetic Risk',
               'chronic Lung Disease', 'Balanced Diet' ,'Obesity', 'Smoking', 'Passive Smoker', 'Chest Pain', 'Coughing of Blood',
               'Fatigue', 'Weight Loss', 'Shortness of Breath', 'Wheezing', 'Swallowing Difficulty',
               'Clubbing of Finger Nails', 'Frequent Cold', 'Dry Cough', 'Snoring']]


scaler = MinMaxScaler(feature_range=(1,9))

data = features.to_numpy()

scaled_data = scaler.fit_transform(data)


#Patience is a list of Patient objects
patients: list[Patient] = []

for i in range (len(scaled_data)):
    patients.append(Patient(index = index[i], level = levels[i], age = age[i], characteristics = scaled_data[i]))


tree: KDTreeDS = KDTreeDS(scaled_data)


query_point = ['1000',47,6,4,1,5,4,4,3,2,3,5,7,7,5,3,2,7,8,2,4,6,3,'high']
level:str = query_point.pop()
age:int = query_point.pop(1)
new_index:str = query_point.pop(0)

new_point = scaler.transform([query_point])


@app.post("/knn_query")
async def knn_query(query: KnnQuery):
    print(query)
    query_point = scaler.transform([query.new_point])
    indices, distances = tree.knn_query(query_point, query.k)
    similar_patients_json = []
    for i in indices:
        similar_patients_json.append(patients[i].model_dump_json())
    return similar_patients_json

@app.post("/ball_query")
async def ball_query(query: BallQuery):
    query_point = scaler.transform([query.new_point])
    indices = tree.ball_query(query_point, query.radius)
    similar_patients_json = []
    for i in indices:
        similar_patients_json.append(patients[i].model_dump_json())
    return similar_patients_json
   
@app.get("/patients")
async def get_patients():
    patients_json = []
    for patient in patients:
        patients_json.append(patient.model_dump_json())
    return patients_json
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port= int(os.environ.get("PORT", 5000)))