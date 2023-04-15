from fastapi import FastAPI
from point6.LogisticRegressionModelJIM import LogisticRegressionModelJIM
import numpy as np

app = FastAPI()
classModel = LogisticRegressionModelJIM()

@app.get("/")
async def root():
    return {"message": "Hola mundo"}

# Realiza la inferencia
@app.post("/train/")
async def train():
    x_train, x_test, y_train, y_test =classModel.get_nmist_data_splited(split_size=0.75)
    x_train_standarized,y_train_standarized = classModel.data_standarization(x_train,y_train)
    result = classModel.train_model(x_train_standarized,y_train_standarized)
    return result

# Realiza la inferencia
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image_array = np.frombuffer(image_bytes, dtype=np.uint8).reshape((1, -1))
    image_array = image_array / 255.0
    prediction = classModel.predict(image_array)

    return {"prediction": int(prediction[0])}