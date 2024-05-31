from fastapi import FastAPI
from uvicorn import run as app_run
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from train import training as start_training
from src.name_entity_recognition.pipeline.prediction_pipeline import ModelPredictor
from src.name_entity_recognition.constants import APP_HOST, APP_PORT
from src.name_entity_recognition.logger import logging


NameEntityRecognitionApp = FastAPI()

origins = ["*"]

NameEntityRecognitionApp.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@NameEntityRecognitionApp.get("/train")
async def training():
    try:
        start_training()

        return Response("Training successful !!")

    except Exception as error:
        logging.error(error)
        raise Response(f"Error Occurred! {error}")


@NameEntityRecognitionApp.post("/predict")
async def predict_route(text: str):
    try:
        prediction_pipeline = ModelPredictor()

        sentence, labels = prediction_pipeline.initiate_model_predictor(sentence=text)

        return sentence, labels

    except Exception as error:
        logging.error(error)
        return Response(f"Error Occurred! {error}")


if __name__ == "__main__":
    app_run(NameEntityRecognitionApp, host=APP_HOST, port=APP_PORT)
