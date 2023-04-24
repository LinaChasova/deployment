import os
import pickle
import pandas as pd
import logging
from sklearn.preprocessing import LabelEncoder
import uvicorn

from typing import Optional, Union
from fastapi import FastAPI, Header
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from src.entities import (
    TransactionModel,
    TransactionResponse,
    FeatureParams,
    SERVICE_NAME,
    SERVICE_VERSION,
)


app = FastAPI(title=SERVICE_NAME)
feature_params = FeatureParams()
model: Optional[Pipeline] = None
transformer: LabelEncoder

logger = logging.getLogger(__name__)


def load_object(path: str) -> Pipeline:
    with open(path, 'rb') as file:
        return pickle.load(file)


async def make_prediction(
        pipeline: Union[Pipeline, RandomForestClassifier], data: pd.DataFrame
) -> float:
    return pipeline.predict_proba(data)[0, 1]


@app.get('/')
def main():
    return 'This is the entry point of our predictor.'


@app.on_event('startup')
def load_model():
    model_path = os.getenv('PATH_TO_MODEL', default='models/model.pkl')
    if model_path is None:
        err = 'Environment variable PATH_TO_MODEL is None.'
        logger.error(msg=f'{err}',
                     extra=Exception(
                         ExMessage='Path to model is None.',
                         ServiceVersion=SERVICE_VERSION,
                     ))
        raise RuntimeError(err)
    global model
    transformer, model = load_object(model_path)

    print('ML service started')
    logger.info('ML service started')


@app.get('/health')
def health() -> bool:
    return model is not None


@app.get('/predict', response_model=TransactionResponse)
async def predict(transaction: TransactionModel, x_request_id: str = Header(...)):
    transaction_dict = transaction.dict(by_alias=True)
    features = pd.DataFrame.from_dict(transaction_dict, orient='index').T
    
    print(f'New request {transaction_dict}.')
    print(features)

    if features.empty:
        risk_rating = .0
        is_risk_calculated = False
        print('Request couldn\'t be completed')
        logger.info(msg='Request couldn\'t be completed')
    else:
        print('Making prediction...')
        logger.info(msg='Making prediction...')
        risk_rating = await make_prediction(model, features)
        is_risk_calculated = True
        print('RiskRating: {risk_rating}.')
        logger.info(msg=f'RiskRating: {risk_rating}.')

    return TransactionResponse(
        risk_rating=risk_rating,
        is_risk_calculated=is_risk_calculated
    )


if __name__ == '__main__':
    uvicorn.run('app:app', host='0.0.0.0', port=os.getenv('PORT', 8000))
