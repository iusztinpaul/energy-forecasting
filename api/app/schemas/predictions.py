from typing import List, Any

from pydantic import BaseModel 

class PredictionResults(BaseModel): 
    datetime_utc: List[Any]
    area: List[Any]
    consumer_type: List[Any]
    energy_consumption: List[Any]
    preds_datetime_utc: List[Any] 
    preds_area: List[Any]
    preds_consumer_type: List[Any]
    preds_energy_consumption: List[Any]