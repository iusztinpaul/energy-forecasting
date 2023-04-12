from typing import List, Any

from pydantic import BaseModel 

class PredictionResults(BaseModel): 
    datetime_utc: List[Any]
    energy_consumption: List[Any]
    preds_datetime_utc: List[Any]
    preds_energy_consumption: List[Any]