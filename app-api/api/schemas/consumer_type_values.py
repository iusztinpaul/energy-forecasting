from typing import Any, List

from pydantic import BaseModel


class UniqueConsumerType(BaseModel):
    values: List[int]
