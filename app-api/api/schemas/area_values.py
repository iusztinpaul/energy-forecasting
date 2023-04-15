from typing import Any, List

from pydantic import BaseModel


class UniqueArea(BaseModel):
    values: List[int]
