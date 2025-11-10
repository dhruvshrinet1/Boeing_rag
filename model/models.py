from pydantic import BaseModel, RootModel
from typing import List, Union
from enum import Enum

class Metadata(BaseModel):

    summary : List[str]
    