from pydantic import BaseModel
import numpy as np

class Patient(BaseModel):
    index: str
    level: str 
    age: int
    characteristics: list[float]