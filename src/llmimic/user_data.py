from dataclasses import dataclass
from typing import Optional

@dataclass
class UserData:
    name: str
    birthday: str
    sex: str
    race: str
    details: Optional[str] = "No additional details provided."

    def __post_init__(self):
        valid_sexes = {"Male", "Female", "Non-binary"}
        if self.sex not in valid_sexes:
            raise ValueError(f"Invalid sex: {self.sex}. Must be one of {valid_sexes}")

        valid_races = {"Caucasian", "Asian", "Black or African American", "Hispanic or Latino", "Other"}
        if self.race not in valid_races:
            raise ValueError(f"Invalid race: {self.race}. Must be one of {valid_races}")