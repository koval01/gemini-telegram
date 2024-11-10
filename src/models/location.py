from typing import Optional

from pydantic import BaseModel, Field, AliasChoices, field_validator, model_validator
from pydantic_core import PydanticCustomError


class Address(BaseModel):
    road: Optional[str] = None
    village: Optional[str] = None
    city: Optional[str] = None
    state_district: Optional[str] = None
    city_district: Optional[str] = None
    state: Optional[str] = None
    house_number: Optional[str] = None
    neighbourhood: Optional[str] = None
    postcode: str
    country: str


class Location(BaseModel):
    osm_type: str
    lat: float
    lon: float
    category: Optional[str] = None
    type: str
    address_type: str = Field(validation_alias=AliasChoices('addresstype', 'address_type'))
    display_name: str
    name: Optional[str] = None
    address: Address

    @model_validator(mode='before')
    def replace_empty_strings(cls, values):
        # Iterate over all fields and replace empty strings with None
        return {
            key: None if isinstance(value, str) and len(value) == 0
            else value
            for key, value in values.items()
        }

    @field_validator('lat', 'lon', mode='before')
    def parse_float(cls, value):
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                raise PydanticCustomError(
                    'float_parsing',
                    f"Error converting str to a float"
                )
        return value
