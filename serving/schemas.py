from pydantic import BaseModel, Field

class HousingRequest(BaseModel):
    MedInc: float = Field(..., description="Median income")
    HouseAge: float = Field(..., description="Median house age")
    AveRooms: float = Field(..., description="Average number of rooms")
    AveBedrms: float = Field(..., description="Average number of bedrooms")
    Population: float = Field(..., description="Block population")
    AveOccup: float = Field(..., description="Average household occupancy")
    Latitude: float = Field(..., description="Latitude")
    Longitude: float = Field(..., description="Longitude")

class HousingResponse(BaseModel):
    prediction: float
