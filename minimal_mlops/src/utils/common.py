from enum import Enum

class Day_of_Week_Enum(Enum):
    weekend: str = "Weekend"
    weekday: str = "Weekday"
    
class Time_of_Day_Enum(Enum):
    morning: str = "Morning"
    afternoon: str = "Afternoon"
    evening: str = "Evening"
    night: str = "Night"
    
class Weather_Enum(Enum):
    snow: str = "Snow"
    rain: str = "Rain"
    clear: str = "Clear"
    
class Traffic_Conditions_Enum(Enum):
    high: str = "High"
    low: str = "Low"
    medium: str = "Medium"
    
