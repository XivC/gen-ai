from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

CITIES = {
    "Москва",
    "Санкт-Петербург",
    "Новосибирск",
    "Екатеринбург",
    "Казань",
    "Благовещенск",
    "Самара",
    "Краснодар",
    "Владивосток",
    "Хабаровск",
}


class Address(BaseModel):
    city: str
    district: str = Field(min_length=2, max_length=40)

    @field_validator("city")
    @classmethod
    def city_must_be_in_list(cls, value: str) -> str:
        if value not in CITIES:
            raise ValueError(f"Город '{value}' не поддерживается")
        return value


class Application(BaseModel):
    full_name: str
    age: int = Field(ge=22, le=65)
    address: Address
    speciality: Literal[
        "программист",
        "дворник",
        "басист",
        "инженер",
        "юрист",
        "егерь",
        "повар",
        "менеджер",
    ]
    desired_course: Literal[
        "Введение в Python",
        "Diplôme de Cuisine",
        "Лингвистика",
        "Кибербезопасность",
        "Финансовый менеджмент",
        "Выживание в тайге",
    ]
    years_of_experience: int = Field(ge=0, le=40)
    graduation_year: int = Field(ge=1980, le=2024)

    @model_validator(mode="after")
    def graduation_consistent_with_age(self) -> "Application":
        current_year = datetime.now().year
        age_at_graduation = self.graduation_year - (current_year - self.age)
        if not (22 <= age_at_graduation <= self.age):
            raise ValueError(
                f"Год окончания {self.graduation_year} не совпадает с возрастом {self.age}: "
                f"На момент окончания возраст должен быть равен {age_at_graduation}"
            )
        return self

    @property
    def city(self) -> str:
        return self.address.city
