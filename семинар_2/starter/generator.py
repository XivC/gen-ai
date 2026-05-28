import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import pandas as pd

from llm_client import get_model, make_client
from schema import CITIES, Application

client = make_client()
MODEL = get_model()

PER_CITY = 5
MAX_WORKERS = 5

SPECIALITIES = [
    "программист",
    "дворник",
    "басист",
    "инженер",
    "юрист",
    "егерь",
    "повар",
    "менеджер",
]
COURSES = [
    "Введение в Python",
    "Diplôme de Cuisine",
    "Лингвистика",
    "Кибербезопасность",
    "Финансовый менеджмент",
    "Выживание в тайге",
]

new_line = "\n"  # uv собрал окружение под python 3.11, не стал менять. Пусть будет так)
SYSTEM_PROMPT = f"""
Ты генерируешь синтетические заявки на курсы повышения
квалификации (ДПО) в России. Создай правдоподобного человека: ФИО, возраст,
адрес (город и район), текущую специальность, желаемый курс, стаж работы и
год окончания вуза.

Список специальностей (выбрать из списка):
{new_line.join(f"{num}. {spec}" for num, spec in enumerate(SPECIALITIES))}

Список курсов (выбрать из списка):
{new_line.join(f"{num}. {spec}" for num, spec in enumerate(COURSES))}

Ограничения:
1. Возраст 22-65
2. Стаж 0-40
3. Год окончания вуза 1980-2024
4. Следи, чтобы возраст и год окончания не противоречили: на момент окончания
вуза человеку должно быть не меньше 22 лет, то есть graduation_year не позже
чем (2026 - age + 22)

Текущий год: 2026
"""


def build_user_prompt(seed_city: str, seed_speciality: str) -> str:
    return (
        f"Создай одну заявку. Город проживания: {seed_city}"
        f"Текущая специальность заявителя: {seed_speciality}"
    )


def generate_one(seed_city: str) -> Application:
    seed_speciality = random.choice(SPECIALITIES)
    return client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(seed_city, seed_speciality)},
        ],
        response_model=Application,
        max_retries=3,
        temperature=0.9,
    )


def run() -> list[Application]:
    tasks = [city for city in CITIES for _ in range(PER_CITY)]
    total_tasks = len(tasks)

    apps: list[Application] = []
    done = 0
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(generate_one, city): city for city in tasks}
        for fut in as_completed(futures):
            city = futures[fut]
            done += 1
            try:
                app = fut.result()
                apps.append(app)
                print(f"VALID[{done}/{total_tasks}]  {city} | {app.full_name} | {app.speciality}")
            except Exception as exc:
                print(f"FAILED[{done}/{total_tasks}]  {city} | {exc}")

    work_time = time.time() - start_time
    print(f"Готово за {work_time:.1f}")
    print(f"{len(apps)}/{len(tasks)} валидных")
    return apps


def to_dataframe(apps: list[Application]) -> pd.DataFrame:
    rows = []
    for a in apps:
        d = a.model_dump()
        addr = d.pop("address")
        d["city"] = addr["city"]
        d["district"] = addr["district"]
        rows.append(d)
    return pd.DataFrame(rows)


def plot_bar(series: pd.Series, title: str, out: str):
    counts = series.value_counts()
    plt.figure(figsize=(8, 4))
    plt.title(title)
    plt.ylabel("Число заявок")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return counts

def main():
    apps = run()
    if not apps:
        print("Нет валидных заявок")
        return

    df = to_dataframe(apps)
    df.to_csv("applications.csv", index=False, encoding="utf-8-sig")

    cities = plot_bar(df["city"], "Распределение по городам", "cities.png")
    specs = plot_bar(df["speciality"], "Распределение по специальностям", "specialities.png")

    total = len(df)
    top_city_percent = round(cities.iloc[0] / total * 100, 2)
    top_speciality_percent = round(specs.iloc[0] / total * 100, 2)
    print(f"Самый частотный город: {top_city_percent}%")
    print(f"Самая частотная специальность: {top_speciality_percent}%")


if __name__ == "__main__":
    main()
