import random
from datetime import datetime, timedelta, time

from database import SessionLocal
from models import (
    Kunde,
    Elbil,
    Ladestation,
    Ladepunkt,
    Tarif,
    Opladning,
    Driftstilstand,
    Servicehaendelse,
)

db = SessionLocal()

random.seed(42)


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def seasonal_temperature(month):
    if month in [12, 1, 2]:
        return round(random.uniform(-8, 6), 1)
    if month in [3, 4, 5]:
        return round(random.uniform(5, 17), 1)
    if month in [6, 7, 8]:
        return round(random.uniform(16, 32), 1)
    return round(random.uniform(5, 18), 1)


def price_by_time(hour, membership):
    if 0 <= hour < 6:
        price = 1.89
    elif 6 <= hour < 16:
        price = 2.49
    elif 16 <= hour < 20:
        price = 3.49
    else:
        price = 2.29

    if membership == "Premium":
        price *= 0.90
    elif membership == "Business":
        price *= 0.85

    return round(price, 2)


print("Deleting old data...")

db.query(Servicehaendelse).delete()
db.query(Driftstilstand).delete()
db.query(Opladning).delete()
db.query(Tarif).delete()
db.query(Ladepunkt).delete()
db.query(Ladestation).delete()
db.query(Elbil).delete()
db.query(Kunde).delete()
db.commit()

print("Creating customers...")

profiles = ["Occasional", "Commuter", "Heavy", "Business"]

first_names = ["Johan", "Emma", "Lucas", "Noah", "Ida", "Sofie", "Freja", "Oscar", "Victor", "Sara"]
last_names = ["Jensen", "Nielsen", "Hansen", "Pedersen", "Andersen", "Larsen", "Madsen"]

for i in range(50):
    name = f"{random.choice(first_names)} {random.choice(last_names)}"
    profile = random.choice(profiles)

    if profile == "Business":
        membership = "Business"
    elif profile == "Heavy":
        membership = "Premium"
    else:
        membership = random.choice(["Basic", "Premium"])

    db.add(Kunde(
        navn=name,
        email=name.lower().replace(" ", ".") + f"{i}@mail.com",
        telefon="20" + str(random.randint(100000, 999999)),
        medlemskabstype=membership,
        oprettet_dato=datetime.now() - timedelta(days=random.randint(30, 900))
    ))

db.commit()

print("Creating EVs...")

brands = {
    "Tesla": ["Model 3", "Model Y"],
    "Volkswagen": ["ID.3", "ID.4"],
    "BMW": ["i4", "iX"],
    "Hyundai": ["IONIQ 5", "Kona"],
    "Mercedes": ["EQA", "EQE"],
}

customers = db.query(Kunde).all()

for i in range(75):
    brand = random.choice(list(brands.keys()))
    db.add(Elbil(
        kunde_id=random.choice(customers).kunde_id,
        maerke=brand,
        model=random.choice(brands[brand]),
        batterikapacitet_kwh=random.choice([50, 60, 70, 75, 82, 90, 100]),
        produktionsaar=random.randint(2019, 2025)
    ))

db.commit()

print("Creating stations and charging points...")

cities = ["Copenhagen", "Aarhus", "Odense", "Aalborg", "Esbjerg", "Kolding", "Randers", "Vejle", "Roskilde", "Helsingor"]

for i, city in enumerate(cities):
    station = Ladestation(
        navn=f"VoltEdge Station {i + 1}",
        by=city,
        adresse=f"Charging Road {i + 1}",
        breddegrad=55 + random.random(),
        laengdegrad=10 + random.random()
    )
    db.add(station)

db.commit()

stations = db.query(Ladestation).all()

for station in stations:
    for _ in range(4):
        db.add(Ladepunkt(
            ladestation_id=station.ladestation_id,
            ladetype=random.choice(["AC", "DC Fast", "Ultra Fast"]),
            maks_effekt_kw=random.choice([11, 22, 50, 150, 300]),
            installationsdato=datetime.now().date() - timedelta(days=random.randint(100, 1200)),
            status="Available"
        ))

db.commit()

print("Creating tariffs...")

for station in stations:
    for navn, tariftype, price in [
        ("Night", "Time of Use", 1.89),
        ("Standard", "Variable", 2.49),
        ("Peak", "Time of Use", 3.49),
        ("Premium", "Subscription", 2.19),
        ("Fleet", "Business", 2.09),
    ]:
        db.add(Tarif(
            ladestation_id=station.ladestation_id,
            navn=navn,
            tariftype=tariftype,
            pris_per_kwh=price,
            start_tid=time(0, 0),
            slut_tid=time(23, 59)
        ))

db.commit()

print("Creating correlated charging sessions...")

customers = db.query(Kunde).all()
cars = db.query(Elbil).all()
points = db.query(Ladepunkt).all()
tariffs = db.query(Tarif).all()

for i in range(1000):
    car = random.choice(cars)
    customer = next(c for c in customers if c.kunde_id == car.kunde_id)
    point = random.choice(points)

    start = datetime.now() - timedelta(
        days=random.randint(1, 365),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59)
    )

    temp = seasonal_temperature(start.month)

    base_energy = random.uniform(0.25, 0.85) * car.batterikapacitet_kwh

    if temp < 0:
        energy = base_energy * random.uniform(1.05, 1.20)
    elif temp > 28:
        energy = base_energy * random.uniform(0.90, 1.00)
    else:
        energy = base_energy

    energy = round(clamp(energy, 5, car.batterikapacitet_kwh), 2)

    efficiency_factor = 1.0
    if temp < 0:
        efficiency_factor = 1.20
    elif temp > 28:
        efficiency_factor = 1.10

    duration = int((energy / point.maks_effekt_kw) * 60 * efficiency_factor)
    duration += random.randint(5, 25)
    duration = clamp(duration, 15, 360)

    price = price_by_time(start.hour, customer.medlemskabstype)

    idle_fee = 0
    if duration > 180:
        idle_fee = random.uniform(5, 25)

    noise = random.uniform(-3, 3)
    total = round((energy * price) + idle_fee + noise, 2)
    total = max(total, 5)

    end = start + timedelta(minutes=duration)

    db.add(Opladning(
        kunde_id=customer.kunde_id,
        elbil_id=car.elbil_id,
        ladepunkt_id=point.ladepunkt_id,
        tarif_id=random.choice(tariffs).tarif_id,
        energi_kwh=energy,
        varighed_minutter=duration,
        temperatur_c=temp,
        pris_per_kwh=price,
        samlet_pris=total,
        starttid=start,
        sluttid=end,
        status=random.choices(
            ["Completed", "Interrupted", "Failed"],
            weights=[0.92, 0.06, 0.02]
        )[0]
    ))

db.commit()

print("Creating correlated operational health data...")

points = db.query(Ladepunkt).all()
latest_health = {}

for i in range(500):
    point = random.choice(points)

    measured_at = datetime.now() - timedelta(
        days=random.randint(1, 365),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59)
    )

    temp = seasonal_temperature(measured_at.month) + random.uniform(0, 12)
    current = random.uniform(20, point.maks_effekt_kw * 1.2)
    voltage = random.uniform(385, 420)

    fault_probability = 0.02
    if temp > 35:
        fault_probability += 0.18
    if current > 180:
        fault_probability += 0.10
    if point.maks_effekt_kw >= 150:
        fault_probability += 0.05

    faults = sum(1 for _ in range(5) if random.random() < fault_probability)

    uptime = 99.8 - faults * random.uniform(0.5, 1.5) - max(temp - 35, 0) * 0.15
    uptime = round(clamp(uptime, 80, 100), 2)

    health = (
        100
        - faults * 8
        - max(temp - 25, 0) * 0.7
        - max(current - 180, 0) * 0.04
        - (100 - uptime) * 1.2
        + random.uniform(-3, 3)
    )

    health = round(clamp(health, 35, 100), 2)
    latest_health[point.ladepunkt_id] = health

    db.add(Driftstilstand(
        ladepunkt_id=point.ladepunkt_id,
        driftsscore=health,
        temperatur_c=round(temp, 1),
        spaending_v=round(voltage, 1),
        stroem_a=round(current, 1),
        fejl_antal=faults,
        oppetid_procent=uptime,
        maaletidspunkt=measured_at
    ))

db.commit()

print("Creating service events based on health problems...")

drift_records = db.query(Driftstilstand).all()
service_count = 0

for record in drift_records:
    should_service = (
        record.driftsscore < 70
        or record.fejl_antal >= 3
        or record.oppetid_procent < 95
        or record.temperatur_c > 38
    )

    if should_service and random.random() < 0.65:
        if record.temperatur_c > 38:
            cause = "Cooling issue"
            service_type = "Corrective Maintenance"
        elif record.fejl_antal >= 3:
            cause = "Repeated fault codes"
            service_type = "Inspection"
        elif record.oppetid_procent < 95:
            cause = "Low uptime"
            service_type = "Preventive Maintenance"
        else:
            cause = "General degradation"
            service_type = "Preventive Maintenance"

        db.add(Servicehaendelse(
            ladepunkt_id=record.ladepunkt_id,
            servicetype=service_type,
            aarsag=cause,
            service_dato=record.maaletidspunkt + timedelta(days=random.randint(1, 10)),
            nedetid_timer=round(random.uniform(0.5, 12), 2),
            beskrivelse="Generated service event based on operational health metrics."
        ))

        service_count += 1

db.commit()

print()
print("======================================")
print("VoltEdge correlated database seeded!")
print("======================================")
print(f"Kunder: {db.query(Kunde).count()}")
print(f"Elbiler: {db.query(Elbil).count()}")
print(f"Ladestationer: {db.query(Ladestation).count()}")
print(f"Ladepunkter: {db.query(Ladepunkt).count()}")
print(f"Tariffer: {db.query(Tarif).count()}")
print(f"Opladninger: {db.query(Opladning).count()}")
print(f"Driftstilstande: {db.query(Driftstilstand).count()}")
print(f"Servicehaendelser: {db.query(Servicehaendelse).count()}")

db.close()