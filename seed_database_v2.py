import random
from datetime import datetime, timedelta, time

from database import SessionLocal
from models import (
    Kunde, Elbil, Ladestation, Ladepunkt, Tarif,
    Opladning, Driftstilstand, Servicehaendelse
)

from simulator import (
    customer_profile,
    generate_session_start,
    seasonal_temperature,
    calculate_energy,
    calculate_duration,
    price_by_time,
    calculate_total_price,
    operational_measurement,
    needs_service
)

db = SessionLocal()
random.seed(42)

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

first_names = ["Johan", "Emma", "Lucas", "Noah", "Ida", "Sofie", "Freja", "Oscar", "Victor", "Sara"]
last_names = ["Jensen", "Nielsen", "Hansen", "Pedersen", "Andersen", "Larsen", "Madsen"]

for i in range(50):
    name = f"{random.choice(first_names)} {random.choice(last_names)}"
    membership = random.choices(
        ["Basic", "Premium", "Business"],
        weights=[0.45, 0.35, 0.20]
    )[0]

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
    "Mercedes": ["EQA", "EQE"]
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

print("Creating stations, charging points and tariffs...")

cities = ["Copenhagen", "Aarhus", "Odense", "Aalborg", "Esbjerg", "Kolding", "Randers", "Vejle", "Roskilde", "Helsingor"]

for i, city in enumerate(cities):
    db.add(Ladestation(
        navn=f"VoltEdge Station {i + 1}",
        by=city,
        adresse=f"Charging Road {i + 1}",
        breddegrad=55 + random.random(),
        laengdegrad=10 + random.random()
    ))

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

for station in stations:
    for navn, tariftype, price in [
        ("Night", "Time of Use", 1.89),
        ("Standard", "Variable", 2.49),
        ("Peak", "Time of Use", 3.49),
        ("Premium", "Subscription", 2.19),
        ("Fleet", "Business", 2.09)
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

for _ in range(1000):
    car = random.choice(cars)
    customer = db.query(Kunde).filter(Kunde.kunde_id == car.kunde_id).first()
    point = random.choice(points)

    profile = customer_profile(customer.medlemskabstype)
    start = generate_session_start(profile)
    temperature = seasonal_temperature(start.month)

    energy = calculate_energy(
        car.batterikapacitet_kwh,
        temperature,
        profile
    )

    duration = calculate_duration(
        energy,
        point.maks_effekt_kw,
        temperature
    )

    price = price_by_time(
        start.hour,
        customer.medlemskabstype
    )

    total = calculate_total_price(
        energy,
        price,
        duration
    )

    end = start + timedelta(minutes=duration)

    db.add(Opladning(
        kunde_id=customer.kunde_id,
        elbil_id=car.elbil_id,
        ladepunkt_id=point.ladepunkt_id,
        tarif_id=random.choice(tariffs).tarif_id,
        energi_kwh=energy,
        varighed_minutter=duration,
        temperatur_c=temperature,
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

print("Creating correlated operational measurements...")

points = db.query(Ladepunkt).all()

for _ in range(500):
    point = random.choice(points)

    measured_at = datetime.now() - timedelta(
        days=random.randint(1, 365),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59)
    )

    measurement = operational_measurement(
        point.maks_effekt_kw,
        point.installationsdato,
        measured_at
    )

    db.add(Driftstilstand(
        ladepunkt_id=point.ladepunkt_id,
        driftsscore=measurement["health"],
        temperatur_c=measurement["temperature"],
        spaending_v=measurement["voltage"],
        stroem_a=measurement["current"],
        fejl_antal=measurement["faults"],
        oppetid_procent=measurement["uptime"],
        maaletidspunkt=measured_at
    ))

db.commit()

print("Creating service events based on operational health...")

drift_records = db.query(Driftstilstand).all()
service_count = 0

for record in drift_records:
    if needs_service(record) and random.random() < 0.65:
        if record.temperatur_c > 38:
            service_type = "Corrective Maintenance"
            cause = "Cooling issue"
        elif record.fejl_antal >= 3:
            service_type = "Inspection"
            cause = "Repeated fault codes"
        elif record.oppetid_procent < 95:
            service_type = "Preventive Maintenance"
            cause = "Low uptime"
        else:
            service_type = "Preventive Maintenance"
            cause = "General degradation"

        db.add(Servicehaendelse(
            ladepunkt_id=record.ladepunkt_id,
            servicetype=service_type,
            aarsag=cause,
            service_dato=record.maaletidspunkt + timedelta(days=random.randint(1, 10)),
            nedetid_timer=round(random.uniform(0.5, 12), 2),
            beskrivelse="Generated service event based on charger health and operational risk."
        ))

        service_count += 1

db.commit()

print()
print("======================================")
print("VoltEdge Simulator v2 database seeded")
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