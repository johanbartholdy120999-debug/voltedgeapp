import random
from datetime import datetime, timedelta


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


def customer_profile(membership):
    if membership == "Business":
        return "Business"
    if membership == "Premium":
        return random.choice(["Heavy", "Commuter"])
    return random.choice(["Occasional", "Weekend"])


def generate_session_start(profile):
    days_back = random.randint(1, 365)
    base = datetime.now() - timedelta(days=days_back)

    if profile == "Business":
        hour = random.randint(7, 17)
    elif profile == "Commuter":
        hour = random.choice([7, 8, 16, 17, 18])
    elif profile == "Weekend":
        hour = random.randint(10, 18)
    else:
        hour = random.randint(8, 22)

    return base.replace(
        hour=hour,
        minute=random.randint(0, 59),
        second=0,
        microsecond=0
    )


def calculate_energy(battery_capacity, temperature, profile):
    if profile == "Heavy":
        base = random.uniform(0.55, 0.90)
    elif profile == "Business":
        base = random.uniform(0.45, 0.85)
    elif profile == "Commuter":
        base = random.uniform(0.25, 0.60)
    else:
        base = random.uniform(0.15, 0.50)

    energy = battery_capacity * base

    if temperature < 0:
        energy *= random.uniform(1.05, 1.20)
    elif temperature > 28:
        energy *= random.uniform(0.90, 1.00)

    return round(clamp(energy, 5, battery_capacity), 2)


def calculate_duration(energy_kwh, charger_power_kw, temperature):
    curve_factor = 1.0

    if energy_kwh > 60:
        curve_factor += 0.25

    if temperature < 0:
        curve_factor += 0.20

    if temperature > 30:
        curve_factor += 0.10

    duration = int((energy_kwh / charger_power_kw) * 60 * curve_factor)
    duration += random.randint(5, 30)

    return int(clamp(duration, 15, 360))


def calculate_total_price(energy_kwh, price_per_kwh, duration_minutes):
    idle_fee = 0

    if duration_minutes > 180:
        idle_fee = random.uniform(5, 25)

    noise = random.uniform(-3, 3)

    total = energy_kwh * price_per_kwh + idle_fee + noise

    return round(max(total, 5), 2)


def operational_measurement(charger_power_kw, installation_date, measured_at):
    age_days = (measured_at.date() - installation_date).days
    age_years = max(age_days / 365, 0)

    ambient_temp = seasonal_temperature(measured_at.month)
    load_factor = random.uniform(0.2, 1.1)

    current = random.uniform(20, charger_power_kw * 1.2)
    voltage = random.uniform(385, 420)

    internal_temp = ambient_temp + load_factor * 12 + age_years * 0.8

    fault_probability = 0.02
    fault_probability += max(internal_temp - 35, 0) * 0.01
    fault_probability += max(current - 180, 0) * 0.001
    fault_probability += age_years * 0.015

    faults = sum(1 for _ in range(5) if random.random() < fault_probability)

    uptime = (
        99.8
        - faults * random.uniform(0.5, 1.8)
        - max(internal_temp - 35, 0) * 0.15
        - age_years * 0.4
    )

    uptime = round(clamp(uptime, 75, 100), 2)

    health = (
        100
        - age_years * 2.5
        - faults * 8
        - max(internal_temp - 25, 0) * 0.7
        - max(current - 180, 0) * 0.04
        - (100 - uptime) * 1.2
        + random.uniform(-3, 3)
    )

    health = round(clamp(health, 30, 100), 2)

    return {
        "temperature": round(internal_temp, 1),
        "voltage": round(voltage, 1),
        "current": round(current, 1),
        "faults": faults,
        "uptime": uptime,
        "health": health
    }


def needs_service(record):
    return (
        record.driftsscore < 70
        or record.fejl_antal >= 3
        or record.oppetid_procent < 95
        or record.temperatur_c > 38
    )