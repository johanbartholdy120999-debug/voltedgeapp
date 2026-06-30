from sqlalchemy import Column, Integer, String, Float, DateTime, Date, Time, ForeignKey, Text
from datetime import datetime

from database import Base


class Kunde(Base):
    __tablename__ = "kunder"

    kunde_id = Column(Integer, primary_key=True, index=True)
    navn = Column(String(100), nullable=False)
    email = Column(String(255), nullable=False)
    telefon = Column(String(20))
    medlemskabstype = Column(String(50))
    oprettet_dato = Column(DateTime, default=datetime.utcnow)


class Elbil(Base):
    __tablename__ = "elbiler"

    elbil_id = Column(Integer, primary_key=True, index=True)
    kunde_id = Column(Integer, ForeignKey("kunder.kunde_id"), nullable=False)
    maerke = Column(String(50))
    model = Column(String(100))
    batterikapacitet_kwh = Column(Float)
    produktionsaar = Column(Integer)


class Ladestation(Base):
    __tablename__ = "ladestationer"

    ladestation_id = Column(Integer, primary_key=True, index=True)
    navn = Column(String(100), nullable=False)
    by = Column(String(100))
    adresse = Column(String(255))
    breddegrad = Column(Float)
    laengdegrad = Column(Float)


class Ladepunkt(Base):
    __tablename__ = "ladepunkter"

    ladepunkt_id = Column(Integer, primary_key=True, index=True)
    ladestation_id = Column(Integer, ForeignKey("ladestationer.ladestation_id"), nullable=False)
    ladetype = Column(String(50))
    maks_effekt_kw = Column(Float)
    installationsdato = Column(Date)
    status = Column(String(50))


class Tarif(Base):
    __tablename__ = "tariffer"

    tarif_id = Column(Integer, primary_key=True, index=True)
    ladestation_id = Column(Integer, ForeignKey("ladestationer.ladestation_id"), nullable=False)
    navn = Column(String(100))
    tariftype = Column(String(50))
    pris_per_kwh = Column(Float)
    start_tid = Column(Time)
    slut_tid = Column(Time)


class Opladning(Base):
    __tablename__ = "opladninger"

    opladning_id = Column(Integer, primary_key=True, index=True)
    kunde_id = Column(Integer, ForeignKey("kunder.kunde_id"), nullable=False)
    elbil_id = Column(Integer, ForeignKey("elbiler.elbil_id"), nullable=False)
    ladepunkt_id = Column(Integer, ForeignKey("ladepunkter.ladepunkt_id"), nullable=False)
    tarif_id = Column(Integer, ForeignKey("tariffer.tarif_id"), nullable=False)

    energi_kwh = Column(Float, nullable=False)
    varighed_minutter = Column(Integer)
    temperatur_c = Column(Float)
    pris_per_kwh = Column(Float)
    samlet_pris = Column(Float)
    starttid = Column(DateTime)
    sluttid = Column(DateTime)
    status = Column(String(50))


class Servicehaendelse(Base):
    __tablename__ = "servicehaendelser"

    servicehaendelse_id = Column(Integer, primary_key=True, index=True)
    ladepunkt_id = Column(Integer, ForeignKey("ladepunkter.ladepunkt_id"), nullable=False)
    servicetype = Column(String(100))
    aarsag = Column(String(255))
    service_dato = Column(DateTime)
    nedetid_timer = Column(Float)
    beskrivelse = Column(Text)


class Driftstilstand(Base):
    __tablename__ = "driftstilstande"

    driftstilstand_id = Column(Integer, primary_key=True, index=True)
    ladepunkt_id = Column(Integer, ForeignKey("ladepunkter.ladepunkt_id"), nullable=False)
    driftsscore = Column(Float)
    temperatur_c = Column(Float)
    spaending_v = Column(Float)
    stroem_a = Column(Float)
    fejl_antal = Column(Integer)
    oppetid_procent = Column(Float)
    maaletidspunkt = Column(DateTime, default=datetime.utcnow)