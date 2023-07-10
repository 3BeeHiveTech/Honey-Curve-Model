"""
This module contains all of the ORM sqlalchemy models needed to work with the threebee_production
database.
"""
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, inspect
from sqlalchemy.ext.declarative import as_declarative
from sqlalchemy.orm import declarative_base  # type: ignore[attr-defined]


@as_declarative()
class Base:
    def to_dict(self) -> dict:
        return {c.key: getattr(self, c.key) for c in inspect(self).mapper.column_attrs}


class Apicultures(Base):
    """threebee_production.apicultures table.

    NOTE: This is not the full list of columns, which will be updated when needed.
    """

    __tablename__ = "apicultures"

    id = Column(Integer, primary_key=True)
    latitude = Column(Float)
    longitude = Column(Float)

    # Children table
    # hives = relationship("Hives", back_populates="parent")  # type: ignore[var-annotated]

    def __repr__(self) -> str:
        return f"Apicultures(id={self.id!r}, latitude={self.latitude!r}, longitude={self.longitude!r})"


class Hives(Base):
    """threebee_production.hives table.

    NOTE: This is not the full list of columns, which will be updated when needed.
    """

    __tablename__ = "hives"

    id = Column(Integer, primary_key=True)
    apiculture_id = Column(Integer, ForeignKey("apicultures.id"))
    name = Column(String(255))
    latitude = Column(Float)
    longitude = Column(Float)

    # Parent table
    # apicultures = relationship("Apicultures", back_populates="children")  # type: ignore[var-annotated]

    def __repr__(self) -> str:
        return f"Hives(id={self.id!r}, latitude={self.latitude!r}, longitude={self.longitude!r})"


class Weights(Base):
    """threebee_production.weights table.

    NOTE: This is not the full list of columns, which will be updated when needed.
    """

    __tablename__ = "weights"

    id = Column(Integer, primary_key=True)
    device_id = Column(Integer)
    hive_id = Column(Integer)
    acquired_at = Column(DateTime)
    total_weight_value = Column(Float)
    # ...
    created_at = Column(DateTime)
    updated_at = Column(DateTime)

    def __repr__(self) -> str:
        return f"Weights(id={self.id!r}, acquired_at={self.acquired_at!r}, total_weight_value={self.total_weight_value!r})"


class Notes(Base):
    """threebee_production.notes table.

    NOTE: This is not the full list of columns, which will be updated when needed.
    """

    __tablename__ = "notes"

    id = Column(Integer, primary_key=True)
    hive_id = Column(Integer)
    note_type = Column(String)
    scheduled_for = Column(DateTime)
    description = Column(String)
    total_weight_value = Column(Float)
    # ...
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    apiculture_id = Column(Integer)
    # ...
    status = Column(String)

    def __repr__(self) -> str:
        return f"Notes(id={self.id!r}, hive_id={self.hive_id!r}, note_type={self.note_type!r}, scheduled_for={self.scheduled_for!r})"


class Locations(Base):
    """threebee_production.locations table.

    NOTE: This is not the full list of columns, which will be updated when needed.
    """

    __tablename__ = "locations"

    id = Column(Integer, primary_key=True)
    device_id = Column(Integer)
    hive_id = Column(Integer)
    acquired_at = Column(DateTime)
    latitude_value = Column(Float)
    longitude_value = Column(Float)
    # ...
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    # ...

    def __repr__(self) -> str:
        return f"Locations(id={self.id!r}, hive_id={self.hive_id!r}, latitude_value={self.latitude_value!r}, longitude_value={self.longitude_value!r})"


class Temperatures(Base):
    """threebee_production.temperatures table."""

    __tablename__ = "temperatures"

    id = Column(Integer, primary_key=True)
    device_id = Column(Integer)
    hive_id = Column(Integer)
    acquired_at = Column(DateTime)
    value = Column(Float)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    raw_value = Column(Float)

    def __repr__(self) -> str:
        return f"Temperatures(id={self.id!r}, hive_id={self.hive_id!r}, value={self.value!r})"


class SoundIntensities(Base):
    """threebee_production.sound_intensities table."""

    __tablename__ = "sound_intensities"

    id = Column(Integer, primary_key=True)
    device_id = Column(Integer)
    hive_id = Column(Integer)
    acquired_at = Column(DateTime)
    value = Column(Float)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    raw_value = Column(Float)

    def __repr__(self) -> str:
        return f"SoundIntensities(id={self.id!r}, hive_id={self.hive_id!r}, value={self.value!r})"


class Humidities(Base):
    """threebee_production.humidities table."""

    __tablename__ = "humidities"

    id = Column(Integer, primary_key=True)
    device_id = Column(Integer)
    hive_id = Column(Integer)
    acquired_at = Column(DateTime)
    value = Column(Float)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    raw_value = Column(Float)

    def __repr__(self) -> str:
        return f"Humidities(id={self.id!r}, hive_id={self.hive_id!r}, value={self.value!r})"


class Devices(Base):
    """threebee_production.devices table."""

    __tablename__ = "devices"

    id = Column(Integer, primary_key=True)
    hive_id = Column(Integer)
    external_id = Column(Integer)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)

    def __repr__(self) -> str:
        return f"Devices(id={self.id!r}, hive_id={self.hive_id!r}, value={self.external_id!r})"
