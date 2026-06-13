"""
Piezo.AI — SQLAlchemy Base Model
==================================
All ORM models inherit from this Base.
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass
