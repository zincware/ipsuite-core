"""Protocols for IPSuite Nodes type annotations."""

import typing as t

import ase
from ase.calculators.calculator import Calculator

calc = t.TypeVar("calc", bound=Calculator)


class HasCalculator(t.Protocol):
    """Protocol for objects that have a calculator."""

    def get_calculator(self, **kwargs) -> calc:
        """Get the calculator."""
        ...


class HasAtoms(t.Protocol):
    """Protocol for objects that have atoms."""

    atoms: list[ase.Atoms]
