"""titiler.core.algorithm."""

import json
from copy import copy
from typing import Dict, List, Literal, Optional, Type

import attr
from fastapi import HTTPException, Query
from pydantic import ValidationError
from typing_extensions import Annotated

from titiler.core.algorithm.base import (  # noqa
    AlgorithmMetadata,
    AlgorithmtList,
    BaseAlgorithm,
)
from titiler.core.algorithm.dem import Contours, HillShade, Slope, TerrainRGB, Terrarium
from titiler.core.algorithm.image import ToBitonal, ToGrayScale
from titiler.core.algorithm.index import NormalizedIndex
from titiler.core.algorithm.math import _Max, _Mean, _Median, _Min, _Std, _Sum, _Var
from titiler.core.algorithm.ops import CastToInt, Ceil, Floor

default_algorithms: Dict[str, Type[BaseAlgorithm]] = {
    "hillshade": HillShade,
    "slope": Slope,
    "contours": Contours,
    "normalizedIndex": NormalizedIndex,
    "terrarium": Terrarium,
    "terrainrgb": TerrainRGB,
    "cast": CastToInt,
    "ceil": Ceil,
    "floor": Floor,
    "min": _Min,
    "max": _Max,
    "median": _Median,
    "mean": _Mean,
    "std": _Std,
    "var": _Var,
    "sum": _Sum,
    "grayscale": ToGrayScale,
    "bitonal": ToBitonal,
}


@attr.s(frozen=True)
class Algorithms:
    """Algorithms."""

    data: Dict[str, Type[BaseAlgorithm]] = attr.ib()

    def get(self, name: str) -> BaseAlgorithm:
        """Fetch a TMS."""
        if name not in self.data:
            raise KeyError(f"Invalid name: {name}")

        return self.data[name]

    def list(self) -> List[str]:
        """List registered Algorithm."""
        return list(self.data.keys())

    def register(
        self,
        algorithms: Dict[str, BaseAlgorithm],
        overwrite: bool = False,
    ) -> "Algorithms":
        """Register Algorithm(s)."""
        # Optimize membership checking by using set operations (average-case O(n) -> O(1) for large inputs)
        if not overwrite:
            overlap = self.data.keys() & algorithms.keys()
            if overlap:
                # Only raise on the FIRST duplicate found (as before)
                for name in overlap:
                    raise Exception(f"{name} is already a registered. Use overwrite=True.")

        # If overwrite is True or no duplicates: return new instance
        # dict unpacking is already optimal (Python 3.9+ C-path)

        return Algorithms({**self.data, **algorithms})

    @property
    def dependency(self):
        """FastAPI PostProcess dependency."""

        def post_process(
            algorithm: Annotated[
                Literal[tuple(self.data.keys())],
                Query(description="Algorithm name"),
            ] = None,
            algorithm_params: Annotated[
                Optional[str],
                Query(description="Algorithm parameter"),
            ] = None,
        ) -> Optional[BaseAlgorithm]:
            """Data Post-Processing options."""
            kwargs = json.loads(algorithm_params) if algorithm_params else {}
            if algorithm:
                try:
                    return self.get(algorithm)(**kwargs)

                except ValidationError as e:
                    raise HTTPException(status_code=400, detail=str(e)) from e

            return None

        return post_process


algorithms = Algorithms(copy(default_algorithms))  # noqa
