from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


DATA_DIR = Path(r"d:\My_Projects\RoadCast AI\data\raw\metr-la")


def load_sensor_location_reference() -> pd.DataFrame:
    """Return a clean node_id → latitude/longitude mapping when the dataset provides it.

    METR-LA does not carry explicit geolocation in the provided parquet files, so the
    repository currently has no real sensor coordinates to map. We therefore return an
    empty DataFrame rather than inventing coordinates, which preserves data integrity.
    """
    return pd.DataFrame(columns=["node_id", "latitude", "longitude", "road_name"])


def get_sensor_locations() -> List[Dict[str, Any]]:
    """Return all available sensor metadata, if any exists in the repository.

    If no coordinates are available, this returns an empty list so the frontend can show
    the map without fabricating any lat/lon values.
    """
    df = load_sensor_location_reference()
    if df.empty:
        return []
    return df.fillna(value={"road_name": None}).to_dict(orient="records")


def build_node_location_lookup() -> Dict[int, Dict[str, Any]]:
    payload = get_sensor_locations()
    lookup: Dict[int, Dict[str, Any]] = {}
    for item in payload:
        node_id = item.get("node_id")
        if node_id is not None:
            lookup[int(node_id)] = item
    return lookup
