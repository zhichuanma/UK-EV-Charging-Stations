from __future__ import annotations

import hashlib
import json
import time
from math import floor
from pathlib import Path

import geopandas as gpd
import osmnx as ox
import pandas as pd
from shapely.geometry import box

BRITISH_NATIONAL_GRID = "EPSG:27700"
WGS84 = "EPSG:4326"
OSM_ELEMENT_COL = "osm_element"
OSM_ID_COL = "osm_id"


def load_station_area_pois(
    df: pd.DataFrame,
    query_tags: dict[str, bool | str | list[str]],
    radius_m: int,
    cache_dir: str | Path,
    tile_size_m: int = 25_000,
    overpass_timeout: int = 180,
    max_retries: int = 3,
    pause_s: float = 1.0,
) -> gpd.GeoDataFrame:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_key = _cache_key(query_tags, radius_m, tile_size_m)
    tile_dir = cache_dir / f"poi_tiles_{cache_key}"
    tile_dir.mkdir(exist_ok=True)
    poi_cache = cache_dir / f"uk_station_area_pois_{cache_key}.parquet"
    keep_cols = list(query_tags.keys()) + ["geometry"]

    if poi_cache.exists():
        print("Loading cached station-area POIs...")
        try:
            return _cache_ready_gdf(gpd.read_parquet(poi_cache))
        except Exception as exc:
            print(f"Cached POI parquet unreadable, rebuilding: {exc}")

    ox.settings.overpass_settings = f"[out:json][timeout:{overpass_timeout}]"

    station_buffers = _build_station_buffers(df=df, radius_m=radius_m)
    tile_to_rows = _map_tiles_to_station_buffers(
        station_buffers=station_buffers,
        tile_size_m=tile_size_m,
    )
    tile_ids = sorted(tile_to_rows)
    print(
        f"Querying {len(tile_ids):,} occupied tiles "
        f"(tile={tile_size_m/1000:.0f}km, radius={radius_m}m)..."
    )

    tile_gdfs: list[gpd.GeoDataFrame] = []
    failures: list[str] = []

    for i, tile_id in enumerate(tile_ids, start=1):
        tile_cache = tile_dir / f"tile_{tile_id[0]}_{tile_id[1]}.parquet"

        if tile_cache.exists():
            print(f"[{i}/{len(tile_ids)}] tile {tile_id[0]},{tile_id[1]}: cache")
            try:
                gdf = _cache_ready_gdf(gpd.read_parquet(tile_cache))
            except Exception as exc:
                print(f"    cache unreadable, redownloading: {exc}")
            else:
                tile_gdfs.append(gdf)
                continue

        print(f"[{i}/{len(tile_ids)}] tile {tile_id[0]},{tile_id[1]}: download")
        local_buffers = station_buffers.iloc[tile_to_rows[tile_id]]

        try:
            gdf = _fetch_tile_pois(
                query_tags=query_tags,
                keep_cols=keep_cols,
                tile_id=tile_id,
                tile_size_m=tile_size_m,
                local_buffers=local_buffers,
                max_retries=max_retries,
                pause_s=pause_s,
            )
        except Exception as exc:
            failures.append(f"tile {tile_id[0]},{tile_id[1]}: {exc}")
            print(f"    failed: {exc}")
            continue

        gdf = _cache_ready_gdf(gdf)
        gdf.to_parquet(tile_cache, index=False)

        tile_gdfs.append(gdf)

    if failures:
        failed_preview = "; ".join(failures[:5])
        raise RuntimeError(
            f"{len(failures)} tile downloads failed. "
            f"Cached tiles are preserved, so rerunning will resume. "
            f"First failures: {failed_preview}"
        )

    pois = _concat_tile_gdfs(tile_gdfs=tile_gdfs)
    pois = _cache_ready_gdf(pois)
    pois.to_parquet(poi_cache, index=False)
    return pois


def _build_station_buffers(df: pd.DataFrame, radius_m: int) -> gpd.GeoDataFrame:
    stations = gpd.GeoDataFrame(
        df[["Latitude", "Longitude"]].copy(),
        geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]),
        crs=WGS84,
    ).to_crs(BRITISH_NATIONAL_GRID)
    stations["geometry"] = stations.geometry.buffer(radius_m)
    return stations[["geometry"]]


def _map_tiles_to_station_buffers(
    station_buffers: gpd.GeoDataFrame,
    tile_size_m: int,
) -> dict[tuple[int, int], list[int]]:
    tile_to_rows: dict[tuple[int, int], set[int]] = {}
    bounds = station_buffers.geometry.bounds

    for row_idx, row in bounds.iterrows():
        min_tx = floor(row.minx / tile_size_m)
        max_tx = floor(row.maxx / tile_size_m)
        min_ty = floor(row.miny / tile_size_m)
        max_ty = floor(row.maxy / tile_size_m)

        for tx in range(min_tx, max_tx + 1):
            for ty in range(min_ty, max_ty + 1):
                tile_to_rows.setdefault((tx, ty), set()).add(row_idx)

    return {tile_id: sorted(row_ids) for tile_id, row_ids in tile_to_rows.items()}


def _fetch_tile_pois(
    query_tags: dict[str, bool | str | list[str]],
    keep_cols: list[str],
    tile_id: tuple[int, int],
    tile_size_m: int,
    local_buffers: gpd.GeoDataFrame,
    max_retries: int,
    pause_s: float,
) -> gpd.GeoDataFrame:
    bbox = _tile_bbox(tile_id=tile_id, tile_size_m=tile_size_m)
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            gdf = ox.features_from_bbox(bbox=bbox, tags=query_tags)
            return _filter_to_station_areas(
                gdf=gdf,
                keep_cols=keep_cols,
                local_buffers=local_buffers,
            )
        except Exception as exc:
            last_error = exc
            if attempt == max_retries:
                break
            sleep_for = pause_s * attempt
            print(f"    retry {attempt}/{max_retries - 1} in {sleep_for:.1f}s")
            time.sleep(sleep_for)

    raise RuntimeError(str(last_error) if last_error is not None else "unknown error")


def _tile_bbox(tile_id: tuple[int, int], tile_size_m: int) -> tuple[float, float, float, float]:
    tx, ty = tile_id
    tile_geom = box(
        tx * tile_size_m,
        ty * tile_size_m,
        (tx + 1) * tile_size_m,
        (ty + 1) * tile_size_m,
    )
    tile_wgs84 = gpd.GeoSeries([tile_geom], crs=BRITISH_NATIONAL_GRID).to_crs(WGS84).iloc[0]
    minx, miny, maxx, maxy = tile_wgs84.bounds
    return (minx, miny, maxx, maxy)


def _filter_to_station_areas(
    gdf: gpd.GeoDataFrame,
    keep_cols: list[str],
    local_buffers: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    if gdf.empty:
        return _empty_gdf(columns=keep_cols)

    gdf = gdf[[col for col in keep_cols if col in gdf.columns]].copy()
    if gdf.empty:
        return _empty_gdf(columns=keep_cols)

    gdf = gdf[gdf.geometry.notna()].copy()
    if gdf.empty:
        return _empty_gdf(columns=keep_cols)

    local_cover = _union_geometry(local_buffers.geometry)
    gdf_proj = gdf.to_crs(BRITISH_NATIONAL_GRID)
    mask = gdf_proj.intersects(local_cover)
    gdf = gdf.loc[mask].copy()

    if gdf.empty:
        return _empty_gdf(columns=keep_cols)

    return gdf


def _concat_tile_gdfs(tile_gdfs: list[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    if not tile_gdfs:
        return _empty_gdf(columns=["geometry"])

    non_empty = [gdf for gdf in tile_gdfs if not gdf.empty]
    if not non_empty:
        return _empty_gdf(columns=tile_gdfs[0].columns.tolist())

    pois = gpd.GeoDataFrame(
        pd.concat(non_empty, axis=0),
        geometry="geometry",
        crs=non_empty[0].crs or WGS84,
    )

    if {OSM_ELEMENT_COL, OSM_ID_COL}.issubset(pois.columns):
        return pois.loc[~pois.duplicated(subset=[OSM_ELEMENT_COL, OSM_ID_COL], keep="first")].copy()

    return pois.loc[~pois.index.duplicated(keep="first")].copy()


def _empty_gdf(columns: list[str]) -> gpd.GeoDataFrame:
    data = {col: pd.Series(dtype="object") for col in columns if col != "geometry"}
    return gpd.GeoDataFrame(
        data,
        geometry=gpd.GeoSeries([], crs=WGS84),
        crs=WGS84,
    )


def _cache_ready_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    prepared = gdf.copy()
    crs = prepared.crs

    if isinstance(prepared.index, pd.MultiIndex):
        level_names = [
            name if name is not None else f"level_{i}"
            for i, name in enumerate(prepared.index.names)
        ]
        prepared = prepared.reset_index()
        rename_map: dict[str, str] = {}
        if level_names:
            rename_map[level_names[0]] = OSM_ELEMENT_COL
        if len(level_names) > 1:
            rename_map[level_names[1]] = OSM_ID_COL
        prepared = prepared.rename(columns=rename_map)

    if OSM_ELEMENT_COL in prepared.columns:
        prepared[OSM_ELEMENT_COL] = prepared[OSM_ELEMENT_COL].astype("string")
    if OSM_ID_COL in prepared.columns:
        prepared[OSM_ID_COL] = prepared[OSM_ID_COL].astype("string")

    return gpd.GeoDataFrame(prepared, geometry="geometry", crs=crs or WGS84)


def _cache_key(
    query_tags: dict[str, bool | str | list[str]],
    radius_m: int,
    tile_size_m: int,
) -> str:
    payload = {
        "radius_m": radius_m,
        "tile_size_m": tile_size_m,
        "query_tags": _normalized_query_tags(query_tags),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:12]


def _normalized_query_tags(
    query_tags: dict[str, bool | str | list[str]],
) -> dict[str, bool | str | list[str]]:
    normalized: dict[str, bool | str | list[str]] = {}
    for key in sorted(query_tags):
        value = query_tags[key]
        if isinstance(value, list):
            normalized[key] = sorted(value)
        else:
            normalized[key] = value
    return normalized


def _union_geometry(geometries: gpd.GeoSeries):
    union_all = getattr(geometries, "union_all", None)
    if callable(union_all):
        return union_all()
    return geometries.unary_union
