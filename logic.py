from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import folium
import networkx as nx
import osmnx as ox
import requests

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_AVG_SPEED_KPH = 30
DEFAULT_NETWORK_TYPE = "drive"
SUPPORTED_ALGORITHMS = {"dijkstra", "bellman-ford", "astar"}
CKAN_BASE_URL = "https://data.ibb.gov.tr/api/3/action"


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.strip().lower())
    slug = slug.strip("_")
    return slug or "graph"


def _ensure_travel_attributes(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    sample_edge = next(iter(graph.edges(data=True)), None)
    if sample_edge:
        attrs = sample_edge[2]
        if "travel_time" in attrs and "speed_kph" in attrs:
            return graph
    graph = ox.add_edge_speeds(graph)
    graph = ox.add_edge_travel_times(graph)
    return graph


def get_graph(place_name: str = "Kadıköy, Istanbul, Turkey", *, use_cache: bool = True) -> Optional[nx.MultiDiGraph]:
    """
    Gerçek trafik ağını (OSM sürüş ağı) indirir veya önbellekten yükler.
    """
    slug = _slugify(place_name)
    cache_path = CACHE_DIR / f"{slug}.graphml"

    try:
        graph: Optional[nx.MultiDiGraph] = None
        if use_cache and cache_path.exists():
            graph = ox.load_graphml(cache_path)
        else:
            graph = ox.graph_from_place(place_name, network_type=DEFAULT_NETWORK_TYPE)
            graph = _ensure_travel_attributes(graph)
            if use_cache:
                ox.save_graphml(graph, cache_path)

        if graph is not None:
            graph = _ensure_travel_attributes(graph)
        return graph
    except Exception as exc:  # pragma: no cover - log + fail gracefully
        print(f"Error fetching graph: {exc}")
        return None


def _build_weight_function(metric: str, traffic_multiplier: float) -> Tuple[Callable[[int, int, Dict], float], str]:
    attr = "travel_time" if metric == "duration" else "length"

    def weight(u: int, v: int, data: Dict) -> float:
        value = data.get(attr)
        if value is None:
            value = data.get("length", 0.0)
        if metric == "duration":
            return value * traffic_multiplier
        return value

    return weight, attr


def _heuristic_factory(graph: nx.MultiDiGraph, metric: str, traffic_multiplier: float) -> Callable[[int, int], float]:
    avg_speed_mps = max(DEFAULT_AVG_SPEED_KPH * 1000 / 3600, 0.1)

    def heuristic(u: int, v: int) -> float:
        u_node = graph.nodes[u]
        v_node = graph.nodes[v]
        dist_m = ox.distance.great_circle_vec(u_node["y"], u_node["x"], v_node["y"], v_node["x"])
        if metric == "duration":
            return (dist_m / avg_speed_mps) * traffic_multiplier
        return dist_m

    return heuristic


def _route_metrics(graph: nx.MultiDiGraph, route: List[int], traffic_multiplier: float) -> Tuple[float, float]:
    distance = 0.0
    duration = 0.0
    if not route:
        return distance, duration

    avg_speed_mps = max(DEFAULT_AVG_SPEED_KPH * 1000 / 3600, 0.1)

    for u, v in zip(route[:-1], route[1:]):
        edge_data = graph.get_edge_data(u, v)
        if not edge_data:
            continue
        # MultiDiGraph -> pick the edge with minimum length as approximation
        best_edge = min(edge_data.values(), key=lambda data: data.get("length", math.inf))
        edge_length = best_edge.get("length", 0.0)
        edge_time = best_edge.get("travel_time")
        if edge_time is None:
            speed_kph = best_edge.get("speed_kph", DEFAULT_AVG_SPEED_KPH)
            speed_mps = max(speed_kph * 1000 / 3600, 0.1)
            edge_time = edge_length / speed_mps
        distance += edge_length
        duration += edge_time * traffic_multiplier
    return distance, duration


def calculate_route(
    graph: nx.MultiDiGraph,
    start_coords: Tuple[float, float],
    end_coords: Tuple[float, float],
    *,
    algorithm: str = "dijkstra",
    metric: str = "distance",
    traffic_multiplier: float = 1.0,
) -> Optional[Dict]:
    """
    İki koordinasyon arasındaki en kısa yolu seçilen algoritmayla hesaplar.
    metric: 'distance' -> metre bazlı, 'duration' -> saniye bazlı.
    """
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    try:
        orig_node = ox.distance.nearest_nodes(graph, start_coords[1], start_coords[0])
        dest_node = ox.distance.nearest_nodes(graph, end_coords[1], end_coords[0])
        weight_fn, attr = _build_weight_function(metric, traffic_multiplier)

        if algorithm == "astar":
            heuristic = _heuristic_factory(graph, metric, traffic_multiplier)
            route = nx.astar_path(graph, orig_node, dest_node, heuristic=heuristic, weight=weight_fn)
            objective = nx.astar_path_length(graph, orig_node, dest_node, heuristic=heuristic, weight=weight_fn)
        else:
            route = nx.shortest_path(graph, orig_node, dest_node, weight=weight_fn, method=algorithm)
            objective = nx.shortest_path_length(graph, orig_node, dest_node, weight=weight_fn)

        distance, duration = _route_metrics(graph, route, traffic_multiplier)
        return {
            "route": route,
            "objective_cost": objective,
            "distance_m": distance,
            "duration_s": duration,
            "start_node": orig_node,
            "end_node": dest_node,
            "metric_attr": attr,
        }
    except Exception as exc:
        print(f"Error calculating route: {exc}")
        return None


def _normalize_drop_points(drop_points: Iterable[Dict]) -> List[Dict]:
    normalized: List[Dict] = []
    for idx, point in enumerate(drop_points, start=1):
        coords = point.get("coords")
        if not coords:
            continue
        name = point.get("name") or f"Teslimat {idx}"
        normalized.append({"name": name, "coords": coords})
    return normalized


def _optimize_order(
    graph: nx.MultiDiGraph,
    start_node: int,
    stops: List[Dict],
    metric: str,
    traffic_multiplier: float,
) -> List[Dict]:
    if len(stops) <= 1:
        return stops

    remaining = []
    for stop in stops:
        node = ox.distance.nearest_nodes(graph, stop["coords"][1], stop["coords"][0])
        remaining.append({**stop, "node": node})

    ordered: List[Dict] = []
    current_node = start_node
    weight_fn, _ = _build_weight_function(metric, traffic_multiplier)

    while remaining:
        best_idx = 0
        best_cost = math.inf
        for idx, candidate in enumerate(remaining):
            try:
                cost = nx.shortest_path_length(
                    graph,
                    current_node,
                    candidate["node"],
                    weight=weight_fn,
                    method="dijkstra",
                )
            except nx.NetworkXNoPath:
                cost = math.inf
            if cost < best_cost:
                best_cost = cost
                best_idx = idx
        chosen = remaining.pop(best_idx)
        ordered.append(chosen)
        current_node = chosen["node"]

    for stop in ordered:
        stop.pop("node", None)
    return ordered


def plan_multi_stop_route(
    graph: nx.MultiDiGraph,
    start_coords: Tuple[float, float],
    drop_points: Iterable[Dict],
    *,
    algorithm: str = "dijkstra",
    metric: str = "distance",
    traffic_multiplier: float = 1.0,
    optimize_order: bool = True,
) -> Optional[Dict]:
    normalized = _normalize_drop_points(drop_points)
    if not normalized:
        return None

    start_node = ox.distance.nearest_nodes(graph, start_coords[1], start_coords[0])
    ordered_points = _optimize_order(graph, start_node, normalized, metric, traffic_multiplier) if optimize_order else normalized

    all_nodes: List[int] = []
    segments: List[Dict] = []
    current_coords = start_coords

    for point in ordered_points:
        segment = calculate_route(
            graph,
            current_coords,
            point["coords"],
            algorithm=algorithm,
            metric=metric,
            traffic_multiplier=traffic_multiplier,
        )
        if not segment or not segment["route"]:
            continue
        if all_nodes:
            all_nodes.extend(segment["route"][1:])
        else:
            all_nodes.extend(segment["route"])
        segments.append(
            {
                "name": point["name"],
                "distance_m": segment["distance_m"],
                "duration_s": segment["duration_s"],
                "end_coords": point["coords"],
            }
        )
        current_coords = point["coords"]

    if not segments:
        return None

    total_distance = sum(seg["distance_m"] for seg in segments)
    total_duration = sum(seg["duration_s"] for seg in segments)
    return {
        "route": all_nodes,
        "segments": segments,
        "distance_m": total_distance,
        "duration_s": total_duration,
        "order": [seg["name"] for seg in segments],
    }


def route_to_coordinates(graph: nx.MultiDiGraph, route: List[int]) -> List[Dict[str, float]]:
    coords: List[Dict[str, float]] = []
    for node in route or []:
        node_data = graph.nodes[node]
        coords.append({"lat": node_data["y"], "lon": node_data["x"]})
    return coords


def create_map(
    graph: nx.MultiDiGraph,
    route: Optional[List[int]] = None,
    start_coords: Optional[Tuple[float, float]] = None,
    end_coords: Optional[Tuple[float, float]] = None,
    drop_points: Optional[Iterable[Dict]] = None,
):
    """
    Graf üzerinde rotayı ve teslimat noktalarını gösteren Folium haritası üretir.
    """
    if start_coords:
        center_lat, center_lon = start_coords
    else:
        gdf_nodes = ox.graph_to_gdfs(graph, edges=False)
        center_lat = gdf_nodes.y.mean()
        center_lon = gdf_nodes.x.mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, control_scale=True)

    if route:
        # Manuel olarak rotayı çiz (ox.plot_route_folium artık mevcut değil)
        try:
            route_coords = [(graph.nodes[node]['y'], graph.nodes[node]['x']) for node in route]
            folium.PolyLine(
                locations=route_coords,
                color="red",
                weight=5,
                opacity=0.8
            ).add_to(m)
        except KeyError:
            # Rota node'ları mevcut graph'ta yoksa (farklı bölge seçilmiş olabilir)
            pass

    if start_coords:
        folium.Marker(
            location=start_coords,
            popup="Başlangıç Noktası (Depo)",
            icon=folium.Icon(color="green", icon="play"),
        ).add_to(m)

    if end_coords:
        folium.Marker(
            location=end_coords,
            popup="Son Teslimat",
            icon=folium.Icon(color="blue", icon="flag"),
        ).add_to(m)

    if drop_points:
        for idx, point in enumerate(drop_points, start=1):
            folium.Marker(
                location=point["coords"],
                popup=f"{point.get('name', f'Teslimat {idx}')}",
                icon=folium.Icon(color="orange", icon="info-sign"),
            ).add_to(m)

    return m


def _call_ckan_api(action: str, params: Dict[str, str]) -> Dict:
    url = f"{CKAN_BASE_URL}/{action}"
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        raise RuntimeError(f"CKAN API isteği başarısız oldu ({action}).") from exc

    if not payload.get("success"):
        error = payload.get("error", {})
        message = error.get("message") or str(error)
        raise RuntimeError(f"CKAN API hatası: {message}")
    return payload["result"]


def fetch_ckan_dataset(dataset_id: str) -> Dict:
    """
    CKAN package_show sonucunu döndürür.
    """
    return _call_ckan_api("package_show", {"id": dataset_id})


def fetch_ckan_resource(resource_id: str) -> Dict:
    """
    CKAN resource_show sonucunu döndürür.
    """
    return _call_ckan_api("resource_show", {"id": resource_id})
