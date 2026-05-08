"""
LawnRoute OSM Graph Builder v3
Improved classification engine with correct handling of:
- Golf course features (bunkers, greens, fairways, paths)
- Structures (buildings, awnings, shelters, pools)
- Parking lots and paved surfaces
- Driveways, sidewalks, patios, landscape beds
- Tree lines and vegetation
- Dual sidewalk edges (both sides of every path)
- Building perimeter trim edges
- Proper equipment constraints per corridor width
- Blow zones: driveways, sidewalks, patios, parking, hardscape
"""

import os
import json
import time
import logging
import requests
import pg8000
import pg8000.dbapi
from datetime import datetime, timezone
from typing import Optional
from shapely.geometry import shape, LineString, Polygon, Point, mapping
from shapely.ops import unary_union
try:
    from shapely.ops import offset_curve
except ImportError:
    from shapely import offset_curve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Database connection
# ---------------------------------------------------------------------------

def get_db():
    import ssl
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    return pg8000.dbapi.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 5432)),
        database=os.getenv("DB_NAME", "lawnroute"),
        user=os.getenv("DB_USER", "lawnroute"),
        password=os.getenv("DB_PASSWORD", "lawnroute_dev_2025"),
        ssl_context=ssl_context,
    )

def dict_row(conn, row, description):
    """Convert a row tuple to a dict using column descriptions."""
    if row is None:
        return None
    return {description[i][0]: row[i] for i in range(len(description))}

def fetchone_dict(cur):
    row = cur.fetchone()
    if row is None:
        return None
    return {cur.description[i][0]: row[i] for i in range(len(cur.description))}

def fetchall_dict(cur):
    rows = cur.fetchall()
    if not rows:
        return []
    return [{cur.description[i][0]: row[i] for i in range(len(cur.description))} for row in rows]

# ---------------------------------------------------------------------------
# Overpass API
# ---------------------------------------------------------------------------

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

OVERPASS_QUERY_TEMPLATE = """[out:json][timeout:90];
(
  way["building"]{bbox};
  way["building:part"]{bbox};
  way["amenity"="parking"]{bbox};
  way["amenity"="swimming_pool"]{bbox};
  way["leisure"="swimming_pool"]{bbox};
  way["leisure"="park"]{bbox};
  way["leisure"="garden"]{bbox};
  way["leisure"="pitch"]{bbox};
  way["leisure"="golf_course"]{bbox};
  way["golf"]{bbox};
  way["landuse"]{bbox};
  way["natural"]{bbox};
  way["highway"]{bbox};
  way["footway"]{bbox};
  way["path"]{bbox};
  way["barrier"]{bbox};
  way["waterway"]{bbox};
  way["man_made"]{bbox};
  way["surface"="asphalt"]{bbox};
  way["surface"="concrete"]{bbox};
  way["surface"="paving_stones"]{bbox};
  way["surface"="gravel"]{bbox};
  way["surface"="sand"]{bbox};
  way["surface"="brick"]{bbox};
  way["surface"="tiles"]{bbox};
  way["leisure"="outdoor_seating"]{bbox};
  node["natural"="tree"]{bbox};
  node["man_made"="utility_pole"]{bbox};
  node["man_made"="pole"]{bbox};
  node["barrier"="bollard"]{bbox};
  node["barrier"="gate"]{bbox};
  node["amenity"="fire_hydrant"]{bbox};
);
out body;
>;
out skel qt;
"""

def build_bbox_string(polygon: Polygon) -> str:
    minx, miny, maxx, maxy = polygon.bounds
    buffer = 0.0002
    return f"({miny - buffer},{minx - buffer},{maxy + buffer},{maxx + buffer})"

def query_overpass(polygon: Polygon, property_id: str) -> dict:
    bbox = build_bbox_string(polygon)
    query = OVERPASS_QUERY_TEMPLATE.replace("{bbox}", bbox)

    for attempt in range(3):
        try:
            logger.info(f"Querying Overpass API (attempt {attempt + 1})")
            resp = requests.post(
                OVERPASS_URL,
                data=f"data={requests.utils.quote(query)}",
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "User-Agent": "LawnRoute/1.0 (lawn care route optimizer)",
                },
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            logger.info(f"Overpass returned {len(data.get('elements', []))} elements")
            return data
        except requests.RequestException as e:
            logger.warning(f"Overpass attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                time.sleep(5 * (attempt + 1))

    raise RuntimeError("Overpass API failed after 3 attempts")

# ---------------------------------------------------------------------------
# OSM element parsing
# ---------------------------------------------------------------------------

def parse_osm_elements(data: dict) -> tuple:
    nodes_by_id = {}
    ways = []

    for element in data.get("elements", []):
        if element["type"] == "node":
            nodes_by_id[element["id"]] = {
                "lat": element["lat"],
                "lon": element["lon"],
                "tags": element.get("tags", {}),
            }
        elif element["type"] == "way":
            ways.append({
                "id": element["id"],
                "nodes": element.get("nodes", []),
                "tags": element.get("tags", {}),
            })

    return nodes_by_id, ways

def way_to_coords(way: dict, nodes_by_id: dict) -> list:
    coords = []
    for node_id in way["nodes"]:
        node = nodes_by_id.get(node_id)
        if node:
            coords.append((node["lon"], node["lat"]))
    return coords

def coords_to_linestring(coords: list) -> Optional[LineString]:
    if len(coords) < 2:
        return None
    return LineString(coords)

def coords_to_polygon(coords: list) -> Optional[Polygon]:
    if len(coords) < 3:
        return None
    try:
        p = Polygon(coords)
        if p.is_valid and not p.is_empty:
            return p
        return None
    except Exception:
        return None

def is_closed_way(way: dict) -> bool:
    nodes = way.get("nodes", [])
    return len(nodes) > 2 and nodes[0] == nodes[-1]

# ---------------------------------------------------------------------------
# Equipment width constants (meters)
# ---------------------------------------------------------------------------

RIDING_MOWER_WIDTH = 1.83
WALK_BEHIND_WIDTH = 0.76
TRIMMER_WIDTH = 0.46
SIDEWALK_OFFSET_M = 0.5

DEG_PER_METER = 1.0 / 111320

# ---------------------------------------------------------------------------
# Feature classification
# ---------------------------------------------------------------------------

def classify_way(tags: dict, coords: list, boundary: Polygon) -> list:
    highway    = tags.get("highway", "")
    surface    = tags.get("surface", "")
    landuse    = tags.get("landuse", "")
    leisure    = tags.get("leisure", "")
    natural    = tags.get("natural", "")
    building   = tags.get("building", "")
    building_p = tags.get("building:part", "")
    amenity    = tags.get("amenity", "")
    golf       = tags.get("golf", "")
    barrier    = tags.get("barrier", "")
    waterway   = tags.get("waterway", "")
    man_made   = tags.get("man_made", "")
    width_tag  = tags.get("width", None)
    closed     = len(coords) > 2 and coords[0] == coords[-1]

    if waterway in ("river", "stream", "canal", "drain", "ditch"):
        return [_edge("none", "water_feature", coords, tags, "Water - excluded")]

    if amenity == "swimming_pool" or leisure == "swimming_pool":
        return [_edge("none", "swimming_pool", coords, tags, "Pool - excluded")]

    if surface == "sand" or golf == "bunker" or natural == "sand":
        return [_edge("none", "sand_bunker", coords, tags, "Sand/bunker - excluded")]

    if golf in ("green", "tee", "hole"):
        return [_edge("none", "golf_feature", coords, tags, f"Golf {golf} - excluded")]

    if highway in ("motorway", "trunk", "primary", "secondary", "tertiary",
                   "residential", "unclassified", "living_street"):
        if closed:
            return [_edge("blow", "road_surface", coords, tags, "Road/parking polygon - blow surface")]
        return [_edge("none", "road", coords, tags, "Public road - excluded")]

    if highway == "service" or tags.get("service") in ("driveway", "parking_aisle", "alley"):
        line = coords_to_linestring(coords)
        if line:
            clipped = line.intersection(boundary)
            if not clipped.is_empty:
                segs = _split_geometry(clipped)
                edges = []
                for s in segs:
                    if s.length > 0:
                        edges.append(_edge("blow", "driveway", _geom_coords(s), tags, "Driveway - blow zone"))
                        edges.append(_edge("trim", "driveway_edge", _geom_coords(s), tags, "Driveway edge - trim", "trimmer"))
                return edges

    if amenity == "parking" or landuse == "parking":
        edges = [_edge("blow", "parking_lot", coords, tags, "Parking lot - blow")]
        if closed and len(coords) >= 3:
            poly = coords_to_polygon(coords)
            if poly and boundary.intersects(poly):
                clipped_poly = poly.intersection(boundary)
                if hasattr(clipped_poly, 'exterior'):
                    segs = _split_geometry(clipped_poly.exterior)
                    edges += [_edge("trim", "parking_edge", _geom_coords(s), tags, "Parking edge - trim", "trimmer") for s in segs if s.length > 0]
        return edges

    if leisure == "outdoor_seating" or amenity == "terrace" or man_made in ("terrace", "patio") or landuse == "plaza":
        if closed and len(coords) >= 3:
            poly = coords_to_polygon(coords)
            if poly and boundary.intersects(poly):
                edges = [_edge("blow", "patio", coords, tags, "Patio - blow zone")]
                clipped_poly = poly.intersection(boundary)
                if hasattr(clipped_poly, 'exterior'):
                    segs = _split_geometry(clipped_poly.exterior)
                    edges += [_edge("trim", "patio_edge", _geom_coords(s), tags, "Patio edge - trim", "trimmer") for s in segs if s.length > 0]
                return edges

    if landuse in ("flowerbed", "plant_nursery") or natural == "flowerbed" or (leisure == "garden" and closed):
        if closed and len(coords) >= 3:
            poly = coords_to_polygon(coords)
            if poly and boundary.intersects(poly):
                clipped_poly = poly.intersection(boundary)
                edges = [_edge("blow", "landscape_bed", coords, tags, "Landscape bed - blow")]
                if hasattr(clipped_poly, 'exterior'):
                    segs = _split_geometry(clipped_poly.exterior)
                    edges += [_edge("trim", "landscape_bed_edge", _geom_coords(s), tags, "Landscape bed edge - trim", "trimmer") for s in segs if s.length > 0]
                return edges

    if building or building_p or man_made in ("shed", "greenhouse", "storage_tank", "silo", "water_tower"):
        if closed and len(coords) >= 3:
            poly = coords_to_polygon(coords)
            if poly and boundary.intersects(poly):
                perimeter = poly.exterior
                clipped = perimeter.intersection(boundary)
                if not clipped.is_empty:
                    segs = _split_geometry(clipped)
                    return [_edge("trim", "building_perimeter", _geom_coords(s), tags, "Building base - trim", "trimmer") for s in segs if s.length > 0]
        return [_edge("none", "building", coords, tags, "Building - excluded")]

    if building in ("roof", "canopy", "awning", "shelter", "carport") or man_made in ("canopy", "shelter"):
        if closed and len(coords) >= 3:
            poly = coords_to_polygon(coords)
            if poly and boundary.intersects(poly):
                perimeter = poly.exterior
                clipped = perimeter.intersection(boundary)
                if not clipped.is_empty:
                    segs = _split_geometry(clipped)
                    return [_edge("trim", "structure_perimeter", _geom_coords(s), tags, "Structure - trim", "trimmer") for s in segs if s.length > 0]
        return [_edge("none", "structure", coords, tags, "Structure - excluded")]

    if barrier in ("fence", "wall", "hedge", "retaining_wall", "guard_rail"):
        line = coords_to_linestring(coords)
        if line:
            clipped = line.intersection(boundary)
            segs = _split_geometry(clipped)
            return [_edge("trim", "barrier_edge", _geom_coords(s), tags, f"{barrier} - trim", "trimmer") for s in segs if s.length > 0]

    if highway in ("footway", "path", "steps", "pedestrian", "cycleway", "bridleway", "track") or tags.get("footway") or tags.get("sidewalk") in ("yes", "left", "right", "both"):
        line = coords_to_linestring(coords)
        if not line:
            return []
        clipped = line.intersection(boundary)
        if clipped.is_empty:
            return []
        width_m = _parse_width(width_tag, 2.0)
        offset_deg = (width_m / 2) * DEG_PER_METER
        edges = []
        segs = _split_geometry(clipped)
        for seg in segs:
            if seg.length == 0:
                continue
            edges.append(_edge("blow", "sidewalk", _geom_coords(seg), tags, "Sidewalk - blow zone"))
            try:
                left = offset_curve(seg, offset_deg)
                left_clipped = left.intersection(boundary)
                if not left_clipped.is_empty:
                    edges.append(_edge("trim", "path_left_edge", _geom_coords(left_clipped), tags, "Path left - trim", "trimmer"))
            except Exception:
                pass
            try:
                right = offset_curve(seg, -offset_deg)
                right_clipped = right.intersection(boundary)
                if not right_clipped.is_empty:
                    edges.append(_edge("trim", "path_right_edge", _geom_coords(right_clipped), tags, "Path right - trim", "trimmer"))
            except Exception:
                pass
        return edges

    if surface in ("asphalt", "concrete", "paving_stones", "cobblestone", "sett", "metal", "wood", "brick", "tiles", "gravel", "compacted"):
        if closed and len(coords) >= 3:
            poly = coords_to_polygon(coords)
            if poly and boundary.intersects(poly):
                clipped_poly = poly.intersection(boundary)
                edges = [_edge("blow", "paved_area", coords, tags, "Paved area - blow")]
                perimeter = clipped_poly.exterior if hasattr(clipped_poly, 'exterior') else None
                if perimeter:
                    segs = _split_geometry(perimeter)
                    edges += [_edge("trim", "paved_edge", _geom_coords(s), tags, "Paved edge - trim", "trimmer") for s in segs if s.length > 0]
                return edges
        line = coords_to_linestring(coords)
        if line:
            return [_edge("blow", "paved_surface", coords, tags, "Paved surface - blow")]

    if natural == "tree_row":
        line = coords_to_linestring(coords)
        if line:
            clipped = line.intersection(boundary)
            segs = _split_geometry(clipped)
            return [_edge("none", "tree_row", _geom_coords(s), tags, "Tree row - obstacle") for s in segs if s.length > 0]

    if golf in ("fairway", "rough", "lateral_water_hazard"):
        if closed and len(coords) >= 3:
            poly = coords_to_polygon(coords)
            if poly and boundary.intersects(poly):
                equipment = "riding_mower" if poly.area * 1e10 > 5000 else "walk_behind"
                return [_edge("mow", "golf_fairway", coords, tags, f"Golf {golf} - mow", equipment)]

    if landuse in ("grass", "meadow", "recreation_ground", "greenfield") or \
       leisure in ("park", "pitch", "common", "golf_course") or \
       natural in ("grassland", "scrub", "heath"):
        if closed and len(coords) >= 3:
            poly = coords_to_polygon(coords)
            if poly and boundary.intersects(poly):
                area_sqm = poly.area * 1e10
                equipment = _mow_equipment_by_area(area_sqm)
                clipped_poly = poly.intersection(boundary)
                edges = [_edge("mow", "open_turf", coords, tags, f"Open turf - {equipment}", equipment)]
                if hasattr(clipped_poly, 'exterior'):
                    perimeter_segs = _split_geometry(clipped_poly.exterior)
                    edges += [_edge("trim", "turf_perimeter", _geom_coords(s), tags, "Turf perimeter - trim", "trimmer") for s in perimeter_segs if s.length > 0]
                return edges

    line = coords_to_linestring(coords)
    if line and boundary.intersects(line):
        return [_edge("mow", "unclassified", coords, tags, "Unclassified - assumed mow", "walk_behind", needs_review=True)]

    return []


def _parse_width(width_tag: Optional[str], default: float) -> float:
    if width_tag:
        try:
            return float(width_tag.replace("m", "").strip())
        except ValueError:
            pass
    return default


def _mow_equipment_by_area(area_sqm: float) -> str:
    if area_sqm > 10000:
        return "riding_mower"
    elif area_sqm > 2000:
        return "walk_behind"
    else:
        return "trimmer"


def _split_geometry(geom) -> list:
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "LineString":
        return [geom]
    if geom.geom_type in ("MultiLineString", "GeometryCollection"):
        result = []
        for g in geom.geoms:
            if g.geom_type == "LineString" and g.length > 0:
                result.append(g)
        return result
    return []


def _geom_coords(geom) -> list:
    if hasattr(geom, "coords"):
        return list(geom.coords)
    return []


def _edge(task_type: str, feature_class: str, coords: list, tags: dict,
          reason: str, equipment: str = "none", needs_review: bool = False) -> dict:
    return {
        "task_type": task_type,
        "feature_class": feature_class,
        "coords": coords,
        "equipment_constraint": equipment,
        "classification_reason": reason,
        "needs_review": needs_review,
        "osm_tags": tags,
        "classified_at": datetime.now(timezone.utc).isoformat(),
    }

# ---------------------------------------------------------------------------
# Tight space detection
# ---------------------------------------------------------------------------

def detect_tight_spaces(edges: list) -> list:
    tight_flagged = []

    for i, edge in enumerate(edges):
        geom_a = edge.get("geometry")
        if not geom_a or edge["task_type"] == "none":
            tight_flagged.append(edge)
            continue

        min_clearance = None
        constraining_feature = None

        for j, other in enumerate(edges):
            if i == j:
                continue
            geom_b = other.get("geometry")
            if not geom_b or other["task_type"] == "none":
                continue
            try:
                dist_m = geom_a.distance(geom_b) * 111320
                if min_clearance is None or dist_m < min_clearance:
                    min_clearance = dist_m
                    constraining_feature = other.get("feature_class", "unknown")
            except Exception:
                continue

        updated = dict(edge)
        if min_clearance is not None:
            updated["measured_corridor_width_m"] = round(min_clearance, 3)
            updated["constraining_feature"] = constraining_feature

            if min_clearance < TRIMMER_WIDTH:
                updated["equipment_constraint"] = "inaccessible"
                updated["tight_space_flag"] = True
                updated["tight_space_reason"] = f"Corridor {min_clearance:.2f}m - inaccessible"
            elif min_clearance < WALK_BEHIND_WIDTH:
                updated["equipment_constraint"] = "trimmer_only_tight"
                updated["tight_space_flag"] = True
                updated["tight_space_reason"] = f"Corridor {min_clearance:.2f}m - trimmer only"
            elif min_clearance < RIDING_MOWER_WIDTH:
                if updated["equipment_constraint"] == "riding_mower":
                    updated["equipment_constraint"] = "walk_behind"
                updated["tight_space_flag"] = True
                updated["tight_space_reason"] = f"Corridor {min_clearance:.2f}m - walk-behind max"
            else:
                updated["tight_space_flag"] = False
        else:
            updated["tight_space_flag"] = False

        tight_flagged.append(updated)

    return tight_flagged

# ---------------------------------------------------------------------------
# Database schema
# ---------------------------------------------------------------------------

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS osm_raw_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    property_id UUID REFERENCES properties(id) ON DELETE CASCADE,
    queried_at TIMESTAMPTZ DEFAULT NOW(),
    bbox TEXT NOT NULL,
    element_count INTEGER,
    raw_response JSONB,
    overpass_query TEXT,
    query_duration_ms INTEGER
);
CREATE INDEX IF NOT EXISTS idx_osm_raw_cache_property ON osm_raw_cache(property_id);

CREATE TABLE IF NOT EXISTS osm_graph_nodes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    property_id UUID REFERENCES properties(id) ON DELETE CASCADE,
    osm_node_id BIGINT,
    location GEOMETRY(POINT, 4326),
    osm_tags JSONB,
    node_type TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_osm_nodes_property ON osm_graph_nodes(property_id);
CREATE INDEX IF NOT EXISTS idx_osm_nodes_location ON osm_graph_nodes USING GIST(location);

CREATE TABLE IF NOT EXISTS osm_graph_edges (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    property_id UUID REFERENCES properties(id) ON DELETE CASCADE,
    osm_way_id BIGINT,
    geometry GEOMETRY(LINESTRING, 4326),
    length_m NUMERIC(10,3),
    task_type TEXT NOT NULL CHECK (task_type IN ('mow','trim','blow','none')),
    feature_class TEXT,
    equipment_constraint TEXT,
    estimated_width_m NUMERIC(6,3),
    measured_corridor_width_m NUMERIC(6,3),
    tight_space_flag BOOLEAN DEFAULT FALSE,
    tight_space_reason TEXT,
    constraining_feature TEXT,
    classification_reason TEXT,
    needs_review BOOLEAN DEFAULT FALSE,
    manager_override BOOLEAN DEFAULT FALSE,
    manager_override_at TIMESTAMPTZ,
    manager_override_by UUID REFERENCES users(id),
    osm_tags JSONB,
    classified_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_osm_edges_property ON osm_graph_edges(property_id);
CREATE INDEX IF NOT EXISTS idx_osm_edges_task ON osm_graph_edges(property_id, task_type);
CREATE INDEX IF NOT EXISTS idx_osm_edges_geometry ON osm_graph_edges USING GIST(geometry);
CREATE INDEX IF NOT EXISTS idx_osm_edges_review ON osm_graph_edges(needs_review) WHERE needs_review = TRUE;

CREATE TABLE IF NOT EXISTS osm_graph_build_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    property_id UUID REFERENCES properties(id) ON DELETE CASCADE,
    triggered_by UUID REFERENCES users(id),
    build_started_at TIMESTAMPTZ DEFAULT NOW(),
    build_completed_at TIMESTAMPTZ,
    osm_element_count INTEGER,
    edges_created INTEGER,
    nodes_created INTEGER,
    mow_edges INTEGER,
    trim_edges INTEGER,
    blow_edges INTEGER,
    none_edges INTEGER,
    tight_spaces_detected INTEGER,
    needs_review_count INTEGER,
    build_duration_ms INTEGER,
    success BOOLEAN DEFAULT FALSE,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS osm_edge_traversals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    edge_id UUID REFERENCES osm_graph_edges(id),
    job_id UUID REFERENCES jobs(id),
    user_id UUID REFERENCES users(id),
    task_type TEXT,
    equipment_used TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    duration_seconds INTEGER,
    passes_made INTEGER DEFAULT 1,
    quality_flag TEXT,
    gps_track GEOMETRY(LINESTRING, 4326),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_traversals_edge ON osm_edge_traversals(edge_id);
CREATE INDEX IF NOT EXISTS idx_traversals_job ON osm_edge_traversals(job_id);
"""

def ensure_schema(conn):
    cur = conn.cursor()
    cur.execute(CREATE_TABLES_SQL)
    conn.commit()
    cur.close()
    logger.info("OSM graph schema ensured")

# ---------------------------------------------------------------------------
# Main graph builder
# ---------------------------------------------------------------------------

def build_graph_for_property(property_id: str, triggered_by_user_id: str = None) -> dict:
    build_start = time.time()
    conn = get_db()

    try:
        ensure_schema(conn)

        cur = conn.cursor()
        cur.execute("""
            SELECT id, name,
                   ST_AsGeoJSON(boundary) as boundary_geojson,
                   latitude, longitude
            FROM properties WHERE id = %s
        """, (property_id,))
        prop = fetchone_dict(cur)
        cur.close()

        if not prop:
            raise ValueError(f"Property {property_id} not found")

        boundary_geojson = prop.get("boundary_geojson")
        if not boundary_geojson:
            raise ValueError(f"Property {property_id} has no boundary - draw it first")

        if isinstance(boundary_geojson, str):
            boundary_geojson = json.loads(boundary_geojson)

        if boundary_geojson.get("type") == "FeatureCollection":
            features = boundary_geojson.get("features", [])
            if not features:
                raise ValueError("Boundary FeatureCollection has no features")
            geom = shape(features[0]["geometry"])
        elif boundary_geojson.get("type") == "Feature":
            geom = shape(boundary_geojson["geometry"])
        else:
            geom = shape(boundary_geojson)

        if not isinstance(geom, Polygon):
            if hasattr(geom, "geoms"):
                geom = list(geom.geoms)[0]

        boundary_polygon = geom
        logger.info(f"Loaded boundary for '{prop['name']}'")

        query_start = time.time()
        bbox_str = build_bbox_string(boundary_polygon)
        osm_data = query_overpass(boundary_polygon, property_id)
        query_duration_ms = int((time.time() - query_start) * 1000)
        element_count = len(osm_data.get("elements", []))

        cur = conn.cursor()
        cur.execute("""
            INSERT INTO osm_raw_cache
                (property_id, bbox, element_count, raw_response, overpass_query, query_duration_ms)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (property_id, bbox_str, element_count, json.dumps(osm_data),
              OVERPASS_QUERY_TEMPLATE.replace("{bbox}", bbox_str), query_duration_ms))
        conn.commit()
        cur.close()
        logger.info(f"Cached {element_count} OSM elements in {query_duration_ms}ms")

        nodes_by_id, ways = parse_osm_elements(osm_data)
        logger.info(f"Parsed {len(nodes_by_id)} nodes, {len(ways)} ways")

        all_edges = []
        for way in ways:
            coords = way_to_coords(way, nodes_by_id)
            if len(coords) < 2:
                continue

            line_or_poly = coords_to_polygon(coords) if is_closed_way(way) else coords_to_linestring(coords)
            if line_or_poly is None:
                continue
            if not boundary_polygon.intersects(line_or_poly):
                continue

            edges = classify_way(way["tags"], coords, boundary_polygon)
            for edge in edges:
                edge["osm_way_id"] = way["id"]
                edge_coords = edge.get("coords", [])
                if len(edge_coords) >= 2:
                    try:
                        line = LineString(edge_coords)
                        clipped = line.intersection(boundary_polygon)
                        if clipped.is_empty:
                            continue
                        if clipped.geom_type == "MultiLineString":
                            for seg in clipped.geoms:
                                if seg.length > 0:
                                    e2 = dict(edge)
                                    e2["geometry"] = seg
                                    e2["length_m"] = seg.length * 111320
                                    all_edges.append(e2)
                        elif clipped.geom_type == "LineString" and clipped.length > 0:
                            edge["geometry"] = clipped
                            edge["length_m"] = clipped.length * 111320
                            all_edges.append(edge)
                    except Exception:
                        continue
                elif edge.get("geometry"):
                    all_edges.append(edge)

        logger.info(f"Produced {len(all_edges)} classified edge segments")

        all_edges = detect_tight_spaces(all_edges)
        tight_count = sum(1 for e in all_edges if e.get("tight_space_flag"))
        logger.info(f"Tight spaces: {tight_count}")

        cur = conn.cursor()
        cur.execute("DELETE FROM osm_graph_edges WHERE property_id = %s", (property_id,))
        cur.execute("DELETE FROM osm_graph_nodes WHERE property_id = %s", (property_id,))

        mow_count = trim_count = blow_count = none_count = needs_review_count = 0

        for edge in all_edges:
            geom = edge.get("geometry")
            if not geom or not isinstance(geom, LineString):
                continue

            task = edge["task_type"]
            if task == "mow": mow_count += 1
            elif task == "trim": trim_count += 1
            elif task == "blow": blow_count += 1
            else: none_count += 1
            if edge.get("needs_review"): needs_review_count += 1

            try:
                midpoint = geom.interpolate(0.5, normalized=True)
                midpoint_lat = midpoint.y
                midpoint_lng = midpoint.x
            except Exception:
                midpoint_lat = None
                midpoint_lng = None

            cur.execute("""
                INSERT INTO osm_graph_edges (
                    property_id, osm_way_id, geometry, length_m,
                    task_type, feature_class, equipment_constraint,
                    tight_space_flag, tight_space_reason, constraining_feature,
                    measured_corridor_width_m,
                    classification_reason, needs_review, osm_tags, classified_at
                ) VALUES (
                    %s, %s, ST_GeomFromGeoJSON(%s), %s,
                    %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s
                )
            """, (
                property_id, edge.get("osm_way_id"),
                json.dumps(mapping(geom)), edge.get("length_m"),
                edge["task_type"], edge.get("feature_class"),
                edge.get("equipment_constraint"),
                edge.get("tight_space_flag", False),
                edge.get("tight_space_reason"),
                edge.get("constraining_feature"),
                edge.get("measured_corridor_width_m"),
                edge.get("classification_reason"),
                edge.get("needs_review", False),
                json.dumps(edge.get("osm_tags", {})),
                edge.get("classified_at"),
            ))

        node_count = 0
        for node_id, node in nodes_by_id.items():
            tags = node.get("tags", {})
            if not tags:
                continue
            pt = Point(node["lon"], node["lat"])
            if not boundary_polygon.contains(pt):
                continue
            node_type = _classify_node(tags)
            cur.execute("""
                INSERT INTO osm_graph_nodes
                    (property_id, osm_node_id, location, osm_tags, node_type)
                VALUES (%s, %s, ST_GeomFromGeoJSON(%s), %s, %s)
            """, (property_id, node_id, json.dumps(mapping(pt)),
                  json.dumps(tags), node_type))
            node_count += 1

        conn.commit()
        cur.close()
        logger.info(f"Stored {len(all_edges)} edges, {node_count} nodes")

        build_duration_ms = int((time.time() - build_start) * 1000)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO osm_graph_build_log (
                property_id, triggered_by, build_completed_at,
                osm_element_count, edges_created, nodes_created,
                mow_edges, trim_edges, blow_edges, none_edges,
                tight_spaces_detected, needs_review_count,
                build_duration_ms, success
            ) VALUES (%s, %s, NOW(), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, TRUE)
        """, (property_id, triggered_by_user_id, element_count,
              len(all_edges), node_count,
              mow_count, trim_count, blow_count, none_count,
              tight_count, needs_review_count, build_duration_ms))
        conn.commit()
        cur.close()

        summary = {
            "property_id": property_id,
            "property_name": prop["name"],
            "osm_elements": element_count,
            "edges_created": len(all_edges),
            "nodes_created": node_count,
            "mow_edges": mow_count,
            "trim_edges": trim_count,
            "blow_edges": blow_count,
            "none_edges": none_count,
            "tight_spaces": tight_count,
            "needs_review": needs_review_count,
            "build_duration_ms": build_duration_ms,
            "success": True,
        }
        logger.info(f"Graph build complete: {summary}")
        return summary

    except Exception as e:
        import traceback
        logger.error(f"Graph build failed: {e}")
        logger.error(traceback.format_exc())
        try:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO osm_graph_build_log
                    (property_id, triggered_by, success, error_message, build_duration_ms)
                VALUES (%s, %s, FALSE, %s, %s)
            """, (property_id, triggered_by_user_id, str(e),
                  int((time.time() - build_start) * 1000)))
            conn.commit()
            cur.close()
        except Exception:
            pass
        raise
    finally:
        conn.close()


def _classify_node(tags: dict) -> str:
    if tags.get("natural") == "tree": return "tree"
    if tags.get("man_made") in ("utility_pole", "pole"): return "utility_pole"
    if tags.get("amenity") == "fire_hydrant": return "fire_hydrant"
    if tags.get("barrier") == "bollard": return "bollard"
    if tags.get("barrier") == "gate": return "gate"
    if "amenity" in tags: return "amenity"
    return "landmark"


def is_closed_way(way: dict) -> bool:
    nodes = way.get("nodes", [])
    return len(nodes) > 2 and nodes[0] == nodes[-1]


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python osm_graph_builder.py <property_id>")
        sys.exit(1)
    property_id = sys.argv[1]
    print(f"\nBuilding OSM graph for property: {property_id}")
    result = build_graph_for_property(property_id)
    print("\n=== Build Summary ===")
    for k, v in result.items():
        print(f"  {k}: {v}")