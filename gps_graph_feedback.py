"""
LawnRoute GPS-to-Graph Feedback Loop
Session 2 of Smart Site Mapping

After each job completes, this module:
1. Analyzes all crew GPS breadcrumb tracks by role (mow/trim/blow)
2. Compares actual paths against existing OSM graph edges
3. Detects new edges the crew walked that OSM missed
4. Detects existing edges no crew ever walked (likely incorrect)
5. Refines mow zone boundaries from mower GPS coverage
6. Adds missing blow zones from blower GPS paths
7. Stores all corrections with confidence scores
8. Auto-applies high-confidence corrections to the graph
9. Flags low-confidence corrections for next visit confirmation

Data collection: every GPS point, every detection, every correction
is stored permanently for training future classifiers.
"""

import os
import json
import time
import logging
import psycopg2
import psycopg2.extras
from datetime import datetime, timezone
from typing import Optional
from shapely.geometry import (
    LineString, Polygon, Point, MultiLineString,
    mapping, shape
)
from shapely.ops import unary_union, snap
try:
    from shapely.ops import offset_curve
except ImportError:
    from shapely import offset_curve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Distance thresholds (meters)
SNAP_DISTANCE_M = 3.0          # GPS points within 3m of an edge = on that edge
NEW_EDGE_MIN_LENGTH_M = 2.0    # Minimum length to consider a new detected edge
CORRIDOR_BUFFER_M = 1.5        # Buffer around GPS path to detect covered area
MOW_PASS_WIDTH_M = 0.76        # Default mower deck width for coverage calculation

# Confidence thresholds
HIGH_CONFIDENCE = 0.80         # Auto-apply corrections above this score
LOW_CONFIDENCE = 0.40          # Flag for review below this score

# Degrees per meter
DEG_PER_METER = 1.0 / 111320

# ---------------------------------------------------------------------------
# Database connection
# ---------------------------------------------------------------------------

def get_db():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", 5432),
        dbname=os.getenv("DB_NAME", "lawnroute"),
        user=os.getenv("DB_USER", "lawnroute"),
        password=os.getenv("DB_PASSWORD", "lawnroute_dev_2025"),
    )

# ---------------------------------------------------------------------------
# Database schema for feedback data
# ---------------------------------------------------------------------------

CREATE_FEEDBACK_TABLES_SQL = """
-- GPS tracks per worker per job (raw breadcrumbs organized by role)
CREATE TABLE IF NOT EXISTS gps_role_tracks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID REFERENCES jobs(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id),
    property_id UUID REFERENCES properties(id),
    role TEXT NOT NULL CHECK (role IN ('mow', 'trim', 'blow')),
    track GEOMETRY(LINESTRING, 4326),
    track_points INTEGER,
    length_m NUMERIC(10,3),
    duration_seconds INTEGER,
    recorded_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_gps_tracks_job ON gps_role_tracks(job_id);
CREATE INDEX IF NOT EXISTS idx_gps_tracks_property ON gps_role_tracks(property_id);
CREATE INDEX IF NOT EXISTS idx_gps_tracks_role ON gps_role_tracks(property_id, role);
CREATE INDEX IF NOT EXISTS idx_gps_tracks_geom ON gps_role_tracks USING GIST(track);

-- Detected edge corrections from GPS analysis
CREATE TABLE IF NOT EXISTS graph_corrections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    property_id UUID REFERENCES properties(id) ON DELETE CASCADE,
    job_id UUID REFERENCES jobs(id),
    correction_type TEXT NOT NULL CHECK (correction_type IN (
        'new_edge_detected',      -- GPS walked path that has no OSM edge
        'edge_unwalked',          -- OSM edge that no crew ever walked
        'edge_type_mismatch',     -- Crew role doesn't match edge classification
        'mow_zone_refined',       -- Mower GPS refines mow zone boundary
        'blow_zone_detected',     -- Blower GPS detects new blow zone
        'tight_space_confirmed',  -- Crew confirmed tight space by using trimmer
        'equipment_override'      -- Crew used different equipment than recommended
    )),
    detected_geometry GEOMETRY(LINESTRING, 4326),
    affected_edge_id UUID REFERENCES osm_graph_edges(id),
    suggested_task_type TEXT,
    suggested_equipment TEXT,
    confidence_score NUMERIC(4,3),
    visit_count INTEGER DEFAULT 1,   -- How many visits confirmed this
    auto_applied BOOLEAN DEFAULT FALSE,
    auto_applied_at TIMESTAMPTZ,
    manager_reviewed BOOLEAN DEFAULT FALSE,
    manager_approved BOOLEAN,
    manager_reviewed_at TIMESTAMPTZ,
    detection_metadata JSONB,        -- Full detection reasoning stored
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_corrections_property ON graph_corrections(property_id);
CREATE INDEX IF NOT EXISTS idx_corrections_type ON graph_corrections(correction_type);
CREATE INDEX IF NOT EXISTS idx_corrections_confidence ON graph_corrections(confidence_score);
CREATE INDEX IF NOT EXISTS idx_corrections_pending 
    ON graph_corrections(property_id, auto_applied, manager_reviewed)
    WHERE auto_applied = FALSE AND manager_reviewed = FALSE;
CREATE INDEX IF NOT EXISTS idx_corrections_geom ON graph_corrections USING GIST(detected_geometry);

-- Mow zone coverage accumulated per property
CREATE TABLE IF NOT EXISTS mow_zone_coverage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    property_id UUID REFERENCES properties(id) ON DELETE CASCADE,
    job_id UUID REFERENCES jobs(id),
    coverage_polygon GEOMETRY(POLYGON, 4326),
    area_sqm NUMERIC(12,3),
    pass_count INTEGER DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_mow_coverage_property ON mow_zone_coverage(property_id);
CREATE INDEX IF NOT EXISTS idx_mow_coverage_geom ON mow_zone_coverage USING GIST(coverage_polygon);

-- Accumulated GPS coverage per property (union of all visits)
CREATE TABLE IF NOT EXISTS property_gps_coverage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    property_id UUID REFERENCES properties(id) ON DELETE CASCADE UNIQUE,
    trim_coverage GEOMETRY(MULTILINESTRING, 4326),  -- All trim paths ever walked
    mow_coverage GEOMETRY(MULTIPOLYGON, 4326),      -- All mow areas ever covered
    blow_coverage GEOMETRY(MULTILINESTRING, 4326),  -- All blow paths ever walked
    visit_count INTEGER DEFAULT 0,
    last_updated TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_property_coverage ON property_gps_coverage(property_id);

-- Edge confidence scores (updated after each visit)
CREATE TABLE IF NOT EXISTS edge_confidence (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    edge_id UUID REFERENCES osm_graph_edges(id) ON DELETE CASCADE UNIQUE,
    property_id UUID REFERENCES properties(id),
    confirmed_visits INTEGER DEFAULT 0,   -- Times crew walked this edge as classified
    contradicted_visits INTEGER DEFAULT 0, -- Times crew contradicted classification
    last_walked_at TIMESTAMPTZ,
    confidence_score NUMERIC(4,3) DEFAULT 0.5,  -- Starts at 0.5, moves toward 1 or 0
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_edge_confidence_edge ON edge_confidence(edge_id);
CREATE INDEX IF NOT EXISTS idx_edge_confidence_property ON edge_confidence(property_id);
"""

def ensure_feedback_schema(conn):
    with conn.cursor() as cur:
        cur.execute(CREATE_FEEDBACK_TABLES_SQL)
    conn.commit()
    logger.info("GPS feedback schema ensured")

# ---------------------------------------------------------------------------
# GPS track loading
# ---------------------------------------------------------------------------

def load_job_breadcrumbs(conn, job_id: str) -> dict:
    """
    Load all GPS breadcrumbs for a job, organized by worker and role.
    Returns dict: {user_id: {role: [points]}}
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT 
                gb.user_id,
                u.name as user_name,
                jc.primary_role as role,
                gb.latitude,
                gb.longitude,
                gb.recorded_at,
                gb.accuracy_m
            FROM gps_breadcrumbs gb
            JOIN users u ON gb.user_id = u.id
            LEFT JOIN job_crew jc ON jc.job_id = gb.job_id AND jc.user_id = gb.user_id
            WHERE gb.job_id = %s
            ORDER BY gb.user_id, gb.recorded_at
        """, (job_id,))
        rows = cur.fetchall()

    tracks = {}
    for row in rows:
        uid = str(row["user_id"])
        role = row.get("role") or "trim"  # Default to trim if role unknown
        if uid not in tracks:
            tracks[uid] = {"name": row["user_name"], "points_by_role": {}}
        if role not in tracks[uid]["points_by_role"]:
            tracks[uid]["points_by_role"][role] = []
        tracks[uid]["points_by_role"][role].append({
            "lon": float(row["longitude"]),
            "lat": float(row["latitude"]),
            "recorded_at": row["recorded_at"],
            "accuracy_m": float(row.get("accuracy_m") or 5.0),
        })

    return tracks


def load_job_task_segments(conn, job_id: str) -> list:
    """
    Load job task segments — periods when a worker was performing a specific role.
    This is more accurate than breadcrumbs alone because it captures role switches.
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT 
                jt.user_id,
                jt.task_type,
                jt.started_at,
                jt.completed_at,
                u.name as user_name
            FROM job_tasks jt
            JOIN users u ON jt.user_id = u.id
            WHERE jt.job_id = %s AND jt.completed_at IS NOT NULL
            ORDER BY jt.user_id, jt.started_at
        """, (job_id,))
        return cur.fetchall()


def build_role_tracks(breadcrumbs: dict, task_segments: list) -> dict:
    """
    Combine breadcrumbs with task segment timing to assign each GPS point
    to the correct role (mow/trim/blow) at the time it was recorded.
    
    Returns: {role: [LineString, ...]}
    """
    role_points = {"mow": [], "trim": [], "blow": []}

    for user_id, user_data in breadcrumbs.items():
        points_by_role = user_data.get("points_by_role", {})

        # Find task segments for this user
        user_segments = [s for s in task_segments
                        if str(s["user_id"]) == user_id]

        if not user_segments:
            # No task segments — use role from breadcrumb data directly
            for role, points in points_by_role.items():
                if role in role_points:
                    role_points[role].extend(points)
            continue

        # Assign each breadcrumb to a role based on timing
        all_points = []
        for role, points in points_by_role.items():
            for p in points:
                all_points.append({**p, "default_role": role})

        all_points.sort(key=lambda p: p["recorded_at"])

        for point in all_points:
            assigned_role = point.get("default_role", "trim")
            pt_time = point["recorded_at"]

            for seg in user_segments:
                if seg["started_at"] <= pt_time <= seg["completed_at"]:
                    assigned_role = seg["task_type"]
                    break

            if assigned_role in role_points:
                role_points[assigned_role].append(point)

    return role_points


def points_to_linestring(points: list) -> Optional[LineString]:
    """Convert list of {lon, lat} dicts to a Shapely LineString."""
    coords = [(p["lon"], p["lat"]) for p in points
              if p.get("lon") and p.get("lat")]
    if len(coords) < 2:
        return None
    try:
        line = LineString(coords)
        return line if line.is_valid and not line.is_empty else None
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Edge comparison — did crew walk this edge?
# ---------------------------------------------------------------------------

def load_property_edges(conn, property_id: str) -> list:
    """Load all active graph edges for a property."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT id, task_type, feature_class, equipment_constraint,
                   ST_AsGeoJSON(geometry) as geojson,
                   length_m, tight_space_flag, needs_review
            FROM osm_graph_edges
            WHERE property_id = %s AND task_type != 'none'
        """, (property_id,))
        rows = cur.fetchall()

    edges = []
    for row in rows:
        try:
            geom = shape(json.loads(row["geojson"]))
            edges.append({
                "id": str(row["id"]),
                "task_type": row["task_type"],
                "feature_class": row["feature_class"],
                "equipment_constraint": row["equipment_constraint"],
                "geometry": geom,
                "length_m": float(row["length_m"] or 0),
                "tight_space_flag": row["tight_space_flag"],
                "needs_review": row["needs_review"],
            })
        except Exception as e:
            logger.warning(f"Could not parse edge geometry: {e}")
    return edges


def gps_track_near_edge(track: LineString, edge_geom: LineString,
                         threshold_m: float = SNAP_DISTANCE_M) -> bool:
    """Check if a GPS track passes within threshold_m of an edge."""
    threshold_deg = threshold_m * DEG_PER_METER
    try:
        return track.distance(edge_geom) <= threshold_deg
    except Exception:
        return False


def fraction_of_edge_covered(track: LineString, edge_geom: LineString,
                               threshold_m: float = SNAP_DISTANCE_M) -> float:
    """
    Calculate what fraction of an edge was covered by the GPS track.
    Samples points along the edge and checks proximity to track.
    """
    if edge_geom.length == 0:
        return 0.0

    threshold_deg = threshold_m * DEG_PER_METER
    sample_count = max(5, int(edge_geom.length / (2 * DEG_PER_METER)))
    covered = 0

    for i in range(sample_count + 1):
        fraction = i / sample_count
        pt = edge_geom.interpolate(fraction, normalized=True)
        if track.distance(pt) <= threshold_deg:
            covered += 1

    return covered / (sample_count + 1)

# ---------------------------------------------------------------------------
# New edge detection
# ---------------------------------------------------------------------------

def detect_new_edges_from_track(track: LineString, existing_edges: list,
                                  role: str, boundary: Polygon,
                                  threshold_m: float = SNAP_DISTANCE_M) -> list:
    """
    Find portions of a GPS track that don't correspond to any existing edge.
    These are candidate new edges that OSM missed.
    
    Returns list of LineString segments representing unmatched track portions.
    """
    if not track or not boundary:
        return []

    threshold_deg = threshold_m * DEG_PER_METER

    # Build union of all existing edge geometries (buffered)
    edge_buffers = []
    for edge in existing_edges:
        if edge["task_type"] == role or role == "trim":
            try:
                buffered = edge["geometry"].buffer(threshold_deg)
                edge_buffers.append(buffered)
            except Exception:
                continue

    if not edge_buffers:
        # No existing edges at all — entire track is new
        clipped = track.intersection(boundary)
        if not clipped.is_empty and clipped.length * 111320 > NEW_EDGE_MIN_LENGTH_M:
            return [clipped] if clipped.geom_type == "LineString" else list(clipped.geoms)
        return []

    covered_area = unary_union(edge_buffers)

    # Find track portions outside covered area
    try:
        uncovered = track.difference(covered_area)
        clipped = uncovered.intersection(boundary)

        if clipped.is_empty:
            return []

        segments = []
        if clipped.geom_type == "LineString":
            segs = [clipped]
        elif clipped.geom_type == "MultiLineString":
            segs = list(clipped.geoms)
        else:
            segs = []

        for seg in segs:
            length_m = seg.length * 111320
            if length_m >= NEW_EDGE_MIN_LENGTH_M:
                segments.append(seg)

        return segments
    except Exception as e:
        logger.warning(f"Error detecting new edges: {e}")
        return []

# ---------------------------------------------------------------------------
# Mow zone refinement
# ---------------------------------------------------------------------------

def build_mow_coverage_polygon(mow_track: LineString,
                                deck_width_m: float = MOW_PASS_WIDTH_M) -> Optional[Polygon]:
    """
    Convert a mower GPS track into a coverage polygon.
    The polygon represents the area actually mowed.
    """
    if not mow_track or mow_track.length == 0:
        return None

    buffer_deg = (deck_width_m / 2) * DEG_PER_METER
    try:
        poly = mow_track.buffer(buffer_deg, cap_style=2)
        return poly if poly.is_valid else None
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def calculate_correction_confidence(
    visit_count: int,
    track_length_m: float,
    fraction_covered: float,
    role_matches_edge: bool,
    is_repeated: bool,
) -> float:
    """
    Calculate confidence score for a detected correction.
    
    Factors:
    - More visits = higher confidence
    - Longer track segment = higher confidence
    - Higher fraction of edge covered = higher confidence
    - Role matches edge classification = higher confidence
    - Seen on multiple visits = significantly higher confidence
    """
    score = 0.0

    # Visit count (max 0.30)
    score += min(0.30, visit_count * 0.10)

    # Track length (max 0.20)
    score += min(0.20, (track_length_m / 50.0) * 0.20)

    # Fraction covered (max 0.25)
    score += fraction_covered * 0.25

    # Role match (0.15)
    if role_matches_edge:
        score += 0.15

    # Repeated across multiple visits (0.10)
    if is_repeated:
        score += 0.10

    return min(1.0, round(score, 3))

# ---------------------------------------------------------------------------
# Main feedback processor
# ---------------------------------------------------------------------------

def process_job_feedback(job_id: str, property_id: str,
                          triggered_by_user_id: str = None) -> dict:
    """
    Main entry point. Process GPS feedback for a completed job.
    
    1. Load breadcrumbs and task segments
    2. Build role-specific GPS tracks
    3. Compare against existing graph edges
    4. Detect new edges, unwalked edges, zone refinements
    5. Store corrections with confidence scores
    6. Auto-apply high-confidence corrections
    7. Update cumulative property GPS coverage
    """
    start_time = time.time()
    conn = get_db()

    try:
        ensure_feedback_schema(conn)

        # Load property boundary
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT ST_AsGeoJSON(boundary) as boundary_geojson
                FROM properties WHERE id = %s
            """, (property_id,))
            prop = cur.fetchone()

        if not prop or not prop["boundary_geojson"]:
            raise ValueError(f"Property {property_id} has no boundary")

        boundary = shape(json.loads(prop["boundary_geojson"]))

        # Load GPS data
        breadcrumbs = load_job_breadcrumbs(conn, job_id)
        task_segments = load_job_task_segments(conn, job_id)

        if not breadcrumbs:
            logger.info(f"No GPS breadcrumbs found for job {job_id}")
            return {"success": True, "message": "No GPS data to process", "corrections": 0}

        logger.info(f"Loaded breadcrumbs for {len(breadcrumbs)} workers")

        # Build role tracks
        role_points = build_role_tracks(breadcrumbs, task_segments)

        role_tracks = {}
        for role, points in role_points.items():
            if points:
                track = points_to_linestring(points)
                if track:
                    role_tracks[role] = track
                    logger.info(f"{role} track: {len(points)} points, "
                               f"{track.length * 111320:.1f}m")

        if not role_tracks:
            return {"success": True, "message": "Could not build tracks from GPS data", "corrections": 0}

        # Store role tracks
        with conn.cursor() as cur:
            for role, track in role_tracks.items():
                length_m = track.length * 111320
                cur.execute("""
                    INSERT INTO gps_role_tracks
                        (job_id, property_id, role, track, track_points, length_m)
                    VALUES (%s, %s, %s, ST_GeomFromGeoJSON(%s), %s, %s)
                """, (job_id, property_id, role,
                      json.dumps(mapping(track)),
                      len(role_points[role]),
                      length_m))
        conn.commit()

        # Load existing graph edges
        existing_edges = load_property_edges(conn, property_id)
        logger.info(f"Comparing against {len(existing_edges)} existing edges")

        corrections = []

        # --- Analysis 1: Check each existing edge against crew tracks ---
        for edge in existing_edges:
            edge_geom = edge["geometry"]
            edge_role = edge["task_type"]
            edge_id = edge["id"]

            # Check if the matching role track walked this edge
            matching_track = role_tracks.get(edge_role)

            if matching_track:
                fraction = fraction_of_edge_covered(matching_track, edge_geom)

                # Update edge confidence
                _update_edge_confidence(conn, edge_id, property_id,
                                         confirmed=(fraction > 0.5))

                if fraction < 0.2:
                    # Edge was not walked — possible misclassification
                    # Check if a DIFFERENT role walked it
                    wrong_role_walked = False
                    for other_role, other_track in role_tracks.items():
                        if other_role != edge_role:
                            other_fraction = fraction_of_edge_covered(
                                other_track, edge_geom)
                            if other_fraction > 0.5:
                                wrong_role_walked = True
                                # Edge type mismatch detected
                                confidence = calculate_correction_confidence(
                                    visit_count=1,
                                    track_length_m=edge["length_m"],
                                    fraction_covered=other_fraction,
                                    role_matches_edge=False,
                                    is_repeated=False,
                                )
                                corrections.append({
                                    "type": "edge_type_mismatch",
                                    "edge_id": edge_id,
                                    "geometry": edge_geom,
                                    "suggested_task_type": other_role,
                                    "confidence": confidence,
                                    "metadata": {
                                        "original_type": edge_role,
                                        "walked_by_role": other_role,
                                        "fraction_covered": other_fraction,
                                    }
                                })
                                break

                    if not wrong_role_walked:
                        # Truly unwalked edge
                        confidence = calculate_correction_confidence(
                            visit_count=1,
                            track_length_m=edge["length_m"],
                            fraction_covered=fraction,
                            role_matches_edge=True,
                            is_repeated=False,
                        )
                        corrections.append({
                            "type": "edge_unwalked",
                            "edge_id": edge_id,
                            "geometry": edge_geom,
                            "suggested_task_type": None,
                            "confidence": confidence,
                            "metadata": {
                                "fraction_walked": fraction,
                                "edge_length_m": edge["length_m"],
                            }
                        })

        # --- Analysis 2: Detect new edges from GPS tracks ---
        for role, track in role_tracks.items():
            new_segments = detect_new_edges_from_track(
                track, existing_edges, role, boundary)

            for seg in new_segments:
                length_m = seg.length * 111320
                confidence = calculate_correction_confidence(
                    visit_count=1,
                    track_length_m=length_m,
                    fraction_covered=1.0,
                    role_matches_edge=True,
                    is_repeated=False,
                )
                corrections.append({
                    "type": "new_edge_detected",
                    "edge_id": None,
                    "geometry": seg,
                    "suggested_task_type": role,
                    "suggested_equipment": _default_equipment(role),
                    "confidence": confidence,
                    "metadata": {
                        "detected_role": role,
                        "segment_length_m": length_m,
                    }
                })

        # --- Analysis 3: Mow coverage polygon ---
        if "mow" in role_tracks:
            mow_poly = build_mow_coverage_polygon(role_tracks["mow"])
            if mow_poly:
                clipped = mow_poly.intersection(boundary)
                if not clipped.is_empty:
                    area_sqm = clipped.area * 1e10
                    with conn.cursor() as cur:
                        cur.execute("""
                            INSERT INTO mow_zone_coverage
                                (property_id, job_id, coverage_polygon, area_sqm)
                            VALUES (%s, %s, ST_GeomFromGeoJSON(%s), %s)
                        """, (property_id, job_id,
                              json.dumps(mapping(clipped)), area_sqm))
                    conn.commit()
                    logger.info(f"Mow coverage polygon stored: {area_sqm:.1f} sqm")

        # --- Analysis 4: Blow zone detection ---
        if "blow" in role_tracks:
            blow_track = role_tracks["blow"]
            new_blow_segs = detect_new_edges_from_track(
                blow_track, existing_edges, "blow", boundary)
            for seg in new_blow_segs:
                corrections.append({
                    "type": "blow_zone_detected",
                    "edge_id": None,
                    "geometry": seg,
                    "suggested_task_type": "blow",
                    "suggested_equipment": "blower",
                    "confidence": 0.75,
                    "metadata": {"segment_length_m": seg.length * 111320}
                })

        # --- Store all corrections ---
        auto_applied_count = 0
        stored_corrections = 0

        with conn.cursor() as cur:
            for correction in corrections:
                geom = correction.get("geometry")
                if not geom:
                    continue

                confidence = correction["confidence"]

                # Check if this correction already exists (repeated visit)
                cur.execute("""
                    SELECT id, visit_count, confidence_score
                    FROM graph_corrections
                    WHERE property_id = %s
                    AND correction_type = %s
                    AND ST_Distance(detected_geometry,
                        ST_GeomFromGeoJSON(%s)) < %s
                    LIMIT 1
                """, (property_id, correction["type"],
                      json.dumps(mapping(geom)),
                      SNAP_DISTANCE_M * DEG_PER_METER))

                existing = cur.fetchone()

                if existing:
                    # Update existing correction — increase visit count and confidence
                    new_visits = existing[1] + 1
                    new_confidence = min(1.0, existing[2] + 0.15)
                    cur.execute("""
                        UPDATE graph_corrections
                        SET visit_count = %s,
                            confidence_score = %s,
                            detection_metadata = detection_metadata || %s
                        WHERE id = %s
                    """, (new_visits, new_confidence,
                          json.dumps({"last_seen_job": job_id}),
                          existing[0]))
                    confidence = new_confidence
                else:
                    # Insert new correction
                    cur.execute("""
                        INSERT INTO graph_corrections (
                            property_id, job_id, correction_type,
                            detected_geometry, affected_edge_id,
                            suggested_task_type, suggested_equipment,
                            confidence_score, detection_metadata
                        ) VALUES (%s, %s, %s, ST_GeomFromGeoJSON(%s), %s,
                                  %s, %s, %s, %s)
                    """, (
                        property_id, job_id, correction["type"],
                        json.dumps(mapping(geom)),
                        correction.get("edge_id"),
                        correction.get("suggested_task_type"),
                        correction.get("suggested_equipment"),
                        confidence,
                        json.dumps(correction.get("metadata", {})),
                    ))
                    stored_corrections += 1

                # Auto-apply high-confidence new edge detections
                if (confidence >= HIGH_CONFIDENCE and
                        correction["type"] == "new_edge_detected" and
                        not existing):
                    _auto_apply_new_edge(conn, property_id, correction)
                    auto_applied_count += 1

        conn.commit()

        # Update cumulative property GPS coverage
        _update_property_coverage(conn, property_id, role_tracks)

        duration_ms = int((time.time() - start_time) * 1000)

        summary = {
            "job_id": job_id,
            "property_id": property_id,
            "workers_tracked": len(breadcrumbs),
            "role_tracks_built": list(role_tracks.keys()),
            "existing_edges_checked": len(existing_edges),
            "corrections_detected": len(corrections),
            "corrections_stored": stored_corrections,
            "auto_applied": auto_applied_count,
            "duration_ms": duration_ms,
            "success": True,
        }

        logger.info(f"GPS feedback complete: {summary}")
        return summary

    except Exception as e:
        logger.error(f"GPS feedback failed: {e}")
        raise
    finally:
        conn.close()


def _update_edge_confidence(conn, edge_id: str, property_id: str,
                              confirmed: bool):
    """Update running confidence score for an edge based on crew behavior."""
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO edge_confidence (edge_id, property_id, confirmed_visits,
                contradicted_visits, last_walked_at, confidence_score)
            VALUES (%s, %s,
                CASE WHEN %s THEN 1 ELSE 0 END,
                CASE WHEN %s THEN 0 ELSE 1 END,
                CASE WHEN %s THEN NOW() ELSE NULL END,
                CASE WHEN %s THEN 0.6 ELSE 0.4 END)
            ON CONFLICT (edge_id) DO UPDATE SET
                confirmed_visits = edge_confidence.confirmed_visits +
                    CASE WHEN %s THEN 1 ELSE 0 END,
                contradicted_visits = edge_confidence.contradicted_visits +
                    CASE WHEN %s THEN 0 ELSE 1 END,
                last_walked_at = CASE WHEN %s THEN NOW()
                    ELSE edge_confidence.last_walked_at END,
                confidence_score = LEAST(1.0, GREATEST(0.0,
                    (edge_confidence.confirmed_visits::float /
                     NULLIF(edge_confidence.confirmed_visits +
                            edge_confidence.contradicted_visits + 1, 0)))),
                updated_at = NOW()
        """, (edge_id, property_id,
              confirmed, confirmed, confirmed, confirmed,
              confirmed, confirmed, confirmed))


def _auto_apply_new_edge(conn, property_id: str, correction: dict):
    """Auto-apply a high-confidence new edge detection to the graph."""
    geom = correction["geometry"]
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO osm_graph_edges (
                property_id, geometry, length_m,
                task_type, feature_class, equipment_constraint,
                classification_reason, needs_review
            ) VALUES (
                %s, ST_GeomFromGeoJSON(%s), %s,
                %s, %s, %s,
                %s, FALSE
            )
        """, (
            property_id,
            json.dumps(mapping(geom)),
            geom.length * 111320,
            correction["suggested_task_type"],
            "gps_detected",
            correction.get("suggested_equipment", "trimmer"),
            f"Auto-detected from GPS track (confidence: {correction['confidence']:.2f})",
        ))

        # Mark correction as applied
        cur.execute("""
            UPDATE graph_corrections SET
                auto_applied = TRUE,
                auto_applied_at = NOW()
            WHERE property_id = %s
            AND correction_type = 'new_edge_detected'
            AND ST_Distance(detected_geometry, ST_GeomFromGeoJSON(%s)) < %s
        """, (property_id, json.dumps(mapping(geom)),
              SNAP_DISTANCE_M * DEG_PER_METER))


def _update_property_coverage(conn, property_id: str, role_tracks: dict):
    """Update cumulative GPS coverage for the property."""
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO property_gps_coverage
                (property_id, visit_count, last_updated)
            VALUES (%s, 1, NOW())
            ON CONFLICT (property_id) DO UPDATE SET
                visit_count = property_gps_coverage.visit_count + 1,
                last_updated = NOW()
        """, (property_id,))
    conn.commit()


def _default_equipment(role: str) -> str:
    defaults = {"mow": "walk_behind", "trim": "trimmer", "blow": "blower"}
    return defaults.get(role, "trimmer")


# ---------------------------------------------------------------------------
# API helper — called from FastAPI route
# ---------------------------------------------------------------------------

def process_feedback_for_job(job_id: str) -> dict:
    """
    Load property_id from job record and run feedback processing.
    Called by the FastAPI endpoint after job completion.
    """
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT property_id, company_id FROM jobs WHERE id = %s
            """, (job_id,))
            job = cur.fetchone()

        if not job:
            raise ValueError(f"Job {job_id} not found")

        return process_job_feedback(
            job_id=job_id,
            property_id=str(job["property_id"]),
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CLI for testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python gps_graph_feedback.py <job_id> <property_id>")
        sys.exit(1)

    job_id = sys.argv[1]
    property_id = sys.argv[2]

    print(f"\nProcessing GPS feedback for job: {job_id}")
    print(f"Property: {property_id}")
    result = process_job_feedback(job_id, property_id)
    print("\n=== Feedback Summary ===")
    for k, v in result.items():
        print(f"  {k}: {v}")