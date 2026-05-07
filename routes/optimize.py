"""
LawnRoute Optimizer API Routes
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List
import logging

from optimizer import optimize_job
from replan import replan_job
from coverage import calculate_coverage
from osm_graph_builder import build_graph_for_property
from gps_graph_feedback import process_feedback_for_job

logger = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# Existing optimizer endpoints
# ---------------------------------------------------------------------------

class OptimizeRequest(BaseModel):
    property_id: str
    job_id: str
    crew: List[dict]
    fragments: List[dict]
    mode: str = "balanced"
    equipment: List[dict] = []

class ReplanRequest(BaseModel):
    job_id: str
    property_id: str
    halted_worker_id: str
    halt_type: str
    remaining_fragments: List[dict]
    active_crew: List[dict]

class CoverageRequest(BaseModel):
    job_id: str
    zone_id: str
    breadcrumbs: List[dict]
    zone_boundary: dict
    deck_width_m: float = 0.76

@router.post("/optimize")
async def optimize(req: OptimizeRequest):
    try:
        result = optimize_job(
            property_id=req.property_id,
            job_id=req.job_id,
            crew=req.crew,
            fragments=req.fragments,
            mode=req.mode,
            equipment=req.equipment,
        )
        return result
    except Exception as e:
        logger.error(f"Optimize error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/replan")
async def replan(req: ReplanRequest):
    try:
        result = replan_job(
            job_id=req.job_id,
            property_id=req.property_id,
            halted_worker_id=req.halted_worker_id,
            halt_type=req.halt_type,
            remaining_fragments=req.remaining_fragments,
            active_crew=req.active_crew,
        )
        return result
    except Exception as e:
        logger.error(f"Replan error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/coverage")
async def coverage(req: CoverageRequest):
    try:
        result = calculate_coverage(
            job_id=req.job_id,
            zone_id=req.zone_id,
            breadcrumbs=req.breadcrumbs,
            zone_boundary=req.zone_boundary,
            deck_width_m=req.deck_width_m,
        )
        return result
    except Exception as e:
        logger.error(f"Coverage error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recommend-crew")
async def recommend_crew(total_sqft: float, zone_types: str = "mixed"):
    try:
        open_sqft = total_sqft * 0.6
        trim_sqft = total_sqft * 0.4
        mowers_needed = max(1, round(open_sqft / 40000))
        trimmers_needed = max(1, round(trim_sqft / 8000))
        blowers_needed = max(1, round((mowers_needed + trimmers_needed) / 3))
        total_crew = mowers_needed + trimmers_needed + blowers_needed
        est_duration_min = round((open_sqft / 40000) * 60 + (trim_sqft / 8000) * 45)
        return {
            "recommended_crew": total_crew,
            "mowers": mowers_needed,
            "trimmers": trimmers_needed,
            "blowers": blowers_needed,
            "estimated_duration_minutes": est_duration_min,
            "total_sqft": total_sqft,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------------------------
# OSM Graph Builder endpoints
# ---------------------------------------------------------------------------

class BuildGraphRequest(BaseModel):
    property_id: str
    triggered_by_user_id: Optional[str] = None

@router.post("/build-graph")
async def build_graph(req: BuildGraphRequest):
    try:
        logger.info(f"Building graph for property {req.property_id}")
        result = build_graph_for_property(
            property_id=req.property_id,
            triggered_by_user_id=req.triggered_by_user_id,
        )
        return {"success": True, "message": "Graph built successfully", "summary": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Graph build error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/graph-status/{property_id}")
async def graph_status(property_id: str):
    import psycopg2
    import psycopg2.extras
    import os
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", 5432),
            dbname=os.getenv("DB_NAME", "lawnroute"),
            user=os.getenv("DB_USER", "lawnroute"),
            password=os.getenv("DB_PASSWORD", "lawnroute_dev_2025"),
        )
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM osm_graph_build_log
                WHERE property_id = %s
                ORDER BY build_started_at DESC LIMIT 1
            """, (property_id,))
            build_log = cur.fetchone()
            cur.execute("""
                SELECT task_type, COUNT(*) as count,
                       AVG(length_m) as avg_length_m,
                       SUM(length_m) as total_length_m,
                       COUNT(*) FILTER (WHERE tight_space_flag) as tight_spaces,
                       COUNT(*) FILTER (WHERE needs_review) as needs_review
                FROM osm_graph_edges
                WHERE property_id = %s GROUP BY task_type
            """, (property_id,))
            edge_stats = cur.fetchall()
        conn.close()
        return {
            "property_id": property_id,
            "has_graph": build_log is not None,
            "latest_build": dict(build_log) if build_log else None,
            "edge_stats": [dict(r) for r in edge_stats],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/graph-edges/{property_id}")
async def graph_edges(property_id: str, task_type: Optional[str] = None):
    import psycopg2
    import psycopg2.extras
    import os
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", 5432),
            dbname=os.getenv("DB_NAME", "lawnroute"),
            user=os.getenv("DB_USER", "lawnroute"),
            password=os.getenv("DB_PASSWORD", "lawnroute_dev_2025"),
        )
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if task_type:
                cur.execute("""
                    SELECT id, task_type, feature_class, equipment_constraint,
                           estimated_width_m, tight_space_flag, needs_review,
                           classification_reason,
                           ST_AsGeoJSON(geometry) as geometry_geojson, length_m
                    FROM osm_graph_edges
                    WHERE property_id = %s AND task_type = %s
                    ORDER BY task_type, length_m DESC
                """, (property_id, task_type))
            else:
                cur.execute("""
                    SELECT id, task_type, feature_class, equipment_constraint,
                           estimated_width_m, tight_space_flag, needs_review,
                           classification_reason,
                           ST_AsGeoJSON(geometry) as geometry_geojson, length_m
                    FROM osm_graph_edges
                    WHERE property_id = %s AND task_type != 'none'
                    ORDER BY task_type, length_m DESC
                """, (property_id,))
            edges = cur.fetchall()
        conn.close()
        return {"property_id": property_id, "edge_count": len(edges),
                "edges": [dict(e) for e in edges]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.patch("/graph-edges/{edge_id}/override")
async def override_edge(edge_id: str, task_type: str, user_id: str):
    import psycopg2
    import os
    valid_types = ["mow", "trim", "blow", "none"]
    if task_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"task_type must be one of {valid_types}")
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", 5432),
            dbname=os.getenv("DB_NAME", "lawnroute"),
            user=os.getenv("DB_USER", "lawnroute"),
            password=os.getenv("DB_PASSWORD", "lawnroute_dev_2025"),
        )
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE osm_graph_edges SET
                    task_type = %s, manager_override = TRUE,
                    manager_override_at = NOW(), manager_override_by = %s,
                    needs_review = FALSE, updated_at = NOW()
                WHERE id = %s
            """, (task_type, user_id, edge_id))
        conn.commit()
        conn.close()
        return {"success": True, "edge_id": edge_id, "new_task_type": task_type}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------------------------
# GPS Feedback endpoints
# ---------------------------------------------------------------------------

class FeedbackRequest(BaseModel):
    job_id: str

@router.post("/process-feedback")
async def process_feedback(req: FeedbackRequest, background_tasks: BackgroundTasks):
    """
    Process GPS feedback for a completed job.
    Triggered automatically when a job is marked complete.
    Runs in background so it doesn't block the job completion response.
    """
    try:
        background_tasks.add_task(process_feedback_for_job, req.job_id)
        return {
            "success": True,
            "message": f"GPS feedback processing started for job {req.job_id}",
            "job_id": req.job_id,
        }
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/feedback-status/{property_id}")
async def feedback_status(property_id: str):
    """Get GPS feedback summary for a property — corrections detected and applied."""
    import psycopg2
    import psycopg2.extras
    import os
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", 5432),
            dbname=os.getenv("DB_NAME", "lawnroute"),
            user=os.getenv("DB_USER", "lawnroute"),
            password=os.getenv("DB_PASSWORD", "lawnroute_dev_2025"),
        )
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT correction_type, COUNT(*) as count,
                       AVG(confidence_score) as avg_confidence,
                       SUM(CASE WHEN auto_applied THEN 1 ELSE 0 END) as auto_applied,
                       SUM(CASE WHEN manager_reviewed THEN 1 ELSE 0 END) as reviewed
                FROM graph_corrections
                WHERE property_id = %s
                GROUP BY correction_type
            """, (property_id,))
            corrections = cur.fetchall()

            cur.execute("""
                SELECT visit_count, last_updated
                FROM property_gps_coverage
                WHERE property_id = %s
            """, (property_id,))
            coverage = cur.fetchone()

        conn.close()
        return {
            "property_id": property_id,
            "visit_count": coverage["visit_count"] if coverage else 0,
            "corrections": [dict(c) for c in corrections],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))