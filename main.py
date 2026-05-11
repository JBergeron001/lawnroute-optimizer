"""
LawnRoute Task Optimizer — main.py

Time estimation philosophy:
  - Mowing speed is NOT the theoretical max speed of the machine.
    It is the real-world effective speed accounting for:
      • Turns at end of each row (180° turn + deceleration + acceleration)
      • Boundary passes (first mow the perimeter, then strip the interior)
      • Row overlap (6-12 inches between passes to avoid missed strips)
      • Gate entries, repositioning around obstacles
      • Equipment efficiency losses (engine load, terrain variation)
    These factors reduce effective speed to ~55-65% of theoretical max.

  - The complexity_factor on each zone is the primary mechanism for
    capturing site-specific difficulty.

  - Property type complexity is handled HERE in the optimizer, not in
    the API layer. The API sends raw area_sqft — no pre-multiplication.

Three optimization modes — each is an independent scenario:
  Lean Crew  : Always exactly 2 crew. Shows minimum labor cost option.
  Optimal    : Acreage-based crew floor + algorithm finds sweet spot
               where adding one more person stops saving meaningful time.
               This is the recommended staffing level for the property.
  Max Speed  : Optimal crew size + 2 extra. Aggressive but realistic.
               Capped at total available crew.

The pre-assigned crew task breakdown is calculated separately and
returned alongside the 3 scenarios.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import time
import math

app = FastAPI(title="LawnRoute Optimizer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Models ───────────────────────────────────────────────────────────────────

class Zone(BaseModel):
    id: str
    label: str
    zone_type: str
    area_sqft: float = 0.0
    slope_grade: float = 0.0
    equipment_restriction: Optional[str] = None
    perimeter_ft: Optional[float] = None
    linear_ft: Optional[float] = None
    complexity_factor: Optional[float] = 1.0
    surface_type: Optional[str] = None
    obstacle_density: Optional[float] = 0.0
    avg_actual_mow_minutes: Optional[int] = None
    avg_actual_trim_minutes: Optional[int] = None
    avg_actual_blow_minutes: Optional[int] = None

class CrewMember(BaseModel):
    id: str
    name: str
    primary_role: str
    secondary_role: Optional[str] = None
    is_foreman: bool = False
    hourly_rate: float = 0.0

class Equipment(BaseModel):
    id: str
    name: str
    equipment_type: str
    deck_width_inches: Optional[int] = None
    cutting_speed_mph: Optional[float] = None
    max_slope_degrees: int = 15

class OptimizeRequest(BaseModel):
    job_id: str
    zones: List[Zone]
    crew: List[CrewMember]
    equipment: List[Equipment]
    available_crew: Optional[List[CrewMember]] = None

class TaskAssignment(BaseModel):
    crew_member_id: str
    crew_member_name: str
    task_order: int
    zone_id: str
    zone_label: str
    task_type: str
    estimated_minutes: int
    role_used: str
    is_role_switch: bool = False

class ScenarioResult(BaseModel):
    crew_size: int
    total_minutes: int
    wage_cost: float
    crew_assignments: List[TaskAssignment]

class OptimizeResponse(BaseModel):
    job_id: str
    lean: ScenarioResult
    optimal: ScenarioResult
    max_speed: ScenarioResult
    assigned: ScenarioResult
    solve_time_ms: int
    blow_note: str

# ─── Speed Constants ──────────────────────────────────────────────────────────

CUTTING_SPEED_MPH = {
    "zero_turn":    3.5,
    "riding_mower": 3.2,
    "walk_behind":  2.5,
    "trimmer":      2.8,
    "blower":       3.2,
    "edger":        2.5,
}

DECK_WIDTH_INCHES = {
    "zero_turn":    60,
    "riding_mower": 54,
    "walk_behind":  30,
    "trimmer":      1,
    "blower":       1,
    "edger":        1,
}

TRIMMER_FT_PER_MIN = 150.0

BLOWER_FT_PER_MIN = {
    "pavement":   200.0,
    "beds":       100.0,
    "mixed":      150.0,
    None:         150.0,
}

TASK_SWITCH_PENALTY_MINUTES = 8
MIN_CREW_SIZE = 2
LARGE_FIELD_ACRES = 10.0

MODE_SPEED_MULTIPLIER = {
    "fastest":  0.88,
    "balanced": 1.0,
    "cheapest": 1.18,
}

BASE_EFFICIENCY = {
    "fastest":  0.68,
    "balanced": 0.60,
    "cheapest": 0.54,
}

# ─── Acreage-Based Crew Floor ─────────────────────────────────────────────────
# Minimum crew size for Optimal scenario based on total mowable acreage.
# No experienced owner sends 2 people to a 40-acre HOA.
# These floors represent real-world minimum sensible staffing.
#
#  < 1 acre   → 2 crew  (typical single residential)
#  1-3 acres  → 2 crew  (large residential, small commercial)
#  3-8 acres  → 3 crew  (medium commercial, small HOA)
#  8-20 acres → 4 crew  (large commercial, medium HOA)
# 20-40 acres → 5 crew  (large HOA, municipal)
# > 40 acres  → 6 crew  (very large HOA, campus)

def get_optimal_crew_floor(zones: List[Zone]) -> int:
    mow_zones = [z for z in zones if z.zone_type in [
        "mow", "berm", "island", "courtyard", "slope"
    ]]
    total_sqft = sum(z.area_sqft for z in mow_zones)
    acres = total_sqft / 43560.0

    if acres < 1.0:
        return 2
    elif acres < 3.0:
        return 2
    elif acres < 8.0:
        return 3
    elif acres < 20.0:
        return 4
    elif acres < 40.0:
        return 5
    else:
        return 6

# ─── Turn Penalty ─────────────────────────────────────────────────────────────

def get_turn_penalty(area_sqft: float) -> float:
    acres = area_sqft / 43560.0
    if acres < 0.5:
        return 1.35
    elif acres < 2.0:
        return 1.20
    elif acres < 5.0:
        return 1.12
    elif acres < 15.0:
        return 1.06
    else:
        return 1.03

# ─── Measurement Waterfall ────────────────────────────────────────────────────

def get_trim_linear_ft(zone: Zone) -> float:
    if zone.linear_ft and zone.linear_ft > 0:
        return zone.linear_ft
    if zone.perimeter_ft and zone.perimeter_ft > 0:
        return zone.perimeter_ft
    return 5.5 * math.sqrt(max(zone.area_sqft, 100))

def get_blow_linear_ft(zone: Zone) -> float:
    if zone.linear_ft and zone.linear_ft > 0:
        return zone.linear_ft
    if zone.perimeter_ft and zone.perimeter_ft > 0:
        return zone.perimeter_ft
    return 4.0 * math.sqrt(max(zone.area_sqft, 100))

def get_zone_complexity(zone: Zone) -> float:
    base = zone.complexity_factor if zone.complexity_factor else 1.0
    obstacle_penalty = 1.0 + (zone.obstacle_density or 0.0) * 0.5
    return base * obstacle_penalty

# ─── Time Estimators ──────────────────────────────────────────────────────────

def estimate_mow_minutes(zone: Zone, equipment_type: str, mode: str = "balanced") -> int:
    if zone.avg_actual_mow_minutes and zone.avg_actual_mow_minutes > 0:
        return max(int(zone.avg_actual_mow_minutes * MODE_SPEED_MULTIPLIER.get(mode, 1.0)), 2)

    speed_mph = CUTTING_SPEED_MPH.get(equipment_type, 3.5)
    deck_ft = DECK_WIDTH_INCHES.get(equipment_type, 30) / 12.0
    efficiency = BASE_EFFICIENCY.get(mode, 0.60)

    area_acres = zone.area_sqft / 43560.0
    hours = area_acres / (speed_mph * deck_ft * efficiency)
    minutes = max(int(hours * 60), 2)

    minutes = int(minutes * get_turn_penalty(zone.area_sqft))

    if zone.slope_grade > 15:
        minutes = int(minutes * 1.6)
    elif zone.slope_grade > 10:
        minutes = int(minutes * 1.35)
    elif zone.slope_grade > 5:
        minutes = int(minutes * 1.15)

    if zone.zone_type in ["berm", "courtyard"]:
        minutes = int(minutes * 1.5)
    elif zone.zone_type == "island":
        minutes = int(minutes * 1.3)
    elif zone.zone_type == "slope":
        minutes = int(minutes * 1.4)

    minutes = int(minutes * get_zone_complexity(zone))
    minutes = int(minutes * MODE_SPEED_MULTIPLIER.get(mode, 1.0))

    return max(minutes, 2)


def estimate_trim_minutes(zone: Zone, mode: str = "balanced") -> int:
    if zone.avg_actual_trim_minutes and zone.avg_actual_trim_minutes > 0:
        return max(int(zone.avg_actual_trim_minutes * MODE_SPEED_MULTIPLIER.get(mode, 1.0)), 2)

    linear_ft = get_trim_linear_ft(zone)
    minutes = max(int(linear_ft / TRIMMER_FT_PER_MIN), 2)

    if zone.slope_grade > 10:
        minutes = int(minutes * 1.3)
    elif zone.slope_grade > 5:
        minutes = int(minutes * 1.15)

    minutes = int(minutes * get_zone_complexity(zone))
    minutes = int(minutes * MODE_SPEED_MULTIPLIER.get(mode, 1.0))

    return max(minutes, 2)


def estimate_blow_minutes(all_zones: List[Zone], mode: str = "balanced") -> int:
    blowable_zones = [z for z in all_zones if z.zone_type not in ["no_mow"]]

    actual_blow = next(
        (z.avg_actual_blow_minutes for z in blowable_zones
         if z.avg_actual_blow_minutes and z.avg_actual_blow_minutes > 0),
        None
    )
    if actual_blow:
        return max(int(actual_blow * MODE_SPEED_MULTIPLIER.get(mode, 1.0)), 10)

    total_blow_ft = 0.0
    has_real_footage = False

    for zone in blowable_zones:
        if zone.zone_type in ["perimeter", "trim"]:
            ft = get_blow_linear_ft(zone)
            if zone.linear_ft or zone.perimeter_ft:
                has_real_footage = True
            total_blow_ft += ft

    if has_real_footage and total_blow_ft > 0:
        blow_rate = BLOWER_FT_PER_MIN[None]
        surface_types = [z.surface_type for z in blowable_zones if z.surface_type]
        if surface_types:
            pavement_count = sum(1 for s in surface_types if s == "pavement")
            beds_count = sum(1 for s in surface_types if s == "beds")
            if pavement_count > beds_count:
                blow_rate = BLOWER_FT_PER_MIN["pavement"]
            elif beds_count > pavement_count:
                blow_rate = BLOWER_FT_PER_MIN["beds"]
        minutes = max(int(total_blow_ft / blow_rate), 10)
    else:
        zone_count = len(blowable_zones)
        minutes = max(10 + (zone_count * 5), 15)

    minutes = int(minutes * MODE_SPEED_MULTIPLIER.get(mode, 1.0))
    return max(minutes, 10)

# ─── Zone Classification ──────────────────────────────────────────────────────

def classify_zones(zones: List[Zone]):
    workable = [z for z in zones if z.zone_type != "no_mow"]
    mow_zones = [z for z in workable if z.zone_type in [
        "mow", "berm", "island", "courtyard", "slope"
    ]]
    trim_zones = [z for z in workable if z.zone_type in ["trim", "perimeter"]]
    large_fields = [
        z for z in mow_zones
        if z.area_sqft >= (LARGE_FIELD_ACRES * 43560)
        and not z.equipment_restriction
    ]
    small_mow = [z for z in mow_zones if z not in large_fields]
    return large_fields, small_mow, trim_zones, workable

# ─── Crew Helpers ─────────────────────────────────────────────────────────────

def get_foreman(crew: List[CrewMember]) -> Optional[CrewMember]:
    foremen = [c for c in crew if c.is_foreman]
    return foremen[0] if foremen else None

def get_mowers(crew: List[CrewMember]) -> List[CrewMember]:
    return sorted(
        [c for c in crew if c.primary_role in ["zero_turn", "walk_behind", "riding_mower"]],
        key=lambda c: c.hourly_rate, reverse=True
    )

def get_trimmers(crew: List[CrewMember]) -> List[CrewMember]:
    return [c for c in crew if c.primary_role == "trimmer"]

def order_crew(crew: List[CrewMember]) -> List[CrewMember]:
    foreman = get_foreman(crew)
    if foreman:
        others = sorted(
            [c for c in crew if c.id != foreman.id],
            key=lambda c: c.hourly_rate, reverse=True
        )
        return [foreman] + others
    return sorted(crew, key=lambda c: c.hourly_rate, reverse=True)

def fill_hourly_rates(crew: List[CrewMember], fallback: float = 20.60):
    for c in crew:
        if not c.hourly_rate or c.hourly_rate <= 0:
            c.hourly_rate = fallback

# ─── Section Division ─────────────────────────────────────────────────────────

def divide_into_sections(mow_zones: List[Zone], crew_count: int) -> List[List[Zone]]:
    if not mow_zones:
        return []
    mower_count = max(1, int(crew_count * 0.7))
    num_sections = min(mower_count, len(mow_zones))
    if num_sections <= 1:
        return [mow_zones]
    sorted_zones = sorted(mow_zones, key=lambda z: z.area_sqft, reverse=True)
    sections: List[List[Zone]] = [[] for _ in range(num_sections)]
    section_loads = [0.0] * num_sections
    for zone in sorted_zones:
        min_idx = section_loads.index(min(section_loads))
        sections[min_idx].append(zone)
        section_loads[min_idx] += zone.area_sqft
    return [s for s in sections if s]

# ─── Core Run ─────────────────────────────────────────────────────────────────

def run_subset(
    subset: List[CrewMember],
    large_fields: List[Zone],
    small_mow: List[Zone],
    trim_zones: List[Zone],
    all_zones: List[Zone],
    mode: str
) -> tuple:
    subset_load = {c.id: 0.0 for c in subset}
    assignments = []
    counter = 1

    subset_mowers = [c for c in subset if c.primary_role in [
        "zero_turn", "walk_behind", "riding_mower"
    ]]
    subset_trimmers = [c for c in subset if c.primary_role == "trimmer"]

    if not subset_mowers:
        subset_mowers = [max(subset, key=lambda c: c.hourly_rate)]

    if not subset_trimmers:
        non_foreman = [c for c in subset if not c.is_foreman and c not in subset_mowers]
        if non_foreman:
            subset_trimmers = [min(non_foreman, key=lambda c: c.hourly_rate)]
            subset_mowers = [c for c in subset_mowers if c not in subset_trimmers]
            if not subset_mowers:
                subset_mowers = [subset[0]]
        else:
            subset_trimmers = subset_mowers[-1:]

    all_mow = sorted(large_fields + small_mow, key=lambda z: z.area_sqft, reverse=True)
    sections = divide_into_sections(all_mow, len(subset))

    if sections:
        for section in sections:
            for zone in section:
                worker = min(subset_mowers, key=lambda c: subset_load.get(c.id, 0))
                eq = worker.primary_role if worker.primary_role in CUTTING_SPEED_MPH else "walk_behind"
                mins = estimate_mow_minutes(zone, eq, mode)
                assignments.append(TaskAssignment(
                    crew_member_id=worker.id,
                    crew_member_name=worker.name,
                    task_order=counter,
                    zone_id=zone.id,
                    zone_label=zone.label,
                    task_type="mow",
                    estimated_minutes=mins,
                    role_used=eq,
                    is_role_switch=False,
                ))
                subset_load[worker.id] = subset_load.get(worker.id, 0) + mins
                counter += 1

    for zone in sorted(trim_zones, key=lambda z: estimate_trim_minutes(z, mode), reverse=True):
        worker = min(subset_trimmers, key=lambda c: subset_load.get(c.id, 0))
        prev = [a for a in assignments if a.crew_member_id == worker.id]
        is_switch = bool(prev) and prev[-1].task_type == "mow"
        mins = estimate_trim_minutes(zone, mode)
        if is_switch:
            mins += TASK_SWITCH_PENALTY_MINUTES
        assignments.append(TaskAssignment(
            crew_member_id=worker.id,
            crew_member_name=worker.name,
            task_order=counter,
            zone_id=zone.id,
            zone_label=zone.label,
            task_type="trim",
            estimated_minutes=mins,
            role_used="trimmer",
            is_role_switch=is_switch,
        ))
        subset_load[worker.id] = subset_load.get(worker.id, 0) + mins
        counter += 1

    blow_mins = estimate_blow_minutes(all_zones, mode) + TASK_SWITCH_PENALTY_MINUTES
    blow_worker = min(subset, key=lambda c: subset_load.get(c.id, 0))
    assignments.append(TaskAssignment(
        crew_member_id=blow_worker.id,
        crew_member_name=blow_worker.name,
        task_order=counter,
        zone_id="cleanup",
        zone_label="Final Blowout & Cleanup",
        task_type="blow",
        estimated_minutes=blow_mins,
        role_used="blower",
        is_role_switch=True,
    ))
    subset_load[blow_worker.id] = subset_load.get(blow_worker.id, 0) + blow_mins

    job_time = max(subset_load.values()) if subset_load else 0
    wage_cost = sum(
        subset_load.get(c.id, 0) * c.hourly_rate / 60.0
        for c in subset
    )
    return assignments, job_time, wage_cost

# ─── Three Scenario Functions ─────────────────────────────────────────────────

def scenario_lean(
    zones: List[Zone],
    available_crew: List[CrewMember]
) -> ScenarioResult:
    """
    Lean Crew: Always exactly 2 crew.
    Both may mow on large sites — first finished trims/blows.
    Time multiplier: cheapest (1.18x) — smaller crew, slower pace.
    """
    large_fields, small_mow, trim_zones, _ = classify_zones(zones)
    ordered = order_crew(available_crew)
    subset = ordered[:2]

    assignments, job_time, wage_cost = run_subset(
        subset, large_fields, small_mow, trim_zones, zones, "cheapest"
    )
    return ScenarioResult(
        crew_size=2,
        total_minutes=int(job_time),
        wage_cost=round(wage_cost, 2),
        crew_assignments=assignments,
    )


def scenario_optimal(
    zones: List[Zone],
    available_crew: List[CrewMember]
) -> ScenarioResult:
    """
    Optimal Crew: Acreage-based floor + 12% improvement algorithm.

    Step 1 — Calculate minimum crew floor from total mowable acreage:
      < 1 acre   → 2 crew
      1-3 acres  → 2 crew
      3-8 acres  → 3 crew
      8-20 acres → 4 crew
      20-40 acres → 5 crew
      > 40 acres → 6 crew

    Step 2 — Start at the floor, add one crew member at a time.
      Stop when adding one more person saves less than 12% time.
      Cap at min(available crew, 10).

    Uses balanced time multiplier (1.0x).
    """
    large_fields, small_mow, trim_zones, _ = classify_zones(zones)
    ordered = order_crew(available_crew)
    max_crew = min(len(ordered), 10)

    # Step 1 — acreage floor
    floor = get_optimal_crew_floor(zones)
    floor = min(floor, max_crew)

    print(f"OPTIMAL floor={floor} max={max_crew}")

    # Step 2 — start at floor
    best_assignments, best_time, best_cost = run_subset(
        ordered[:floor], large_fields, small_mow, trim_zones, zones, "balanced"
    )
    best_size = floor
    print(f"OPTIMAL size={floor} time={best_time:.0f}min")

    # Iterate upward from floor
    for size in range(floor + 1, max_crew + 1):
        subset = ordered[:size]
        assignments, job_time, wage_cost = run_subset(
            subset, large_fields, small_mow, trim_zones, zones, "balanced"
        )
        improvement = (best_time - job_time) / best_time if best_time > 0 else 0
        print(f"OPTIMAL size={size} time={job_time:.0f}min improvement={improvement:.1%}")

        if improvement >= 0.12:
            best_size = size
            best_time = job_time
            best_cost = wage_cost
            best_assignments = assignments
        else:
            break

    print(f"OPTIMAL final: {best_size} crew, {best_time:.0f}min, ${best_cost:.2f}")
    return ScenarioResult(
        crew_size=best_size,
        total_minutes=int(best_time),
        wage_cost=round(best_cost, 2),
        crew_assignments=best_assignments,
    )


def scenario_max_speed(
    zones: List[Zone],
    available_crew: List[CrewMember],
    optimal_size: int
) -> ScenarioResult:
    """
    Max Speed: Optimal crew size + 2 extra.
    Uses fastest time multiplier (0.88x).
    Capped at total available crew.
    """
    large_fields, small_mow, trim_zones, _ = classify_zones(zones)
    ordered = order_crew(available_crew)
    size = min(optimal_size + 2, len(ordered))
    subset = ordered[:size]

    assignments, job_time, wage_cost = run_subset(
        subset, large_fields, small_mow, trim_zones, zones, "fastest"
    )
    print(f"MAX SPEED: {size} crew, {job_time:.0f}min, ${wage_cost:.2f}")
    return ScenarioResult(
        crew_size=size,
        total_minutes=int(job_time),
        wage_cost=round(wage_cost, 2),
        crew_assignments=assignments,
    )


def scenario_assigned(
    zones: List[Zone],
    assigned_crew: List[CrewMember]
) -> ScenarioResult:
    """
    Assigned Crew: Runs assignment for exactly the pre-assigned crew.
    Uses balanced multiplier. Saved to job_tasks in the DB.
    """
    large_fields, small_mow, trim_zones, _ = classify_zones(zones)
    ordered = order_crew(assigned_crew)

    assignments, job_time, wage_cost = run_subset(
        ordered, large_fields, small_mow, trim_zones, zones, "balanced"
    )
    return ScenarioResult(
        crew_size=len(ordered),
        total_minutes=int(job_time),
        wage_cost=round(wage_cost, 2),
        crew_assignments=assignments,
    )

# ─── API Routes ───────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "lawnroute-optimizer"}


@app.post("/optimize", response_model=OptimizeResponse)
def optimize(request: OptimizeRequest):
    start = time.time()

    if not request.zones:
        raise HTTPException(status_code=400, detail="No zones provided")
    if not request.crew:
        raise HTTPException(status_code=400, detail="No crew provided")

    assigned_ids = {c.id for c in request.crew}
    extra_crew = [c for c in (request.available_crew or []) if c.id not in assigned_ids]
    all_crew = list(request.crew) + extra_crew

    fill_hourly_rates(all_crew)
    fill_hourly_rates(list(request.crew))

    try:
        lean_result = scenario_lean(request.zones, all_crew)
        optimal_result = scenario_optimal(request.zones, all_crew)
        max_result = scenario_max_speed(request.zones, all_crew, optimal_result.crew_size)
        assigned_result = scenario_assigned(request.zones, list(request.crew))
    except Exception as e:
        import traceback
        print(f"OPTIMIZER ERROR: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Optimizer error: {str(e)}")

    blow_zones = [z for z in request.zones if z.zone_type in ["perimeter", "trim"]]
    has_real_footage = any(z.linear_ft or z.perimeter_ft for z in blow_zones)
    if has_real_footage:
        total_blow_ft = sum((z.linear_ft or z.perimeter_ft or 0) for z in blow_zones)
        blow_note = f"Whoever finishes first blows — approx {int(total_blow_ft):,} ft of surfaces"
    else:
        blow_note = "Whoever finishes first picks up the blower"

    solve_ms = int((time.time() - start) * 1000)

    return OptimizeResponse(
        job_id=request.job_id,
        lean=lean_result,
        optimal=optimal_result,
        max_speed=max_result,
        assigned=assigned_result,
        solve_time_ms=solve_ms,
        blow_note=blow_note,
    )


@app.get("/recommend-crew")
def recommend_crew(total_sqft: float, zone_types: str = "mow,trim"):
    zones_list = zone_types.split(",")
    has_slopes = "berm" in zones_list or "slope" in zones_list
    has_courtyards = "courtyard" in zones_list
    has_large_trim = "perimeter" in zones_list
    acres = total_sqft / 43560.0

    lean = max(MIN_CREW_SIZE, int(acres / 2))
    optimal = max(3, int(acres / 1.5))
    fast = max(4, int(acres / 1.0))

    if has_slopes:
        lean += 1
        optimal += 1
    if has_courtyards:
        optimal += 1
    if has_large_trim:
        lean += 1

    return {"lean": lean, "optimal": optimal, "fast": fast, "acres": round(acres, 2)}


class BuildGraphRequest(BaseModel):
    property_id: str
    triggered_by_user_id: Optional[str] = None


@app.post("/build-graph")
def build_graph(request: BuildGraphRequest):
    try:
        from osm_graph_builder import build_graph_for_property
        result = build_graph_for_property(
            request.property_id,
            request.triggered_by_user_id
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        print(f"BUILD GRAPH ERROR: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
