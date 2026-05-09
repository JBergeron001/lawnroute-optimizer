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

# ─── Models ──────────────────────────────────────────────────────────────────

class Zone(BaseModel):
    id: str
    label: str
    zone_type: str
    area_sqft: float = 0.0
    slope_grade: float = 0.0
    equipment_restriction: Optional[str] = None
    # Measurement waterfall fields — all optional, optimizer uses best available
    perimeter_ft: Optional[float] = None      # perimeter of zone boundary
    linear_ft: Optional[float] = None         # actual path length for trim/blow
    complexity_factor: Optional[float] = 1.0  # manual complexity override
    surface_type: Optional[str] = None        # for blow zones: pavement, beds, etc.
    obstacle_density: Optional[float] = 0.0  # 0.0-1.0 obstacle density

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
    mode: str = "balanced"
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

class OptimizeResponse(BaseModel):
    job_id: str
    total_estimated_minutes: int
    crew_assignments: List[TaskAssignment]
    solve_time_ms: int
    recommended_crew_size: int
    mode: str
    crew_split_suggestion: str
    blow_note: str

# ─── Speed Constants ──────────────────────────────────────────────────────────
# Walking speed = 3.1 mph
# Trimmer = walking speed (3.1 mph)
# Mowers = 1.5x walking speed (4.65 mph)
# Blower = 1.2x walking speed (3.72 mph)
# Edger = walking speed (3.1 mph)

CUTTING_SPEED_MPH = {
    "zero_turn":    4.65,
    "riding_mower": 4.65,
    "walk_behind":  4.65,
    "trimmer":      3.1,
    "blower":       3.72,
    "edger":        3.1,
}

DECK_WIDTH_INCHES = {
    "zero_turn":    60,
    "riding_mower": 54,
    "walk_behind":  30,
    "trimmer":      1,
    "blower":       1,
    "edger":        1,
}

# Trimmer linear feet per minute at walking speed
TRIMMER_FT_PER_MIN = 150.0

# Blower linear feet per minute — varies by surface
BLOWER_FT_PER_MIN = {
    "pavement":   200.0,  # driveways, sidewalks — fast
    "beds":       100.0,  # landscape beds — slow, more careful
    "mixed":      150.0,  # default mixed surfaces
    None:         150.0,  # unknown — use mixed
}

# Task switch penalty: crew member changes equipment/role between tasks
# Includes walking to trailer, swapping equipment, returning to work area
TASK_SWITCH_PENALTY_MINUTES = 8

# Minimum crew size enforced in all modes
MIN_CREW_SIZE = 2

# Large field threshold — single mower handles alone unless above this
LARGE_FIELD_ACRES = 10.0

# Mode speed multipliers — affect time estimates
# fastest = crew works faster (more focused, larger crew)
# cheapest = smaller crew, slightly slower pace
MODE_SPEED_MULTIPLIER = {
    "fastest": 0.85,
    "balanced": 1.0,
    "cheapest": 1.15,
}

# Property type complexity multipliers
# Applied to area_sqft before time estimation
# These reflect real-world complexity: HOA has many small zones, obstacles, tight spaces
PROPERTY_TYPE_COMPLEXITY = {
    "residential_hoa":  1.0,
    "commercial":       1.2,
    "municipal":        1.3,
    "hoa_community":    2.5,   # Large HOA: many zones, obstacles, tight spaces
    "industrial":       1.1,
}

# ─── Measurement Waterfall ────────────────────────────────────────────────────
# The optimizer always uses the best available measurement.
# Priority: manual input > PostGIS calculation > GPS trace (future) > estimation

def get_trim_linear_ft(zone: Zone) -> float:
    """
    Get the best available linear footage for trim estimation.
    Priority: linear_ft > perimeter_ft > estimated from area
    """
    if zone.linear_ft and zone.linear_ft > 0:
        return zone.linear_ft
    if zone.perimeter_ft and zone.perimeter_ft > 0:
        return zone.perimeter_ft
    # Fallback: approximate from area using sqrt relationship
    # 5.5 is empirically calibrated for typical lawn zones
    return 5.5 * math.sqrt(max(zone.area_sqft, 100))

def get_blow_linear_ft(zone: Zone) -> float:
    """
    Get the best available linear footage for blow estimation.
    Blow zones use linear_ft (actual path length) > perimeter_ft > area estimation
    """
    if zone.linear_ft and zone.linear_ft > 0:
        return zone.linear_ft
    if zone.perimeter_ft and zone.perimeter_ft > 0:
        return zone.perimeter_ft
    # Fallback: estimate based on area
    return 4.0 * math.sqrt(max(zone.area_sqft, 100))

def get_zone_complexity(zone: Zone) -> float:
    """
    Get the combined complexity multiplier for a zone.
    Combines zone-level complexity_factor with obstacle density.
    """
    base = zone.complexity_factor if zone.complexity_factor else 1.0
    # Obstacle density adds up to 50% time penalty at max density
    obstacle_penalty = 1.0 + (zone.obstacle_density or 0.0) * 0.5
    return base * obstacle_penalty

# ─── Time Estimators ─────────────────────────────────────────────────────────

def estimate_mow_minutes(zone: Zone, equipment_type: str, mode: str = "balanced") -> int:
    """
    Estimate mow time for a zone.
    Uses area_sqft with deck width and speed.
    Applies slope penalty, zone type penalty, complexity factor, and mode multiplier.
    """
    speed_mph = CUTTING_SPEED_MPH.get(equipment_type, 4.65)
    deck_ft = DECK_WIDTH_INCHES.get(equipment_type, 30) / 12.0

    # Efficiency varies by mode
    efficiency = 0.80
    if mode == "fastest":
        efficiency = 0.88
    elif mode == "cheapest":
        efficiency = 0.75

    area_acres = zone.area_sqft / 43560.0
    hours = area_acres / (speed_mph * deck_ft * efficiency)
    minutes = max(int(hours * 60), 2)

    # Slope penalty
    if zone.slope_grade > 15:
        minutes = int(minutes * 1.5)
    elif zone.slope_grade > 10:
        minutes = int(minutes * 1.25)
    elif zone.slope_grade > 5:
        minutes = int(minutes * 1.1)

    # Zone type penalty (berms and courtyards have more turns and obstacles)
    if zone.zone_type in ["berm", "courtyard"]:
        minutes = int(minutes * 1.4)
    elif zone.zone_type == "island":
        minutes = int(minutes * 1.2)

    # Zone complexity factor (manual override + obstacle density)
    minutes = int(minutes * get_zone_complexity(zone))

    # Mode speed multiplier
    minutes = int(minutes * MODE_SPEED_MULTIPLIER.get(mode, 1.0))

    return max(minutes, 2)


def estimate_trim_minutes(zone: Zone, mode: str = "balanced") -> int:
    """
    Estimate trim time for a zone.
    Uses best available linear footage (waterfall: linear_ft > perimeter_ft > estimated).
    Applies slope penalty and mode multiplier.
    """
    linear_ft = get_trim_linear_ft(zone)
    minutes = max(int(linear_ft / TRIMMER_FT_PER_MIN), 2)

    # Slope penalty for trim (harder to hold trimmer steady on slopes)
    if zone.slope_grade > 10:
        minutes = int(minutes * 1.3)
    elif zone.slope_grade > 5:
        minutes = int(minutes * 1.1)

    # Zone complexity
    minutes = int(minutes * get_zone_complexity(zone))

    # Mode multiplier
    minutes = int(minutes * MODE_SPEED_MULTIPLIER.get(mode, 1.0))

    return max(minutes, 2)


def estimate_blow_minutes(all_zones: List[Zone], mode: str = "balanced") -> int:
    """
    Estimate blow/cleanup time.
    Uses real linear footage from blow/perimeter zones when available.
    Falls back to zone count estimation.

    Blow zones include: perimeter zones, trim zones, driveways, sidewalks,
    landscape beds, patios, courtyards — any surface that collects debris.
    """
    # Sum actual linear footage from perimeter/trim zones that need blowing
    blowable_zones = [z for z in all_zones if z.zone_type not in ["no_mow"]]

    total_blow_ft = 0.0
    has_real_footage = False

    for zone in blowable_zones:
        if zone.zone_type in ["perimeter", "trim"]:
            ft = get_blow_linear_ft(zone)
            if zone.linear_ft or zone.perimeter_ft:
                has_real_footage = True
            total_blow_ft += ft

    if has_real_footage and total_blow_ft > 0:
        # Use real footage — determine surface type from zones
        # Default to mixed surface speed
        blow_rate = BLOWER_FT_PER_MIN[None]

        # Check if any zones specify surface type
        surface_types = [z.surface_type for z in blowable_zones if z.surface_type]
        if surface_types:
            # Use most common surface type's rate
            pavement_count = sum(1 for s in surface_types if s == "pavement")
            beds_count = sum(1 for s in surface_types if s == "beds")
            if pavement_count > beds_count:
                blow_rate = BLOWER_FT_PER_MIN["pavement"]
            elif beds_count > pavement_count:
                blow_rate = BLOWER_FT_PER_MIN["beds"]

        minutes = max(int(total_blow_ft / blow_rate), 10)
    else:
        # Fallback: zone count based estimation
        zone_count = len(blowable_zones)
        minutes = max(10 + (zone_count * 3), 15)

    # Mode multiplier
    minutes = int(minutes * MODE_SPEED_MULTIPLIER.get(mode, 1.0))

    return max(minutes, 10)

# ─── Zone Classification ──────────────────────────────────────────────────────

def classify_zones(zones: List[Zone]):
    """
    Separate zones by task type for assignment.
    Returns: large_fields, small_mow, trim_zones, blow_zones, workable
    """
    workable = [z for z in zones if z.zone_type != "no_mow"]

    mow_zones = [z for z in workable if z.zone_type in [
        "mow", "berm", "island", "courtyard", "slope"
    ]]
    trim_zones = [z for z in workable if z.zone_type in ["trim", "perimeter"]]

    # Large fields: single large mow zone handled by one dedicated mower
    large_fields = [
        z for z in mow_zones
        if z.area_sqft >= (LARGE_FIELD_ACRES * 43560)
        and not z.equipment_restriction
    ]
    small_mow = [z for z in mow_zones if z not in large_fields]

    return large_fields, small_mow, trim_zones, workable

# ─── Crew Classification ──────────────────────────────────────────────────────

def get_foreman(crew: List[CrewMember]) -> Optional[CrewMember]:
    foremen = [c for c in crew if c.is_foreman]
    return foremen[0] if foremen else None

def get_mowers(crew: List[CrewMember]) -> List[CrewMember]:
    return sorted(
        [c for c in crew if c.primary_role in ["zero_turn", "walk_behind", "riding_mower"]],
        key=lambda c: c.hourly_rate,
        reverse=True
    )

def get_trimmers(crew: List[CrewMember]) -> List[CrewMember]:
    return [c for c in crew if c.primary_role == "trimmer"]

# ─── Section Division ─────────────────────────────────────────────────────────

def divide_into_sections(mow_zones: List[Zone], crew_count: int) -> List[List[Zone]]:
    """
    Divide mow zones into sections based on crew size.
    Each section gets balanced work load.
    More crew = more sections = parallel work = faster completion.

    With 2 crew: 2 sections
    With 3-4 crew: 3 sections
    With 5+ crew: crew_count sections (one per mower roughly)
    """
    if not mow_zones:
        return []

    # Determine number of sections based on crew size
    # Not every crew member mows — approximately 70% mow, 30% trim
    mower_count = max(1, int(crew_count * 0.7))
    num_sections = min(mower_count, len(mow_zones))

    if num_sections <= 1:
        return [mow_zones]

    # Sort zones by area descending for balanced distribution
    sorted_zones = sorted(mow_zones, key=lambda z: z.area_sqft, reverse=True)

    # Distribute zones to sections using round-robin
    # This balances total area across sections
    sections: List[List[Zone]] = [[] for _ in range(num_sections)]
    section_loads = [0.0] * num_sections

    for zone in sorted_zones:
        # Assign to least loaded section
        min_load_idx = section_loads.index(min(section_loads))
        sections[min_load_idx].append(zone)
        section_loads[min_load_idx] += zone.area_sqft

    # Remove empty sections
    return [s for s in sections if s]

# ─── Core Assignment Logic ────────────────────────────────────────────────────

def run_subset(
    subset: List[CrewMember],
    large_fields: List[Zone],
    small_mow: List[Zone],
    trim_zones: List[Zone],
    all_zones: List[Zone],
    mode: str
):
    """
    Assign tasks to a specific crew subset and calculate job time and cost.
    Returns: (assignments, job_completion_time_minutes, total_wage_cost)
    """
    subset_load = {c.id: 0.0 for c in subset}
    assignments = []
    counter = 1

    # Identify mowers and trimmers in subset
    subset_mowers = [c for c in subset if c.primary_role in [
        "zero_turn", "walk_behind", "riding_mower"
    ]]
    subset_trimmers = [c for c in subset if c.primary_role == "trimmer"]

    # Fallback: if no dedicated mowers, use highest-paid crew member
    if not subset_mowers:
        subset_mowers = [max(subset, key=lambda c: c.hourly_rate)]

    # Fallback: if no dedicated trimmers, use lowest-paid non-foreman
    if not subset_trimmers:
        non_foreman = [c for c in subset if not c.is_foreman and c not in subset_mowers]
        if non_foreman:
            # Lowest wage earner trims
            subset_trimmers = [min(non_foreman, key=lambda c: c.hourly_rate)]
            # Remove trimmer from mower pool if they were in it
            subset_mowers = [c for c in subset_mowers if c not in subset_trimmers]
            if not subset_mowers:
                subset_mowers = [subset[0]]
        else:
            subset_trimmers = [subset[-1]]

    # Divide mow zones into sections based on crew size
    all_mow = sorted(large_fields + small_mow, key=lambda z: z.area_sqft, reverse=True)
    sections = divide_into_sections(all_mow, len(subset))

    if sections:
        # Assign each section's zones to the least-loaded mower
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

    # Assign trim zones to trimmers
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

    # Assign blow to whoever finishes first
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

    # Job completion time = when the last crew member finishes
    job_time = max(subset_load.values()) if subset_load else 0
    # Wage cost = sum of (time * hourly_rate) for each crew member
    wage_cost = sum(
        subset_load.get(c.id, 0) * c.hourly_rate / 60.0
        for c in subset
    )

    return assignments, job_time, wage_cost


def assign_zones_to_crew(zones: List[Zone], crew: List[CrewMember], mode: str) -> List[TaskAssignment]:
    """
    Main assignment function. Handles all 3 modes:
    - cheapest (Lean Crew): always minimum 2 crew, minimize wage cost
    - fastest (Max Speed): find crew size that minimizes completion time
    - balanced (Optimal): find best time/cost ratio
    """
    FALLBACK_WAGE = 20.60

    # Classify zones
    large_fields, small_mow, trim_zones, workable_zones = classify_zones(zones)

    # Ensure all crew have a wage
    for c in crew:
        if not c.hourly_rate or c.hourly_rate <= 0:
            c.hourly_rate = FALLBACK_WAGE

    # Order crew: foreman first, then by wage descending
    foreman = get_foreman(crew)
    if foreman:
        others = sorted(
            [c for c in crew if c.id != foreman.id],
            key=lambda c: c.hourly_rate,
            reverse=True
        )
        ordered_crew = [foreman] + others
    else:
        ordered_crew = sorted(crew, key=lambda c: c.hourly_rate, reverse=True)

    # Enforce minimum crew size
    if len(ordered_crew) < MIN_CREW_SIZE:
        # Duplicate crew members to meet minimum (only happens with 1-person companies)
        ordered_crew = ordered_crew * MIN_CREW_SIZE

    if mode == "cheapest":
        # Lean Crew: always use exactly minimum 2 crew
        # Minimizes wage cost — good for simple jobs
        subset = ordered_crew[:MIN_CREW_SIZE]
        assignments, job_time, wage_cost = run_subset(
            subset, large_fields, small_mow, trim_zones, zones, mode
        )
        print(f"LEAN CREW: {len(subset)} crew, {job_time:.0f} min, ${wage_cost:.2f}")
        return assignments

    elif mode == "fastest":
        # Max Speed: find the crew size that minimizes completion time
        # Start at minimum crew and add crew until time stops improving
        best_assignments, best_time, best_cost = run_subset(
            ordered_crew[:MIN_CREW_SIZE],
            large_fields, small_mow, trim_zones, zones, mode
        )
        print(f"FASTEST: size={MIN_CREW_SIZE} time={best_time:.0f}min")

        for size in range(MIN_CREW_SIZE + 1, len(ordered_crew) + 1):
            subset = ordered_crew[:size]
            assignments, job_time, wage_cost = run_subset(
                subset, large_fields, small_mow, trim_zones, zones, mode
            )
            print(f"FASTEST: size={size} time={job_time:.0f}min")
            if job_time < best_time * 0.95:
                # At least 5% improvement to justify adding crew
                best_time = job_time
                best_cost = wage_cost
                best_assignments = assignments
            else:
                # Time not improving meaningfully — stop adding crew
                break

        print(f"FASTEST final: {best_time:.0f}min")
        return best_assignments

    else:
        # Balanced (Optimal): find best time * cost score
        # Lower score = better balance of speed and cost
        best_assignments, best_time, best_cost = run_subset(
            ordered_crew[:MIN_CREW_SIZE],
            large_fields, small_mow, trim_zones, zones, mode
        )
        best_score = best_time * best_cost
        print(f"BALANCED: size={MIN_CREW_SIZE} score={best_score:.2f}")

        for size in range(MIN_CREW_SIZE + 1, len(ordered_crew) + 1):
            subset = ordered_crew[:size]
            assignments, job_time, wage_cost = run_subset(
                subset, large_fields, small_mow, trim_zones, zones, mode
            )
            score = job_time * wage_cost
            print(f"BALANCED: size={size} score={score:.2f}")
            if score < best_score * 0.97:
                # At least 3% improvement in combined score
                best_score = score
                best_assignments = assignments
            else:
                # Score not improving — stop
                break

        print(f"BALANCED final: score={best_score:.2f}")
        return best_assignments

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
    if request.mode not in ["fastest", "cheapest", "balanced"]:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {request.mode}. Use fastest, cheapest, or balanced.")

    # Combine assigned crew with available crew pool for optimization
    all_crew = list(request.crew) + [
        c for c in (request.available_crew or [])
        if c.id not in {m.id for m in request.crew}
    ]

    try:
        assignments = assign_zones_to_crew(request.zones, all_crew, request.mode)
    except Exception as e:
        import traceback
        print(f"OPTIMIZER ERROR: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Optimizer error: {str(e)}")

    # Calculate totals
    crew_times: dict = {}
    for a in assignments:
        crew_times[a.crew_member_id] = crew_times.get(a.crew_member_id, 0) + a.estimated_minutes

    total_minutes = max(crew_times.values()) if crew_times else 0

    # Recommended crew size based on total work volume
    total_work = sum(a.estimated_minutes for a in assignments)
    recommended = max(MIN_CREW_SIZE, min(10, round(total_work / 45)))

    # Build crew split description
    large_fields, small_mow, trim_zones, _ = classify_zones(request.zones)
    mowers = get_mowers(request.crew)
    trimmers = get_trimmers(request.crew)

    if not trimmers:
        non_foreman = [c for c in mowers if not c.is_foreman]
        trim_count = max(1, len(request.crew) // 4) if len(request.crew) >= 4 else 1
        trimmer_count = min(trim_count, len(non_foreman))
        mower_count = len(request.crew) - trimmer_count
    else:
        mower_count = len(mowers)
        trimmer_count = len(trimmers)

    # Build informative blow note based on measurement quality
    blow_zones = [z for z in request.zones if z.zone_type in ["perimeter", "trim"]]
    has_real_footage = any(z.linear_ft or z.perimeter_ft for z in blow_zones)
    if has_real_footage:
        total_blow_ft = sum(
            (z.linear_ft or z.perimeter_ft or 0)
            for z in blow_zones
        )
        blow_note = f"Whoever finishes first blows — approx {int(total_blow_ft):,} ft of surfaces"
    else:
        blow_note = "Whoever finishes first picks up the blower"

    solve_ms = int((time.time() - start) * 1000)

    return OptimizeResponse(
        job_id=request.job_id,
        total_estimated_minutes=total_minutes,
        crew_assignments=assignments,
        solve_time_ms=solve_ms,
        recommended_crew_size=recommended,
        mode=request.mode,
        crew_split_suggestion=f"{mower_count} mow / {trimmer_count} trim",
        blow_note=blow_note,
    )


@app.get("/recommend-crew")
def recommend_crew(total_sqft: float, zone_types: str = "mow,trim"):
    """
    Recommend crew sizes for a property based on total area and zone types.
    Returns lean, optimal, and fast crew size recommendations.
    """
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

    return {
        "lean": lean,
        "optimal": optimal,
        "fast": fast,
        "acres": round(acres, 2),
    }


class BuildGraphRequest(BaseModel):
    property_id: str
    triggered_by_user_id: Optional[str] = None


@app.post("/build-graph")
def build_graph(request: BuildGraphRequest):
    """
    Trigger OSM graph build for a property.
    The actual graph building happens in osm_graph_builder.py.
    """
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
