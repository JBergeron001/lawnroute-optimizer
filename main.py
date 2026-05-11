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
    capturing site-specific difficulty. It should be set based on:
      • Shape irregularity (irregular = more turns per acre)
      • Number of internal obstacles (trees, beds, structures)
      • Access constraints (tight gates, narrow passages)
      • Terrain variation within the zone

  - Property type complexity is handled HERE in the optimizer, not in
    the API layer. The API sends raw area_sqft — no pre-multiplication.

Architecture note:
  - When GPS actuals accumulate (avg_actual_mow_minutes on property_zones),
    the optimizer should prefer those over formula estimates.
    Hook: pass avg_actual_minutes in zone data from API, use here as override.
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
    # Measurement waterfall — optimizer uses best available
    perimeter_ft: Optional[float] = None      # perimeter of zone boundary
    linear_ft: Optional[float] = None         # actual path length for trim/blow
    complexity_factor: Optional[float] = 1.0  # zone-level complexity override
    surface_type: Optional[str] = None        # for blow zones: pavement, beds, etc.
    obstacle_density: Optional[float] = 0.0  # 0.0-1.0 obstacle density
    # GPS actuals — fed back after completed jobs via GPS analysis engine
    # When present, these override formula-based estimates
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
# These are EFFECTIVE speeds, not theoretical max speeds.
# Reduced ~1 mph from theoretical to account for:
#   - Row-end turn time (decelerate, 180°, accelerate) every 60-100 ft
#   - 6-12" overlap between passes (adds ~10-15% more passes than pure math)
#   - Boundary perimeter passes (first lap around entire zone before stripping)
#   - Repositioning around obstacles, gates, narrow passages
#   - Terrain variation slowing consistent speed
#
# Real-world effective speed benchmarks (industry standard):
#   Zero-turn 60": ~3.0-3.5 ac/hr on open turf → ~2.5 mph effective
#   Walk-behind 30": ~1.0-1.5 ac/hr → ~1.8 mph effective
#   Trimmer: ~150-200 linear ft/min at walking pace
#   Blower: ~150-250 linear ft/min depending on surface

CUTTING_SPEED_MPH = {
    "zero_turn":    3.5,   # was 4.65 — reduced for turns, overlap, boundary passes
    "riding_mower": 3.2,   # was 4.65
    "walk_behind":  2.5,   # was 4.65 — significantly slower, more turns per acre
    "trimmer":      2.8,   # was 3.1 — walking with trimmer, not free walking
    "blower":       3.2,   # was 3.72
    "edger":        2.5,   # was 3.1
}

DECK_WIDTH_INCHES = {
    "zero_turn":    60,
    "riding_mower": 54,
    "walk_behind":  30,
    "trimmer":      1,
    "blower":       1,
    "edger":        1,
}

# Trimmer linear feet per minute
# 150 ft/min = 2.5 mph walking pace — realistic for trim work
TRIMMER_FT_PER_MIN = 150.0

# Blower linear feet per minute — varies by surface type
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
MODE_SPEED_MULTIPLIER = {
    "fastest":  0.88,   # larger crew, more focused — modest improvement
    "balanced": 1.0,
    "cheapest": 1.18,   # smaller crew, slightly slower pace
}

# ─── Efficiency ───────────────────────────────────────────────────────────────
# Base efficiency accounts for the mechanical losses in mowing:
#   - Row overlap (~10% more area than pure math)
#   - Turn time at row ends (~15-20% of total time on small-medium zones)
#   - Boundary pass (adds ~5-10% more time)
#   - Operator pace variation
#
# 0.60 means the mower is doing productive cutting work 60% of the time.
# This is the industry standard for realistic production rate estimates.
# Note: the turn_penalty_factor below adds additional time for zones
# where turns are more frequent (smaller zones, irregular shapes).

BASE_EFFICIENCY = {
    "fastest":  0.68,   # larger crew, better flow
    "balanced": 0.60,
    "cheapest": 0.54,   # smaller crew, less parallel work
}

# ─── Turn Penalty ─────────────────────────────────────────────────────────────
# Smaller zones require more turns per acre than large open fields.
# A 1-acre square zone at 60" deck width requires ~72 row passes.
# A 10-acre field at 60" deck requires ~720 passes but with better flow.
#
# Turn penalty adds extra time based on estimated turns per acre.
# Formula: turns_per_acre = zone_width / deck_width_ft
# This is approximated from zone area (assuming roughly square shape).
#
# Penalty is expressed as additional fraction of base mow time:
#   < 0.5 acre:  +35% (many turns relative to cutting time)
#   0.5-2 acres: +20%
#   2-5 acres:   +12%
#   5-15 acres:  +6%
#   > 15 acres:  +3% (large open fields — turns are minimal fraction)

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
# Priority: GPS actuals > manual input > PostGIS calc > estimation

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
    # Obstacle density adds up to 50% time penalty at max density
    obstacle_penalty = 1.0 + (zone.obstacle_density or 0.0) * 0.5
    return base * obstacle_penalty

# ─── Time Estimators ──────────────────────────────────────────────────────────

def estimate_mow_minutes(zone: Zone, equipment_type: str, mode: str = "balanced") -> int:
    """
    Estimate mow time for a zone.

    If GPS actuals exist (avg_actual_mow_minutes from completed jobs),
    use those as the baseline — they capture real-world conditions
    including all turns, overlaps, and obstacles automatically.

    Otherwise, use the calibrated formula with:
      - Realistic effective speed (not theoretical max)
      - Turn penalty based on zone size
      - Slope penalty
      - Zone type penalty
      - Zone complexity factor (manual override + obstacle density)
      - Mode multiplier
    """
    # GPS actual override — most accurate estimate available
    if zone.avg_actual_mow_minutes and zone.avg_actual_mow_minutes > 0:
        # Apply mode multiplier to actual (fastest mode still runs faster)
        return max(int(zone.avg_actual_mow_minutes * MODE_SPEED_MULTIPLIER.get(mode, 1.0)), 2)

    speed_mph = CUTTING_SPEED_MPH.get(equipment_type, 3.5)
    deck_ft = DECK_WIDTH_INCHES.get(equipment_type, 30) / 12.0
    efficiency = BASE_EFFICIENCY.get(mode, 0.60)

    area_acres = zone.area_sqft / 43560.0
    hours = area_acres / (speed_mph * deck_ft * efficiency)
    minutes = max(int(hours * 60), 2)

    # Turn penalty — accounts for row-end turns, boundary passes, overlap
    # This is the key addition that makes small/irregular zones more realistic
    minutes = int(minutes * get_turn_penalty(zone.area_sqft))

    # Slope penalty — slopes slow turning and require more careful passes
    if zone.slope_grade > 15:
        minutes = int(minutes * 1.6)
    elif zone.slope_grade > 10:
        minutes = int(minutes * 1.35)
    elif zone.slope_grade > 5:
        minutes = int(minutes * 1.15)

    # Zone type penalty — berms and courtyards have tight turns, obstacles
    if zone.zone_type in ["berm", "courtyard"]:
        minutes = int(minutes * 1.5)
    elif zone.zone_type == "island":
        minutes = int(minutes * 1.3)
    elif zone.zone_type == "slope":
        minutes = int(minutes * 1.4)

    # Zone complexity factor (manual override + obstacle density)
    minutes = int(minutes * get_zone_complexity(zone))

    # Mode speed multiplier
    minutes = int(minutes * MODE_SPEED_MULTIPLIER.get(mode, 1.0))

    return max(minutes, 2)


def estimate_trim_minutes(zone: Zone, mode: str = "balanced") -> int:
    """
    Estimate trim time for a zone.
    Uses GPS actuals if available, otherwise best available linear footage.
    """
    # GPS actual override
    if zone.avg_actual_trim_minutes and zone.avg_actual_trim_minutes > 0:
        return max(int(zone.avg_actual_trim_minutes * MODE_SPEED_MULTIPLIER.get(mode, 1.0)), 2)

    linear_ft = get_trim_linear_ft(zone)
    minutes = max(int(linear_ft / TRIMMER_FT_PER_MIN), 2)

    # Slope penalty
    if zone.slope_grade > 10:
        minutes = int(minutes * 1.3)
    elif zone.slope_grade > 5:
        minutes = int(minutes * 1.15)

    # Zone complexity
    minutes = int(minutes * get_zone_complexity(zone))

    # Mode multiplier
    minutes = int(minutes * MODE_SPEED_MULTIPLIER.get(mode, 1.0))

    return max(minutes, 2)


def estimate_blow_minutes(all_zones: List[Zone], mode: str = "balanced") -> int:
    """
    Estimate blow/cleanup time.
    Uses real linear footage from blow/perimeter zones when available.
    """
    blowable_zones = [z for z in all_zones if z.zone_type not in ["no_mow"]]

    # Check for GPS actuals on any blow zone
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
        min_load_idx = section_loads.index(min(section_loads))
        sections[min_load_idx].append(zone)
        section_loads[min_load_idx] += zone.area_sqft
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
            subset_trimmers = [subset[-1]]

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


def assign_zones_to_crew(zones: List[Zone], crew: List[CrewMember], mode: str) -> List[TaskAssignment]:
    FALLBACK_WAGE = 20.60
    large_fields, small_mow, trim_zones, workable_zones = classify_zones(zones)

    for c in crew:
        if not c.hourly_rate or c.hourly_rate <= 0:
            c.hourly_rate = FALLBACK_WAGE

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

    if len(ordered_crew) < MIN_CREW_SIZE:
        ordered_crew = ordered_crew * MIN_CREW_SIZE

    if mode == "cheapest":
        subset = ordered_crew[:MIN_CREW_SIZE]
        assignments, job_time, wage_cost = run_subset(
            subset, large_fields, small_mow, trim_zones, zones, mode
        )
        print(f"LEAN CREW: {len(subset)} crew, {job_time:.0f} min, ${wage_cost:.2f}")
        return assignments

    elif mode == "fastest":
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
                best_time = job_time
                best_cost = wage_cost
                best_assignments = assignments
            else:
                break
        print(f"FASTEST final: {best_time:.0f}min")
        return best_assignments

    else:
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
                best_score = score
                best_assignments = assignments
            else:
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
        raise HTTPException(status_code=400, detail=f"Invalid mode: {request.mode}")

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

    crew_times: dict = {}
    for a in assignments:
        crew_times[a.crew_member_id] = crew_times.get(a.crew_member_id, 0) + a.estimated_minutes

    total_minutes = max(crew_times.values()) if crew_times else 0

    total_work = sum(a.estimated_minutes for a in assignments)
    recommended = max(MIN_CREW_SIZE, min(10, round(total_work / 45)))

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

    blow_zones = [z for z in request.zones if z.zone_type in ["perimeter", "trim"]]
    has_real_footage = any(z.linear_ft or z.perimeter_ft for z in blow_zones)
    if has_real_footage:
        total_blow_ft = sum(
            (z.linear_ft or z.perimeter_ft or 0) for z in blow_zones
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
