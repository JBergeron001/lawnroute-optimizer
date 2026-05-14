"""
LawnRoute Task Optimizer — main.py

Calibration basis:
  Meadowbrook CC Estates — 11 acres, complex layout, 5 crew, 6-7 hours real time.
  Effective mow rate: ~0.46 acres/person/hour on a complex site.

Deck widths:
  60 inch (5.0 ft) zero-turn — open mow zones >= 0.5 acres
  36 inch (3.0 ft) walk-behind — complex zones (berm, courtyard, island, slope)
                                  and small mow zones < 0.5 acres

Fixed overhead per job:
  Setup/arrival:   7 minutes (unload, walkthrough, prep)
  End walkthrough: 7 minutes (foreman quality check before leaving)
  Zone transition: 2.5 minutes per zone change (mower repositioning)

Crew assignment model:

  Crew of 2:
    Both mow simultaneously.
    First to finish mowing trims all trim zones.
    Second to finish mowing blows.
    Job time = when last task finishes.

  Crew of 3+:
    Trimmer count based on workload — NOT acreage:
      Estimate total trim time for 1 person.
      Estimate mow time for remaining crew.
      If 1 trimmer finishes before mowers → 1 trimmer (they blow after).
      If trim time > mow time → 2 trimmers to avoid trim being the bottleneck.
      Never more than 2 trimmers regardless of site size.
    Trimmer(s) are cheapest worker(s). Ties broken by first in list only.
    Trimmer never mows. Mowers never trim.
    Mowing and trimming run in parallel.
    Whoever finishes their primary task first picks up the blower.
    Job time = max(slowest mower, trimmer finish time) + blow time for finisher.

  Travel between zones:
    2.5 minutes added per zone transition for each mower.
    Phase 2: replace with actual PostGIS distances between zone centroids.
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

# ─── Constants ────────────────────────────────────────────────────────────────

MOW_SPEED_MPH   = 4.5
DECK_FT_OPEN    = 5.0    # 60 inch zero-turn
DECK_FT_COMPLEX = 3.0    # 36 inch walk-behind

COMPLEX_ZONE_TYPES = {"berm", "courtyard", "island", "slope"}

# Fixed overhead added to every job
JOB_SETUP_MINUTES      = 7    # arrival, unload, walkthrough
JOB_WALKTHROUGH_MINUTES = 7   # foreman end-of-job quality check
ZONE_TRAVEL_MINUTES    = 2.5  # repositioning between zones per mower
TASK_SWITCH_PENALTY    = 6    # tool swap time

MIN_CREW_SIZE = 2
LARGE_FIELD_ACRES = 10.0

BASE_EFFICIENCY = {
    "fastest":  0.42,
    "balanced": 0.35,   # calibrated from Meadowbrook ground truth
    "cheapest": 0.30,
}

MODE_SPEED_MULTIPLIER = {
    "fastest":  0.90,
    "balanced": 1.00,
    "cheapest": 1.15,
}

TRIMMER_FT_PER_MIN = 130.0
BLOWER_FT_PER_MIN  = 140.0

# ─── Deck Width ───────────────────────────────────────────────────────────────

def get_deck_ft(zone: Zone) -> float:
    if zone.zone_type in COMPLEX_ZONE_TYPES:
        return DECK_FT_COMPLEX
    if zone.zone_type == "mow" and zone.area_sqft < (0.5 * 43560):
        return DECK_FT_COMPLEX
    return DECK_FT_OPEN

# ─── Acreage-Based Crew Floor ─────────────────────────────────────────────────

def get_optimal_crew_floor(zones: List[Zone]) -> int:
    mow_zones = [z for z in zones if z.zone_type in [
        "mow", "berm", "island", "courtyard", "slope"
    ]]
    acres = sum(z.area_sqft for z in mow_zones) / 43560.0
    if acres < 1.0:    return 2
    elif acres < 3.0:  return 2
    elif acres < 8.0:  return 3
    elif acres < 20.0: return 4
    elif acres < 40.0: return 5
    else:              return 6

# ─── Turn Penalty ─────────────────────────────────────────────────────────────

def get_turn_penalty(area_sqft: float) -> float:
    acres = area_sqft / 43560.0
    if acres < 0.25:   return 1.55
    elif acres < 0.5:  return 1.40
    elif acres < 1.0:  return 1.28
    elif acres < 2.0:  return 1.18
    elif acres < 5.0:  return 1.10
    elif acres < 15.0: return 1.05
    else:              return 1.02

# ─── Measurement Helpers ──────────────────────────────────────────────────────

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

def estimate_mow_minutes(zone: Zone, mode: str = "balanced") -> int:
    # Real historical data overrides calculation — self-learning
    if zone.avg_actual_mow_minutes and zone.avg_actual_mow_minutes > 0:
        return max(int(zone.avg_actual_mow_minutes * MODE_SPEED_MULTIPLIER.get(mode, 1.0)), 2)

    deck_ft = get_deck_ft(zone)
    efficiency = BASE_EFFICIENCY.get(mode, 0.35)
    area_acres = zone.area_sqft / 43560.0
    hours = area_acres / (MOW_SPEED_MPH * deck_ft * efficiency)
    minutes = max(int(hours * 60), 2)

    minutes = int(minutes * get_turn_penalty(zone.area_sqft))

    if zone.slope_grade > 15:   minutes = int(minutes * 1.7)
    elif zone.slope_grade > 10: minutes = int(minutes * 1.4)
    elif zone.slope_grade > 5:  minutes = int(minutes * 1.18)

    if zone.zone_type == "berm":        minutes = int(minutes * 1.55)
    elif zone.zone_type == "courtyard": minutes = int(minutes * 1.50)
    elif zone.zone_type == "island":    minutes = int(minutes * 1.35)
    elif zone.zone_type == "slope":     minutes = int(minutes * 1.45)

    minutes = int(minutes * get_zone_complexity(zone))
    minutes = int(minutes * MODE_SPEED_MULTIPLIER.get(mode, 1.0))
    return max(minutes, 2)


def estimate_trim_minutes(zone: Zone, mode: str = "balanced") -> int:
    if zone.avg_actual_trim_minutes and zone.avg_actual_trim_minutes > 0:
        return max(int(zone.avg_actual_trim_minutes * MODE_SPEED_MULTIPLIER.get(mode, 1.0)), 2)

    linear_ft = get_trim_linear_ft(zone)
    minutes = max(int(linear_ft / TRIMMER_FT_PER_MIN), 2)

    if zone.slope_grade > 10:  minutes = int(minutes * 1.3)
    elif zone.slope_grade > 5: minutes = int(minutes * 1.15)

    minutes = int(minutes * get_zone_complexity(zone))
    minutes = int(minutes * MODE_SPEED_MULTIPLIER.get(mode, 1.0))
    return max(minutes, 2)


def estimate_blow_minutes(all_zones: List[Zone], mode: str = "balanced") -> int:
    blowable = [z for z in all_zones if z.zone_type != "no_mow"]
    actual = next(
        (z.avg_actual_blow_minutes for z in blowable
         if z.avg_actual_blow_minutes and z.avg_actual_blow_minutes > 0),
        None
    )
    if actual:
        return max(int(actual * MODE_SPEED_MULTIPLIER.get(mode, 1.0)), 10)
    total_ft = sum(get_blow_linear_ft(z) for z in blowable)
    minutes = max(int(total_ft / BLOWER_FT_PER_MIN), 10)
    return max(int(minutes * MODE_SPEED_MULTIPLIER.get(mode, 1.0)), 10)

# ─── Zone Classification ──────────────────────────────────────────────────────

def classify_zones(zones: List[Zone]):
    workable   = [z for z in zones if z.zone_type != "no_mow"]
    mow_zones  = [z for z in workable if z.zone_type in [
        "mow", "berm", "island", "courtyard", "slope"
    ]]
    trim_zones = [z for z in workable if z.zone_type in ["trim", "perimeter"]]
    large_fields = [
        z for z in mow_zones
        if z.area_sqft >= (LARGE_FIELD_ACRES * 43560) and not z.equipment_restriction
    ]
    small_mow = [z for z in mow_zones if z not in large_fields]
    return large_fields, small_mow, trim_zones, workable

# ─── Crew Helpers ─────────────────────────────────────────────────────────────

def fill_hourly_rates(crew: List[CrewMember], fallback: float = 20.60):
    for c in crew:
        if not c.hourly_rate or c.hourly_rate <= 0:
            c.hourly_rate = fallback

def order_crew(crew: List[CrewMember]) -> List[CrewMember]:
    foremen = [c for c in crew if c.is_foreman]
    others  = sorted([c for c in crew if not c.is_foreman],
                     key=lambda c: c.hourly_rate, reverse=True)
    return foremen + others

def select_trimmers(
    crew: List[CrewMember],
    trim_zones: List[Zone],
    mow_crew_count: int,
    all_mow_zones: List[Zone],
    mode: str
) -> List[CrewMember]:
    """
    Select dedicated trimmer(s) based on trim workload vs mow workload.

    Logic:
    1. Estimate total trim time for 1 person
    2. Estimate mow time for the mowing crew (longest section = bottleneck)
    3. If 1 trimmer finishes before or with mowers → 1 trimmer
    4. If trim time > mow time → 2 trimmers to prevent trim bottleneck
    5. Never more than 2 trimmers
    6. Always cheapest worker(s) — ties broken by first in list only
    """
    if not trim_zones:
        return []

    # Estimate total trim time for 1 person
    total_trim_minutes = sum(estimate_trim_minutes(z, mode) for z in trim_zones)

    # Estimate mow bottleneck — divide total mow time by mower count
    total_mow_minutes = sum(estimate_mow_minutes(z, mode) for z in all_mow_zones)
    mow_bottleneck = total_mow_minutes / max(mower_count := max(mow_crew_count, 1), 1)

    print(f"TRIM: total_trim={total_trim_minutes:.0f}min mow_bottleneck={mow_bottleneck:.0f}min")

    # 1 trimmer unless trim is the bottleneck
    needed = 1
    if total_trim_minutes > mow_bottleneck and needed < 2:
        needed = 2
        print(f"TRIM: adding 2nd trimmer — trim ({total_trim_minutes:.0f}min) > mow ({mow_bottleneck:.0f}min)")

    # Cap at 2, cap at crew size - 1 (always keep at least 1 mower)
    needed = min(needed, 2, len(crew) - 1)
    if needed <= 0:
        return []

    # Cheapest worker(s) — ties broken by first in list
    sorted_crew = sorted(crew, key=lambda c: c.hourly_rate)
    return sorted_crew[:needed]


def divide_into_sections(mow_zones: List[Zone], mower_count: int) -> List[List[Zone]]:
    if not mow_zones or mower_count <= 0:
        return []
    num_sections = min(mower_count, len(mow_zones))
    if num_sections <= 1:
        return [mow_zones]
    sorted_zones = sorted(mow_zones, key=lambda z: z.area_sqft, reverse=True)
    sections: List[List[Zone]] = [[] for _ in range(num_sections)]
    loads = [0.0] * num_sections
    for zone in sorted_zones:
        idx = loads.index(min(loads))
        sections[idx].append(zone)
        loads[idx] += zone.area_sqft
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

    fill_hourly_rates(subset)

    all_mow = sorted(large_fields + small_mow, key=lambda z: z.area_sqft, reverse=True)
    blow_mins = estimate_blow_minutes(all_zones, mode)
    assignments = []
    counter = 1
    crew_load = {c.id: 0.0 for c in subset}

    # Fixed setup time — added to foreman (or first crew member)
    foreman = next((c for c in subset if c.is_foreman), subset[0])
    crew_load[foreman.id] += JOB_SETUP_MINUTES

    # ── Crew of 2: both mow, first done trims, second blows ───────────────────
    if len(subset) == 2:
        ordered = order_crew(subset)
        sections = divide_into_sections(all_mow, 2)

        for i, member in enumerate(ordered):
            section = sections[i] if i < len(sections) else []
            for j, zone in enumerate(section):
                mins = estimate_mow_minutes(zone, mode)
                # Add zone travel time between zones (not before the first)
                if j > 0:
                    mins += ZONE_TRAVEL_MINUTES
                assignments.append(TaskAssignment(
                    crew_member_id=member.id,
                    crew_member_name=member.name,
                    task_order=counter,
                    zone_id=zone.id,
                    zone_label=zone.label,
                    task_type="mow",
                    estimated_minutes=int(mins),
                    role_used="mow",
                    is_role_switch=False,
                ))
                crew_load[member.id] += mins
                counter += 1

        # First to finish mowing → trims
        first_done  = min(ordered, key=lambda c: crew_load.get(c.id, 0))
        second_done = max(ordered, key=lambda c: crew_load.get(c.id, 0))

        for zone in sorted(trim_zones, key=lambda z: estimate_trim_minutes(z, mode), reverse=True):
            mins = estimate_trim_minutes(zone, mode) + TASK_SWITCH_PENALTY
            assignments.append(TaskAssignment(
                crew_member_id=first_done.id,
                crew_member_name=first_done.name,
                task_order=counter,
                zone_id=zone.id,
                zone_label=zone.label,
                task_type="trim",
                estimated_minutes=int(mins),
                role_used="trimmer",
                is_role_switch=True,
            ))
            crew_load[first_done.id] += mins
            counter += 1

        # Second to finish mowing → blows
        blow_total = blow_mins + TASK_SWITCH_PENALTY
        assignments.append(TaskAssignment(
            crew_member_id=second_done.id,
            crew_member_name=second_done.name,
            task_order=counter,
            zone_id="cleanup",
            zone_label="Final Blowout & Cleanup",
            task_type="blow",
            estimated_minutes=int(blow_total),
            role_used="blower",
            is_role_switch=True,
        ))
        crew_load[second_done.id] += blow_total

    # ── Crew of 3+: dedicated trimmer(s), parallel streams ────────────────────
    else:
        # Determine trimmer count based on workload
        trimmers = select_trimmers(
            subset, trim_zones,
            mow_crew_count=len(subset) - 1,  # conservative estimate
            all_mow_zones=all_mow,
            mode=mode
        )
        trimmer_ids = {c.id for c in trimmers}
        mowers = [c for c in subset if c.id not in trimmer_ids]

        if not mowers:
            mowers = list(subset)
            trimmers = []
            trimmer_ids = set()

        ordered_mowers = order_crew(mowers)
        sections = divide_into_sections(all_mow, len(ordered_mowers))

        # Mowers work their sections simultaneously
        for i, member in enumerate(ordered_mowers):
            section = sections[i] if i < len(sections) else []
            for j, zone in enumerate(section):
                mins = estimate_mow_minutes(zone, mode)
                # Zone travel time between zones
                if j > 0:
                    mins += ZONE_TRAVEL_MINUTES
                assignments.append(TaskAssignment(
                    crew_member_id=member.id,
                    crew_member_name=member.name,
                    task_order=counter,
                    zone_id=zone.id,
                    zone_label=zone.label,
                    task_type="mow",
                    estimated_minutes=int(mins),
                    role_used="mow",
                    is_role_switch=False,
                ))
                crew_load[member.id] += mins
                counter += 1

        # Trimmer(s) work all trim zones in parallel — never mow
        if trimmers and trim_zones:
            trim_sorted = sorted(trim_zones,
                                 key=lambda z: estimate_trim_minutes(z, mode),
                                 reverse=True)
            trimmer_list = list(trimmers)
            for i, zone in enumerate(trim_sorted):
                trimmer = trimmer_list[i % len(trimmer_list)]
                mins = estimate_trim_minutes(zone, mode)
                assignments.append(TaskAssignment(
                    crew_member_id=trimmer.id,
                    crew_member_name=trimmer.name,
                    task_order=counter,
                    zone_id=zone.id,
                    zone_label=zone.label,
                    task_type="trim",
                    estimated_minutes=int(mins),
                    role_used="trimmer",
                    is_role_switch=False,
                ))
                crew_load[trimmer.id] += mins
                counter += 1

        # Blow — whoever finishes their primary task first
        blow_worker = min(subset, key=lambda c: crew_load.get(c.id, 0))
        blow_total = blow_mins + TASK_SWITCH_PENALTY
        assignments.append(TaskAssignment(
            crew_member_id=blow_worker.id,
            crew_member_name=blow_worker.name,
            task_order=counter,
            zone_id="cleanup",
            zone_label="Final Blowout & Cleanup",
            task_type="blow",
            estimated_minutes=int(blow_total),
            role_used="blower",
            is_role_switch=True,
        ))
        crew_load[blow_worker.id] += blow_total

    # Fixed end walkthrough — added to foreman
    crew_load[foreman.id] += JOB_WALKTHROUGH_MINUTES

    job_time = max(crew_load.values())
    wage_cost = sum(
        crew_load.get(c.id, 0) * c.hourly_rate / 60.0
        for c in subset
    )
    return assignments, job_time, wage_cost

# ─── Scenario Functions ───────────────────────────────────────────────────────

def scenario_lean(zones: List[Zone], available_crew: List[CrewMember]) -> ScenarioResult:
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


def scenario_optimal(zones: List[Zone], available_crew: List[CrewMember]) -> ScenarioResult:
    large_fields, small_mow, trim_zones, _ = classify_zones(zones)
    ordered = order_crew(available_crew)
    max_crew = min(len(ordered), 10)
    floor = min(get_optimal_crew_floor(zones), max_crew)

    best_assignments, best_time, best_cost = run_subset(
        ordered[:floor], large_fields, small_mow, trim_zones, zones, "balanced"
    )
    best_size = floor
    print(f"OPTIMAL floor={floor} time={best_time:.0f}min")

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
    large_fields, small_mow, trim_zones, _ = classify_zones(zones)
    ordered = order_crew(available_crew)
    size = min(optimal_size + 2, len(ordered))
    subset = ordered[:size]
    assignments, job_time, wage_cost = run_subset(
        subset, large_fields, small_mow, trim_zones, zones, "fastest"
    )
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
        lean_result     = scenario_lean(request.zones, all_crew)
        optimal_result  = scenario_optimal(request.zones, all_crew)
        max_result      = scenario_max_speed(request.zones, all_crew, optimal_result.crew_size)
        assigned_result = scenario_assigned(request.zones, list(request.crew))
    except Exception as e:
        import traceback
        print(f"OPTIMIZER ERROR: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Optimizer error: {str(e)}")

    blow_zones = [z for z in request.zones if z.zone_type in ["perimeter", "trim"]]
    has_real_footage = any(z.linear_ft or z.perimeter_ft for z in blow_zones)
    if has_real_footage:
        total_ft = sum((z.linear_ft or z.perimeter_ft or 0) for z in blow_zones)
        blow_note = f"Whoever finishes first blows — approx {int(total_ft):,} ft"
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
    has_slopes     = "berm" in zones_list or "slope" in zones_list
    has_courtyards = "courtyard" in zones_list
    has_large_trim = "perimeter" in zones_list
    acres = total_sqft / 43560.0

    lean    = max(MIN_CREW_SIZE, int(acres / 2))
    optimal = max(3, int(acres / 1.5))
    fast    = max(4, int(acres / 1.0))

    if has_slopes:     lean += 1; optimal += 1
    if has_courtyards: optimal += 1
    if has_large_trim: lean += 1

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
