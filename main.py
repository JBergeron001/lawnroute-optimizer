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

# --- Models ---

class Zone(BaseModel):
    id: str
    label: str
    zone_type: str
    area_sqft: float
    slope_grade: float = 0
    equipment_restriction: Optional[str] = None
    perimeter_ft: Optional[float] = None

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

# --- Speed constants ---
# Walking speed = 3.1 mph
# Trimmer = normal walking speed
# Mowers = 1.5x walking speed
# Blower = 1.2x walking speed
# Edger = normal walking speed

CUTTING_SPEED = {
    "zero_turn": 4.65,
    "riding_mower": 4.65,
    "walk_behind": 4.65,
    "trimmer": 3.1,
    "blower": 3.72,
    "edger": 3.1,
}

DECK_WIDTH = {
    "zero_turn": 60,
    "riding_mower": 54,
    "walk_behind": 30,
    "trimmer": 1,
    "blower": 1,
    "edger": 1,
}

TRIMMER_FT_PER_MIN = 150

# Task-switch penalty in minutes (trailer trip + travel to new start)
TASK_SWITCH_PENALTY_MINUTES = 8

# Large field threshold — one mower handles alone unless above this
LARGE_FIELD_ACRES = 10.0

# --- Mode multipliers ---






# --- Estimators ---

def estimate_mow_minutes(zone: Zone, equipment_type: str, mode: str = "balanced") -> int:
    speed_mph = CUTTING_SPEED.get(equipment_type, 4.65)
    deck_ft = DECK_WIDTH.get(equipment_type, 30) / 12
    area_acres = zone.area_sqft / 43560
    efficiency = 0.80
    if mode == "fastest":
        efficiency = 0.88
    elif mode == "cheapest":
        efficiency = 0.75
    hours = area_acres / (speed_mph * deck_ft * efficiency)
    minutes = max(int(hours * 60), 2)
    if zone.slope_grade > 15:
        minutes = int(minutes * 1.5)
    elif zone.slope_grade > 10:
        minutes = int(minutes * 1.25)
    if zone.zone_type in ["berm", "courtyard"]:
        minutes = int(minutes * 1.4)

    return max(minutes, 2)

def estimate_trim_minutes(zone: Zone, mode: str = "balanced") -> int:
    if zone.perimeter_ft and zone.perimeter_ft > 0:
        linear_ft = zone.perimeter_ft
    else:
        linear_ft = 5.5 * math.sqrt(zone.area_sqft)
    minutes = max(int(linear_ft / TRIMMER_FT_PER_MIN), 2)
    if zone.slope_grade > 10:
        minutes = int(minutes * 1.3)

    return max(minutes, 2)

def estimate_blow_minutes(all_zones: List[Zone], mode: str = 'balanced') -> int:
    # Blow clears boundaries around trim zones - based on perimeter length not area
    BLOWER_FT_PER_MIN = 325  # blower walks at 3.72 mph along edges
    total_perimeter = 0
    for z in all_zones:
        if z.zone_type in ['trim', 'perimeter', 'mow', 'berm', 'courtyard']:
            if z.perimeter_ft and z.perimeter_ft > 0:
                total_perimeter += z.perimeter_ft
            else:
                total_perimeter += 4 * (z.area_sqft ** 0.5)
    minutes = max(int(total_perimeter / BLOWER_FT_PER_MIN), 5)
    return minutes

def get_mowers(crew: list) -> list:
    mowers = [c for c in crew if c.primary_role in ['zero_turn', 'walk_behind', 'riding_mower']]
    return sorted(mowers, key=lambda c: c.hourly_rate, reverse=True)

def get_trimmers(crew: list) -> list:
    return [c for c in crew if c.primary_role == 'trimmer']

def get_foreman(crew: list):
    foremen = [c for c in crew if c.is_foreman]
    return foremen[0] if foremen else None
def classify_zones(zones: list):
    """
    Separate zones into:
    - large_fields: single mow zones above threshold (keep together, one mower)
    - small_mow: smaller mow zones distributed across mowers
    - trim_zones: trim/perimeter zones
    - no_mow: skip
    """
    workable = [z for z in zones if z.zone_type != "no_mow"]
    mow_zones = [z for z in workable if z.zone_type in ["mow", "berm", "island", "courtyard"]]
    trim_zones = [z for z in workable if z.zone_type in ["trim", "perimeter"]]

    large_fields = [z for z in mow_zones
                    if z.area_sqft >= (LARGE_FIELD_ACRES * 43560)
                    and not z.equipment_restriction]
    small_mow = [z for z in mow_zones if z not in large_fields]

    return large_fields, small_mow, trim_zones, workable

# --- Core assignment logic ---

def assign_zones_to_crew(zones, crew, mode):
    FALLBACK_WAGE = 20.60
    large_fields, small_mow, trim_zones, workable_zones = classify_zones(zones)

    # Ensure all crew have a wage - use fallback if missing
    for c in crew:
        if not c.hourly_rate or c.hourly_rate == 0:
            c.hourly_rate = FALLBACK_WAGE

    # Sort crew: foreman first, then by wage descending
    foreman = get_foreman(crew)
    if foreman:
        others = sorted([c for c in crew if c.id != foreman.id], key=lambda c: c.hourly_rate, reverse=True)
        ordered_crew = [foreman] + others
    else:
        ordered_crew = sorted(crew, key=lambda c: c.hourly_rate, reverse=True)

    # Enforce minimum 2 crew
    if len(ordered_crew) < 2:
        ordered_crew = ordered_crew + ordered_crew

    def run_subset(subset, mode):
        subset_load = {c.id: 0.0 for c in subset}
        subset_assignments = []
        counter = 1

        subset_mowers = [c for c in subset if c.primary_role in ['zero_turn', 'walk_behind', 'riding_mower']]
        subset_trimmers = [c for c in subset if c.primary_role == 'trimmer']
        if not subset_mowers:
            subset_mowers = [subset[0]]
        if not subset_trimmers:
            non_foreman = [c for c in subset if not c.is_foreman and c.id != subset[0].id]
            subset_trimmers = [non_foreman[-1]] if non_foreman else [subset[-1]]
            subset_mowers = [c for c in subset_mowers if c not in subset_trimmers] or [subset[0]]

        # --- MOW ZONE DISTRIBUTION ---
        # Distribute zones across mowers by load balancing
        # If more mowers than zones, split total mow time equally
        all_mow = sorted(large_fields + small_mow, key=lambda z: z.area_sqft, reverse=True)
        total_mow_mins = sum(estimate_mow_minutes(z, subset_mowers[0].primary_role if subset_mowers[0].primary_role in CUTTING_SPEED else 'walk_behind', mode) for z in all_mow)
        if len(subset_mowers) >= len(all_mow) and len(all_mow) > 0:
            # More mowers than zones - give each mower equal share of total time
            mins_per_mower = total_mow_mins / len(subset_mowers)
            for i, worker in enumerate(subset_mowers):
                eq = worker.primary_role if worker.primary_role in CUTTING_SPEED else 'walk_behind'
                subset_assignments.append(TaskAssignment(
                    crew_member_id=worker.id, crew_member_name=worker.name,
                    task_order=counter, zone_id=all_mow[0].id, zone_label=f'Mow Section {i+1}',
                    task_type='mow', estimated_minutes=int(mins_per_mower), role_used=eq, is_role_switch=False
                ))
                subset_load[worker.id] = subset_load.get(worker.id, 0) + mins_per_mower
                counter += 1
        else:
            # Distribute zones across mowers by load balancing
            for zone in all_mow:
                worker = min(subset_mowers, key=lambda c: subset_load.get(c.id, 0))
                eq = worker.primary_role if worker.primary_role in CUTTING_SPEED else 'walk_behind'
                mins = estimate_mow_minutes(zone, eq, mode)
                subset_assignments.append(TaskAssignment(
                    crew_member_id=worker.id, crew_member_name=worker.name,
                    task_order=counter, zone_id=zone.id, zone_label=zone.label,
                    task_type='mow', estimated_minutes=mins, role_used=eq, is_role_switch=False
                ))
                subset_load[worker.id] = subset_load.get(worker.id, 0) + mins
                counter += 1



        # --- TRIM ZONE SPLITTING ---
        # If more trimmers than zones, split largest trim zones
        all_trim = sorted(trim_zones, key=lambda z: estimate_trim_minutes(z, mode), reverse=True)
        expanded_trim = []
        for zone in all_trim:
            if len(subset_trimmers) > len(expanded_trim) + (len(all_trim) - all_trim.index(zone) - 1):
                idle_trimmers = len(subset_trimmers) - len(expanded_trim)
                if idle_trimmers > 1:
                    section_sqft = zone.area_sqft / idle_trimmers
                    for i in range(idle_trimmers):
                        from copy import copy
                        z_copy = copy(zone)
                        z_copy.area_sqft = section_sqft
                        z_copy.label = f'{zone.label} (Section {i+1})'
                        expanded_trim.append(z_copy)
                else:
                    expanded_trim.append(zone)
            else:
                expanded_trim.append(zone)

        for zone in expanded_trim:
            worker = min(subset_trimmers, key=lambda c: subset_load.get(c.id, 0))
            prev = [a for a in subset_assignments if a.crew_member_id == worker.id]
            is_switch = bool(prev) and prev[-1].task_type == 'mow'
            mins = estimate_trim_minutes(zone, mode) + (TASK_SWITCH_PENALTY_MINUTES if is_switch else 0)
            subset_assignments.append(TaskAssignment(
                crew_member_id=worker.id, crew_member_name=worker.name,
                task_order=counter, zone_id=zone.id, zone_label=zone.label,
                task_type='trim', estimated_minutes=mins, role_used='trimmer', is_role_switch=is_switch
            ))
            subset_load[worker.id] = subset_load.get(worker.id, 0) + mins
            counter += 1

        # --- BLOW TRIGGERING ---
        # Blow starts when mow+trim is 85% complete
        # Whoever finishes mow/trim first picks up the blower
        mow_trim_time = max(subset_load.values()) if subset_load else 0
        blow_start = mow_trim_time * 0.85
        blow_mins = estimate_blow_minutes(zones, mode)
        # Find who finishes mow+trim first
        mow_trim_workers = subset_mowers + [t for t in subset_trimmers if t not in subset_mowers]
        blow_worker = min(mow_trim_workers, key=lambda c: subset_load.get(c.id, 0))
        # Blow time = remaining blow after mow+trim overlap
        overlap = max(0, mow_trim_time - blow_start)
        effective_blow = max(blow_mins - overlap, 5)
        subset_assignments.append(TaskAssignment(
            crew_member_id=blow_worker.id, crew_member_name=blow_worker.name,
            task_order=counter, zone_id='cleanup', zone_label='Final Blowout & Cleanup',
            task_type='blow', estimated_minutes=int(effective_blow), role_used='blower', is_role_switch=True
        ))
        subset_load[blow_worker.id] = subset_load.get(blow_worker.id, 0) + effective_blow

        job_time = max(subset_load.values())
        wage_cost = sum(subset_load.get(c.id, 0) * c.hourly_rate / 60 for c in subset)
        return subset_assignments, job_time, wage_cost
    if mode == 'cheapest':
        # Start with 2 crew, add more only if wage bill goes DOWN
        best_assignments, best_time, best_cost = run_subset(ordered_crew[:2], mode)
        print(f'CHEAPEST size=2 cost={best_cost:.2f}')
        for size in range(3, len(ordered_crew) + 1):
            subset = ordered_crew[:size]
            assignments, job_time, wage_cost = run_subset(subset, mode)
            print(f'CHEAPEST size={size} cost={wage_cost:.2f}')
            if wage_cost < best_cost:
                best_cost = wage_cost
                best_assignments = assignments
                best_time = job_time
            else:
                pass  # continue trying all sizes
        print(f'CHEAPEST final cost=')
        return best_assignments

    elif mode == 'fastest':
        # Find crew size that minimizes job completion time
        best_assignments, best_time, best_cost = run_subset(ordered_crew[:2], mode)
        print(f'FASTEST size=2 time={best_time:.0f}min')
        for size in range(3, len(ordered_crew) + 1):
            subset = ordered_crew[:size]
            assignments, job_time, wage_cost = run_subset(subset, mode)
            print(f'FASTEST size={size} time={job_time:.0f}min')
            if job_time < best_time:
                best_time = job_time
                best_assignments = assignments
            else:
                pass  # continue trying all sizes
        print(f'FASTEST final time={best_time:.0f}min')
        return best_assignments

    else:  # balanced
        # Find best time/cost ratio
        best_assignments, best_time, best_cost = run_subset(ordered_crew[:2], mode)
        best_score = best_time * best_cost
        print(f'BALANCED size=2 score={best_score:.2f}')
        for size in range(3, len(ordered_crew) + 1):
            subset = ordered_crew[:size]
            assignments, job_time, wage_cost = run_subset(subset, mode)
            score = job_time * wage_cost
            print(f'BALANCED size={size} score={score:.2f}')
            if score < best_score:
                best_score = score
                best_assignments = assignments
            else:
                pass  # continue trying all sizes
        print(f'BALANCED final score={best_score:.2f}')
        return best_assignments
# --- Routes ---

@app.get("/health")
def health():
    return {"status": "ok", "service": "optimizer"}

@app.post("/optimize", response_model=OptimizeResponse)
def optimize(request: OptimizeRequest):
    start = time.time()

    if not request.zones:
        raise HTTPException(status_code=400, detail="No zones provided")
    if not request.crew:
        raise HTTPException(status_code=400, detail="No crew provided")

    all_crew = list(request.crew) + list(request.available_crew or [])
    assignments = assign_zones_to_crew(request.zones, all_crew, request.mode)
    crew_times = {}
    for a in assignments:
        crew_times[a.crew_member_id] = crew_times.get(a.crew_member_id, 0) + a.estimated_minutes

    total_minutes = max(crew_times.values()) if crew_times else 0

    total_work = sum(a.estimated_minutes for a in assignments)
    recommended = max(2, min(10, round(total_work / 45)))

    # Build crew split suggestion
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

    crew_split = f"{mower_count} mow / {trimmer_count} trim"
    blow_note = "Whoever finishes first picks up the blower"

    solve_ms = int((time.time() - start) * 1000)

    return OptimizeResponse(
        job_id=request.job_id,
        total_estimated_minutes=total_minutes,
        crew_assignments=assignments,
        solve_time_ms=solve_ms,
        recommended_crew_size=recommended,
        mode=request.mode,
        crew_split_suggestion=crew_split,
        blow_note=blow_note,
    )

@app.get("/recommend-crew")
def recommend_crew(total_sqft: float, zone_types: str = "mow,trim"):
    zones_list = zone_types.split(",")
    has_slopes = "berm" in zones_list or "slope" in zones_list
    has_courtyards = "courtyard" in zones_list
    acres = total_sqft / 43560

    lean = max(2, int(acres / 2))
    optimal = max(3, int(acres / 1.5))
    fast = max(4, int(acres / 1))

    if has_slopes:
        lean += 1
        optimal += 1
    if has_courtyards:
        optimal += 1

    return {
        "lean": lean,
        "optimal": optimal,
        "fast": fast,
        "acres": round(acres, 2)
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

