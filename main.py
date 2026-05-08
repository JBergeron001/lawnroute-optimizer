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
MODE_SPEED_MULTIPLIER = {
    "fastest": 0.85,
    "balanced": 1.0,
    "cheapest": 1.15,
}

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
    minutes = int(minutes * MODE_SPEED_MULTIPLIER.get(mode, 1.0))
    return max(minutes, 2)

def estimate_trim_minutes(zone: Zone, mode: str = "balanced") -> int:
    if zone.perimeter_ft and zone.perimeter_ft > 0:
        linear_ft = zone.perimeter_ft
    else:
        linear_ft = 5.5 * math.sqrt(zone.area_sqft)
    minutes = max(int(linear_ft / TRIMMER_FT_PER_MIN), 2)
    if zone.slope_grade > 10:
        minutes = int(minutes * 1.3)
    minutes = int(minutes * MODE_SPEED_MULTIPLIER.get(mode, 1.0))
    return max(minutes, 2)

def estimate_blow_minutes(all_zones: List[Zone], mode: str = "balanced") -> int:
    total_sqft = sum(z.area_sqft for z in all_zones if z.zone_type != "no_mow")
    total_acres = total_sqft / 43560
    minutes = max(int(total_acres * 12), 10)
    minutes = int(minutes * MODE_SPEED_MULTIPLIER.get(mode, 1.0))
    return max(minutes, 5)

# --- Role classification helpers ---

def get_mowers(crew: list) -> list:
    """Return crew sorted by wage descending — highest paid mow."""
    mowers = [c for c in crew if c.primary_role in ["zero_turn", "walk_behind", "riding_mower"]]
    return sorted(mowers, key=lambda c: c.hourly_rate, reverse=True)

def get_trimmers(crew: list) -> list:
    """Return crew assigned to trim — lowest paid non-foreman workers."""
    return [c for c in crew if c.primary_role == "trimmer"]

def get_foreman(crew: list):
    """Return foreman if present."""
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
    assignments = []
    task_counter = 1
    crew_load = {c.id: 0 for c in crew}

    large_fields, small_mow, trim_zones, workable_zones = classify_zones(zones)

    mowers = get_mowers(crew)
    trimmers = get_trimmers(crew)
    all_crew = crew

    # If no dedicated trimmers, pull lowest-paid non-foreman mowers for trim
    if not trimmers:
        non_foreman_mowers = [c for c in mowers if not c.is_foreman]
        if non_foreman_mowers:
            # Pull bottom wage earners for trim based on crew size
            trim_count = max(1, len(crew) // 4) if len(crew) >= 4 else 1
            trimmers = non_foreman_mowers[-trim_count:]
            mowers = [c for c in mowers if c not in trimmers]

    # --- MODE: CHEAPEST ---
    if mode == 'cheapest':
        leader = get_foreman(crew)
        print(f'CHEAPEST DEBUG: crew={len(crew)}, foreman={leader.name if leader else None}')
        if not leader:
            leader = max(crew, key=lambda c: c.hourly_rate)
        others = sorted([c for c in crew if c.id != leader.id], key=lambda c: c.hourly_rate)
        # Pick cheapest trimmer - lowest paid non-foreman
        non_foreman = [c for c in crew if not c.is_foreman and c.id != leader.id]
        trimmer = min(non_foreman, key=lambda c: c.hourly_rate) if non_foreman else leader
        cheap_assignments = []
        counter = 1
        all_mow = sorted(large_fields + small_mow, key=lambda z: z.area_sqft, reverse=True)
        for zone in all_mow:
            eq = leader.primary_role if leader.primary_role in CUTTING_SPEED else 'walk_behind'
            mins = estimate_mow_minutes(zone, eq, mode)
            cheap_assignments.append(TaskAssignment(crew_member_id=leader.id, crew_member_name=leader.name, task_order=counter, zone_id=zone.id, zone_label=zone.label, task_type='mow', estimated_minutes=mins, role_used=eq, is_role_switch=False))
            counter += 1
        for zone in sorted(trim_zones, key=lambda z: estimate_trim_minutes(z, mode), reverse=True):
            mins = estimate_trim_minutes(zone, mode)
            cheap_assignments.append(TaskAssignment(crew_member_id=trimmer.id, crew_member_name=trimmer.name, task_order=counter, zone_id=zone.id, zone_label=zone.label, task_type='trim', estimated_minutes=mins, role_used='trimmer', is_role_switch=False))
            counter += 1
        blow_mins = estimate_blow_minutes(zones, mode) + TASK_SWITCH_PENALTY_MINUTES
        cheap_assignments.append(TaskAssignment(crew_member_id=trimmer.id, crew_member_name=trimmer.name, task_order=counter, zone_id='cleanup', zone_label='Final Blowout & Cleanup', task_type='blow', estimated_minutes=blow_mins, role_used='blower', is_role_switch=True))
        return cheap_assignments

    # --- MODE: FASTEST / BALANCED ---

    # PHASE 1: Large fields — assign one dedicated mower each
    # Only split a large field if extraordinarily large (>10 acres) AND 2+ mowers available
    for zone in large_fields:
        if mowers:
            # Pick least loaded mower
            worker = min(mowers, key=lambda c: crew_load.get(c.id, 0))
        else:
            worker = min(all_crew, key=lambda c: crew_load.get(c.id, 0))

        eq = worker.primary_role if worker.primary_role in CUTTING_SPEED else "walk_behind"
        mins = estimate_mow_minutes(zone, eq, mode)

        # Extraordinarily large field (>10 acres) with 2+ mowers — split it
        field_acres = zone.area_sqft / 43560
        if field_acres > LARGE_FIELD_ACRES and len(mowers) >= 2:
            # Two mowers split the field — each gets half the time
            half_mins = max(mins // 2, 2)
            second_mower = min(
                [m for m in mowers if m.id != worker.id],
                key=lambda c: crew_load.get(c.id, 0)
            )
            eq2 = second_mower.primary_role if second_mower.primary_role in CUTTING_SPEED else "walk_behind"

            assignments.append(TaskAssignment(
                crew_member_id=worker.id,
                crew_member_name=worker.name,
                task_order=task_counter,
                zone_id=zone.id,
                zone_label=f"{zone.label} (North Half)",
                task_type="mow",
                estimated_minutes=half_mins,
                role_used=eq,
                is_role_switch=False
            ))
            crew_load[worker.id] = crew_load.get(worker.id, 0) + half_mins
            task_counter += 1

            assignments.append(TaskAssignment(
                crew_member_id=second_mower.id,
                crew_member_name=second_mower.name,
                task_order=task_counter,
                zone_id=zone.id,
                zone_label=f"{zone.label} (South Half)",
                task_type="mow",
                estimated_minutes=half_mins,
                role_used=eq2,
                is_role_switch=False
            ))
            crew_load[second_mower.id] = crew_load.get(second_mower.id, 0) + half_mins
            task_counter += 1
        else:
            # Normal large field — one mower handles it alone
            assignments.append(TaskAssignment(
                crew_member_id=worker.id,
                crew_member_name=worker.name,
                task_order=task_counter,
                zone_id=zone.id,
                zone_label=zone.label,
                task_type="mow",
                estimated_minutes=mins,
                role_used=eq,
                is_role_switch=False
            ))
            crew_load[worker.id] = crew_load.get(worker.id, 0) + mins
            task_counter += 1

    # PHASE 2: Small mow zones — distribute across mowers by load balancing
    small_mow.sort(key=lambda z: z.area_sqft, reverse=True)

    for zone in small_mow:
        if mowers:
            worker = min(mowers, key=lambda c: crew_load.get(c.id, 0))
        else:
            worker = min(all_crew, key=lambda c: crew_load.get(c.id, 0))

        eq = worker.primary_role if worker.primary_role in CUTTING_SPEED else "walk_behind"
        mins = estimate_mow_minutes(zone, eq, mode)
        assignments.append(TaskAssignment(
            crew_member_id=worker.id,
            crew_member_name=worker.name,
            task_order=task_counter,
            zone_id=zone.id,
            zone_label=zone.label,
            task_type="mow",
            estimated_minutes=mins,
            role_used=eq,
            is_role_switch=False
        ))
        crew_load[worker.id] = crew_load.get(worker.id, 0) + mins
        task_counter += 1

    # PHASE 3: Trim zones — assign to trimmers, balance load
    trim_zones.sort(key=lambda z: estimate_trim_minutes(z, mode), reverse=True)
    trim_workers = trimmers if trimmers else all_crew

    for zone in trim_zones:
        worker = min(trim_workers, key=lambda c: crew_load.get(c.id, 0))
        mins = estimate_trim_minutes(zone, mode)

        worker_tasks = [a for a in assignments if a.crew_member_id == worker.id]
        last_task = worker_tasks[-1] if worker_tasks else None
        is_switch = last_task is not None and last_task.task_type == "mow"
        if is_switch:
            mins += TASK_SWITCH_PENALTY_MINUTES

        assignments.append(TaskAssignment(
            crew_member_id=worker.id,
            crew_member_name=worker.name,
            task_order=task_counter,
            zone_id=zone.id,
            zone_label=zone.label,
            task_type="trim",
            estimated_minutes=mins,
            role_used="trimmer",
            is_role_switch=is_switch
        ))
        crew_load[worker.id] = crew_load.get(worker.id, 0) + mins
        task_counter += 1

    # PHASE 4: Blow — assign to whoever finishes first
    # In real-time this is dynamic, but here we schedule it for the
    # least loaded crew member(s) to start as mow+trim sections complete
    blow_mins = estimate_blow_minutes(zones, mode) + TASK_SWITCH_PENALTY_MINUTES
    total_acres = sum(z.area_sqft for z in workable_zones) / 43560

    if mode == "fastest" and len(crew) >= 2:
        sorted_crew = sorted(all_crew, key=lambda c: crew_load.get(c.id, 0))
        num_blowers = min(len(crew), 3)
        blow_per_person = max(int(blow_mins / num_blowers), 5)
        for i in range(num_blowers):
            worker = sorted_crew[i]
            label = f"Final Blowout & Cleanup (Section {i+1})" if num_blowers > 1 else "Final Blowout & Cleanup"
            assignments.append(TaskAssignment(
                crew_member_id=worker.id,
                crew_member_name=worker.name,
                task_order=task_counter,
                zone_id="cleanup",
                zone_label=label,
                task_type="blow",
                estimated_minutes=blow_per_person,
                role_used="blower",
                is_role_switch=True
            ))
            crew_load[worker.id] = crew_load.get(worker.id, 0) + blow_per_person
            task_counter += 1
    elif total_acres > 5 and len(crew) >= 2:
        sorted_crew = sorted(all_crew, key=lambda c: crew_load.get(c.id, 0))
        blow_per_person = max(int(blow_mins / 2), 5)
        for i in range(min(2, len(sorted_crew))):
            worker = sorted_crew[i]
            assignments.append(TaskAssignment(
                crew_member_id=worker.id,
                crew_member_name=worker.name,
                task_order=task_counter,
                zone_id="cleanup",
                zone_label=f"Final Blowout & Cleanup {'(Zone A)' if i == 0 else '(Zone B)'}",
                task_type="blow",
                estimated_minutes=blow_per_person,
                role_used="blower",
                is_role_switch=True
            ))
            crew_load[worker.id] = crew_load.get(worker.id, 0) + blow_per_person
            task_counter += 1
    else:
        worker = min(all_crew, key=lambda c: crew_load.get(c.id, 0))
        assignments.append(TaskAssignment(
            crew_member_id=worker.id,
            crew_member_name=worker.name,
            task_order=task_counter,
            zone_id="cleanup",
            zone_label="Final Blowout & Cleanup",
            task_type="blow",
            estimated_minutes=blow_mins,
            role_used="blower",
            is_role_switch=True
        ))

    return assignments

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

    assignments = assign_zones_to_crew(request.zones, request.crew, request.mode)

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

