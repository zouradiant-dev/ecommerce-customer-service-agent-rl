"""Stage-separated CUDA telemetry and fail-closed memory qualification.

No torch import on CPU. Stage peak reserved memory plus the same stage's
observed non-allocator peak forms a conservative capacity bound. Fragmentation
is already inside reserved memory and is diagnostic only, never added again.
Observed driver free memory must also meet the unchanged 1 GiB margin.
These sampled checks are engineering evidence, not a guarantee against OOM.
"""
import json
from pathlib import Path
import time

MEMORY_POLICY = "stage_reserved_upper_bound_and_driver_free_v2"
GIB = 2**30
MIN_HEADROOM = GIB
STAGES = ("initialization", "training", "saving", "reload")
FIELDS = ("total_bytes", "driver_free_bytes", "allocated_bytes", "reserved_bytes",
          "peak_allocated_bytes", "peak_reserved_bytes", "non_allocator_used_bytes", "peak_inactive_split_bytes")


def validate_sample(row):
    if row.get("policy") != MEMORY_POLICY or row.get("stage") not in STAGES:
        raise ValueError("unknown_memory_policy_or_stage")
    if any(type(row.get(name)) is not int or row[name] < 0 for name in FIELDS):
        raise ValueError("invalid_memory_measurement")
    total, allocated, reserved = row["total_bytes"], row["allocated_bytes"], row["reserved_bytes"]
    if (total <= 0 or row["driver_free_bytes"] > total or allocated > reserved or reserved > total
            or not allocated <= row["peak_allocated_bytes"] <= row["peak_reserved_bytes"] <= total
            or reserved > row["peak_reserved_bytes"]
            or row["peak_inactive_split_bytes"] > row["peak_reserved_bytes"]
            or row["non_allocator_used_bytes"] != max(0, total - row["driver_free_bytes"] - reserved)):
        raise ValueError("inconsistent_memory_measurement")


def summarize_memory(rows, expected_steps):
    if type(expected_steps) is not int or expected_steps not in (10, 100):
        raise ValueError("unexpected_memory_qualification_steps")
    if not rows:
        raise ValueError("missing_memory_measurements")
    for row in rows:
        validate_sample(row)
    if len({row["total_bytes"] for row in rows}) != 1:
        raise ValueError("memory_device_total_changed")
    expected_events = [("initialization", "begin"), ("initialization", "end"), ("training", "begin")]
    for step in range(1, expected_steps + 1):
        expected_events.extend((('training', 'reward'), ('training', 'step_end')))
    expected_events.extend((('training', 'end'), ('saving', 'begin'), ('saving', 'end'),
                            ('reload', 'begin'), ('reload', 'end')))
    if [(r["stage"], r.get("point")) for r in rows] != expected_events:
        raise ValueError("memory_stage_or_step_events_incomplete")
    for point in ("reward", "step_end"):
        if [r.get("step") for r in rows if r.get("point") == point] != list(range(1, expected_steps + 1)):
            raise ValueError("memory_optimizer_step_sequence_incomplete")
    if any(r.get("cache_released") is not (r["point"] in ("begin", "end")) for r in rows):
        raise ValueError("memory_boundary_cache_release_not_verified")
    total = rows[0]["total_bytes"]
    stages = {}
    for stage in STAGES:
        samples = [r for r in rows if r["stage"] == stage]
        peak = max(r["peak_allocated_bytes"] for r in samples)
        reserved = max(r["peak_reserved_bytes"] for r in samples)
        external = max(r["non_allocator_used_bytes"] for r in samples)
        fragmented = max(r["peak_inactive_split_bytes"] for r in samples)
        boundary_free = min(r["driver_free_bytes"] for r in samples if r["point"] in ("begin", "end"))
        observed_free = min(r["driver_free_bytes"] for r in samples)
        # Independent maxima give a conservative budget, not simultaneous live
        # demand. Reserved already includes tensors, inactive splits and cache.
        stages[stage] = {"peak_allocated_bytes": peak, "peak_reserved_bytes": reserved,
                         "peak_observed_non_allocator_bytes": external,
                         "peak_inactive_split_bytes": fragmented,
                         "reserved_upper_bound_headroom_bytes": total - reserved - external,
                         "min_observed_driver_free_bytes": observed_free,
                         "boundary_driver_free_bytes": boundary_free}
    headroom = min(min(s["reserved_upper_bound_headroom_bytes"], s["min_observed_driver_free_bytes"]) for s in stages.values())
    return {"policy": MEMORY_POLICY, "minimum_headroom_bytes": MIN_HEADROOM, "total_bytes": total,
            "stages": stages, "memory_headroom_gib": headroom / GIB,
            "memory_safety_passed": headroom >= MIN_HEADROOM,
            "peak_allocated_gib": max(s["peak_allocated_bytes"] for s in stages.values()) / GIB,
            "peak_reserved_gib": max(s["peak_reserved_bytes"] for s in stages.values()) / GIB,
            "fragmentation_diagnostic_only_already_in_reserved": True,
            "capacity_bound_is_conservative_not_simultaneous_peak": True,
            "limitations": "Stage reserved peak plus stage observed external peak is a conservative bound, not a simultaneous measurement. External usage is sampled; longer or different trajectories can still OOM."}


class CudaMemoryAudit:
    def __init__(self, cuda, path):
        self.cuda, self.path = cuda, Path(path)
        self.rows, self.stage = [], None
        self.started = time.monotonic()
        self.path.open("x", encoding="utf-8").close()
        # The deployment pins native PyTorch CUDA allocation. Do not silently
        # apply native allocator interpretations to another backend.
        if cuda.memory.get_allocator_backend() != "native":
            raise RuntimeError("memory_audit_requires_native_cuda_allocator")

    def sample(self, point, *, step=None, release_cache=False):
        self.cuda.synchronize()
        if release_cache:
            self.cuda.empty_cache()
            self.cuda.synchronize()
        free, total = self.cuda.mem_get_info()
        reserved = self.cuda.memory_reserved()
        stats = self.cuda.memory_stats()
        if "inactive_split_bytes.all.peak" not in stats:
            raise RuntimeError("native_allocator_fragmentation_measurement_missing")
        row = {"event": "V13_MEMORY_SAMPLE", "policy": MEMORY_POLICY, "stage": self.stage,
               "point": point, "step": step, "elapsed_seconds": time.monotonic() - self.started,
               "cache_released": release_cache, "total_bytes": int(total), "driver_free_bytes": int(free),
               "allocated_bytes": int(self.cuda.memory_allocated()), "reserved_bytes": int(reserved),
               "peak_allocated_bytes": int(self.cuda.max_memory_allocated()),
               "peak_reserved_bytes": int(self.cuda.max_memory_reserved()),
               "peak_inactive_split_bytes": int(stats["inactive_split_bytes.all.peak"]),
               "non_allocator_used_bytes": max(0, int(total) - int(free) - int(reserved))}
        validate_sample(row)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        self.rows.append(row)
        print(json.dumps(row, allow_nan=False), flush=True)
        return row

    def begin(self, stage):
        if stage not in STAGES:
            raise ValueError("unknown_memory_stage")
        self.cuda.synchronize()
        self.cuda.empty_cache()
        self.cuda.reset_peak_memory_stats()
        self.stage = stage
        return self.sample("begin", release_cache=True)

    def end(self):
        return self.sample("end", release_cache=True)
