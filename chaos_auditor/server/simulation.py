"""
Service graph simulation engine for the Chaos Auditor environment.

Realistic distributed system simulation featuring:
- Service dependencies with cascading failure propagation
- Connection pool drain over time (not instant)
- Memory leak simulation under sustained stress
- Request queue backpressure between services
- Gradual cascade timing (failures take 2-3 ticks to propagate)
- Self-healing with realistic recovery curves
- Circuit breakers with half-open probing state
- Monitoring with configurable blind spots (the core mechanic)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class ServiceState(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"
    DOWN = "DOWN"


class CircuitBreakerState(str, Enum):
    CLOSED = "CLOSED"        # Normal — traffic flows
    OPEN = "OPEN"            # Tripped — traffic blocked
    HALF_OPEN = "HALF_OPEN"  # Probing — limited test traffic


@dataclass
class Service:
    name: str
    health: float = 100.0
    status: ServiceState = ServiceState.HEALTHY
    cpu_usage: float = 10.0
    memory_usage: float = 20.0
    error_rate: float = 0.0
    connection_count: int = 10
    max_connections: int = 200
    data_integrity: float = 1.0
    response_time_ms: int = 50
    disk_usage: float = 20.0
    request_queue_depth: int = 0
    max_queue_depth: int = 500

    dependencies: List[str] = field(default_factory=list)

    # Self-healing capabilities
    has_health_check: bool = True
    has_auto_restart: bool = True
    has_circuit_breaker: bool = False
    has_auto_scaling: bool = False
    has_anomaly_detection: bool = False

    # Monitoring coverage — ONLY these metrics trigger alerts
    monitored_metrics: List[str] = field(
        default_factory=lambda: ["cpu_usage", "error_rate", "status"]
    )

    # Internal timers and state
    restart_timer: int = 0
    restart_delay: int = 3
    cb_state: CircuitBreakerState = CircuitBreakerState.CLOSED
    cb_target: Optional[str] = None
    cb_cooldown: int = 0
    instances: int = 1
    base_cpu: float = 10.0
    base_memory: float = 20.0
    base_response_time: int = 50
    memory_leak_rate: float = 0.0  # MB per tick leaked under stress
    ticks_under_stress: int = 0    # How long this service has been stressed

    # Damage tracking
    damage_sources: List[str] = field(default_factory=list)
    is_killed: bool = False

    def get_monitoring_view(self) -> Dict[str, Any]:
        """Return ONLY what monitoring dashboards would show."""
        view: Dict[str, Any] = {"name": self.name, "status": self.status.value}
        metric_map = {
            "cpu_usage": round(self.cpu_usage, 1),
            "memory_usage": round(self.memory_usage, 1),
            "error_rate": round(self.error_rate, 1),
            "response_time_ms": self.response_time_ms,
            "connection_count": self.connection_count,
            "disk_usage": round(self.disk_usage, 1),
            "data_integrity": round(self.data_integrity, 3),
            "request_queue_depth": self.request_queue_depth,
        }
        for metric in self.monitored_metrics:
            if metric in metric_map:
                view[metric] = metric_map[metric]
        return view

    def get_deep_view(self) -> Dict[str, Any]:
        """Return ALL metrics — only via deep_inspect."""
        conn_pct = round(self.connection_count / max(self.max_connections, 1) * 100, 1)
        queue_pct = round(self.request_queue_depth / max(self.max_queue_depth, 1) * 100, 1)

        indicators = {}
        indicators["cpu"] = "RED" if self.cpu_usage > 85 else "YELLOW" if self.cpu_usage > 60 else "GREEN"
        indicators["memory"] = "RED" if self.memory_usage > 90 else "YELLOW" if self.memory_usage > 70 else "GREEN"
        indicators["errors"] = "RED" if self.error_rate > 20 else "YELLOW" if self.error_rate > 5 else "GREEN"
        indicators["connections"] = "RED" if conn_pct > 90 else "YELLOW" if conn_pct > 70 else "GREEN"
        indicators["data_integrity"] = "RED" if self.data_integrity < 0.7 else "YELLOW" if self.data_integrity < 0.95 else "GREEN"
        indicators["disk"] = "RED" if self.disk_usage > 90 else "YELLOW" if self.disk_usage > 75 else "GREEN"
        indicators["latency"] = "RED" if self.response_time_ms > 1000 else "YELLOW" if self.response_time_ms > 300 else "GREEN"
        indicators["queue"] = "RED" if queue_pct > 80 else "YELLOW" if queue_pct > 50 else "GREEN"

        return {
            "name": self.name,
            "status": self.status.value,
            "health": round(self.health, 1),
            "cpu_usage": f"{self.cpu_usage:.1f}% [{indicators['cpu']}]",
            "memory_usage": f"{self.memory_usage:.1f}% [{indicators['memory']}]",
            "error_rate": f"{self.error_rate:.1f}% [{indicators['errors']}]",
            "connections": f"{self.connection_count}/{self.max_connections} ({conn_pct}%) [{indicators['connections']}]",
            "data_integrity": f"{self.data_integrity:.3f} [{indicators['data_integrity']}]",
            "response_time": f"{self.response_time_ms}ms [{indicators['latency']}]",
            "disk_usage": f"{self.disk_usage:.1f}% [{indicators['disk']}]",
            "request_queue": f"{self.request_queue_depth}/{self.max_queue_depth} ({queue_pct}%) [{indicators['queue']}]",
            "instances": self.instances,
            "dependencies": self.dependencies,
            "circuit_breaker": self.cb_state.value if self.has_circuit_breaker else "N/A",
            "ticks_under_stress": self.ticks_under_stress,
        }

    def update_status(self) -> None:
        if self.health <= 0 or self.is_killed:
            self.status = ServiceState.DOWN
            self.health = max(0, self.health)
        elif self.health <= 30 or self.error_rate > 50:
            self.status = ServiceState.CRITICAL
        elif self.health <= 60 or self.error_rate > 15 or self.cpu_usage > 90:
            self.status = ServiceState.DEGRADED
        else:
            self.status = ServiceState.HEALTHY

    @property
    def is_stressed(self) -> bool:
        return (
            self.cpu_usage > 70
            or self.memory_usage > 75
            or self.error_rate > 10
            or self.connection_count > self.max_connections * 0.7
        )


@dataclass
class AlertRecord:
    service_name: str
    metric: str
    value: float
    threshold: float
    message: str
    severity: str = "warning"


ALERT_THRESHOLDS = {
    "cpu_usage": 85.0,
    "memory_usage": 90.0,
    "error_rate": 20.0,
    "response_time_ms": 1000.0,
    "disk_usage": 90.0,
    "connection_count": 0.9,  # fraction of max
    "request_queue_depth": 0.8,  # fraction of max
}


class ServiceGraph:
    """Realistic distributed system simulation."""

    def __init__(self) -> None:
        self.services: Dict[str, Service] = {}
        self.alerts: List[AlertRecord] = []
        self.alert_history: List[AlertRecord] = []  # Cumulative across ticks
        self.tick_count: int = 0
        self.network_partitions: Set[Tuple[str, str]] = set()
        self.total_alerts_fired: int = 0

    def add_service(self, service: Service) -> None:
        service.base_cpu = service.cpu_usage
        service.base_memory = service.memory_usage
        service.base_response_time = service.response_time_ms
        self.services[service.name] = service

    def get_service(self, name: str) -> Optional[Service]:
        return self.services.get(name)

    def is_partitioned(self, a: str, b: str) -> bool:
        return (a, b) in self.network_partitions or (b, a) in self.network_partitions

    # ── Chaos Actions ──────────────────────────────────────────────

    def kill(self, target: str) -> str:
        svc = self.get_service(target)
        if not svc:
            return f"[ERROR] Service '{target}' not found in system topology."
        svc.health = 0
        svc.is_killed = True
        svc.error_rate = 100.0
        svc.connection_count = 0
        svc.request_queue_depth = 0
        svc.restart_timer = 0
        svc.damage_sources.append("kill")
        svc.update_status()

        # Check which services depend on this one
        dependents = [s.name for s in self.services.values() if target in s.dependencies]
        dep_msg = f" Dependent services: {', '.join(dependents)}." if dependents else ""

        return (
            f"Process killed on {target}. Service is DOWN. All active connections dropped.\n"
            f"  Health: 0/100 | Error rate: 100% | Connections: 0/{svc.max_connections}\n"
            f"  Recovery: {'auto-restart in ~{} ticks'.format(svc.restart_delay) if svc.has_auto_restart else 'NO auto-restart — manual intervention required.'}\n"
            f"  Impact:{dep_msg}"
        )

    def spike_traffic(self, target: str, multiplier: float) -> str:
        svc = self.get_service(target)
        if not svc:
            return f"[ERROR] Service '{target}' not found in system topology."
        multiplier = max(1.5, min(multiplier, 10.0))

        old_cpu = svc.cpu_usage
        old_conns = svc.connection_count
        old_rt = svc.response_time_ms

        svc.cpu_usage = min(100.0, svc.cpu_usage * multiplier)
        svc.connection_count = min(svc.max_connections, int(svc.connection_count * multiplier))
        svc.response_time_ms = int(svc.response_time_ms * (1 + (multiplier - 1) * 0.6))
        svc.memory_usage = min(100.0, svc.memory_usage * (1 + (multiplier - 1) * 0.25))
        svc.request_queue_depth = min(svc.max_queue_depth, int(svc.request_queue_depth + multiplier * 30))

        if svc.cpu_usage > 90:
            svc.health = max(0, svc.health - 15)
            svc.error_rate = min(100, svc.error_rate + 12)

        svc.damage_sources.append(f"spike_{multiplier:.1f}x")
        svc.update_status()

        conn_pct = svc.connection_count / max(svc.max_connections, 1) * 100

        return (
            f"Traffic spike {multiplier:.1f}x applied to {target}.\n"
            f"  CPU: {old_cpu:.0f}% -> {svc.cpu_usage:.0f}%\n"
            f"  Connections: {old_conns} -> {svc.connection_count}/{svc.max_connections} ({conn_pct:.0f}%)\n"
            f"  Response time: {old_rt}ms -> {svc.response_time_ms}ms\n"
            f"  Queue depth: {svc.request_queue_depth}/{svc.max_queue_depth}\n"
            f"  {'Auto-scaling will trigger next tick.' if svc.has_auto_scaling and svc.cpu_usage > 80 else ''}"
        )

    def corrupt_data(self, target: str, data_type: str = "cache") -> str:
        svc = self.get_service(target)
        if not svc:
            return f"[ERROR] Service '{target}' not found in system topology."

        old_integrity = svc.data_integrity
        svc.data_integrity = max(0.0, svc.data_integrity - 0.4)
        svc.damage_sources.append(f"corrupt_{data_type}")

        is_monitored = "data_integrity" in svc.monitored_metrics
        # Data corruption does NOT affect health/cpu/error_rate — that's the key
        dependents = [s.name for s in self.services.values() if target in s.dependencies]

        return (
            f"Data corruption injected into {target} ({data_type} layer).\n"
            f"  Data integrity: {old_integrity:.2f} -> {svc.data_integrity:.2f}\n"
            f"  Service health: {svc.health:.0f}/100 (unchanged — corruption doesn't crash services)\n"
            f"  Service status: {svc.status.value} (unchanged)\n"
            f"  Monitoring: {'ALERT — data_integrity is monitored on this service' if is_monitored else 'NO ALERT — data_integrity is NOT in this services monitored metrics'}\n"
            f"  Propagation risk: {', '.join(dependents) + ' consume data from this service' if dependents else 'No downstream consumers'}"
        )

    def add_latency(self, target: str, latency_ms: int) -> str:
        svc = self.get_service(target)
        if not svc:
            return f"[ERROR] Service '{target}' not found in system topology."
        latency_ms = max(50, min(latency_ms, 2000))

        old_rt = svc.response_time_ms
        svc.response_time_ms += latency_ms

        if svc.response_time_ms > 500:
            svc.health = max(0, svc.health - 8)
            svc.error_rate = min(100, svc.error_rate + 3)

        svc.damage_sources.append(f"latency_{latency_ms}ms")
        svc.update_status()

        threshold = ALERT_THRESHOLDS.get("response_time_ms", 1000)
        is_monitored = "response_time_ms" in svc.monitored_metrics
        below_threshold = svc.response_time_ms < threshold

        return (
            f"Network latency injected on {target}: +{latency_ms}ms.\n"
            f"  Response time: {old_rt}ms -> {svc.response_time_ms}ms\n"
            f"  Alert threshold: {threshold}ms | Current: {'BELOW' if below_threshold else 'ABOVE'} threshold\n"
            f"  Monitoring: {'response_time is monitored' if is_monitored else 'response_time is NOT monitored on this service'}\n"
            f"  {'Silent — no alert will fire.' if (not is_monitored or below_threshold) else 'Alert will fire on next monitoring check.'}\n"
            f"  Upstream impact: services depending on {target} will see increased latency on their calls"
        )

    def partition_network(self, service_a: str, service_b: str) -> str:
        if not self.get_service(service_a):
            return f"[ERROR] Service '{service_a}' not found in system topology."
        if not self.get_service(service_b):
            return f"[ERROR] Service '{service_b}' not found in system topology."
        self.network_partitions.add((service_a, service_b))
        self.services[service_a].damage_sources.append(f"partition_from_{service_b}")
        self.services[service_b].damage_sources.append(f"partition_from_{service_a}")

        a_deps_b = service_b in self.services[service_a].dependencies
        b_deps_a = service_a in self.services[service_b].dependencies
        impact = []
        if a_deps_b:
            impact.append(f"{service_a} depends on {service_b} — calls will fail")
        if b_deps_a:
            impact.append(f"{service_b} depends on {service_a} — calls will fail")
        if not impact:
            impact.append("No direct dependency — impact may be indirect")

        cb_a = self.services[service_a].has_circuit_breaker
        cb_b = self.services[service_b].has_circuit_breaker

        return (
            f"Network partition created: {service_a} <-X-> {service_b}\n"
            f"  All TCP connections between these services will fail.\n"
            f"  Impact: {'; '.join(impact)}\n"
            f"  Circuit breakers: {service_a}={'yes' if cb_a else 'no'}, {service_b}={'yes' if cb_b else 'no'}"
        )

    def fill_disk(self, target: str, percentage: float = 95.0) -> str:
        svc = self.get_service(target)
        if not svc:
            return f"[ERROR] Service '{target}' not found in system topology."
        percentage = max(50.0, min(percentage, 99.0))

        old_disk = svc.disk_usage
        svc.disk_usage = percentage
        if percentage > 95:
            svc.health = max(0, svc.health - 15)
            svc.error_rate = min(100, svc.error_rate + 8)
        svc.damage_sources.append(f"fill_disk_{percentage:.0f}%")
        svc.update_status()

        is_monitored = "disk_usage" in svc.monitored_metrics
        threshold = ALERT_THRESHOLDS.get("disk_usage", 90)

        return (
            f"Disk filled on {target}.\n"
            f"  Disk usage: {old_disk:.0f}% -> {percentage:.0f}%\n"
            f"  {'Write operations will start failing.' if percentage > 92 else 'Write performance degraded.' if percentage > 80 else 'Approaching capacity.'}\n"
            f"  {'Logs cannot be written — debugging will be impaired.' if percentage > 95 else ''}\n"
            f"  Monitoring: {'disk_usage is monitored (threshold: {}%)'.format(threshold) if is_monitored else 'disk_usage is NOT monitored — silent degradation'}"
        )

    def exhaust_connections(self, target: str) -> str:
        svc = self.get_service(target)
        if not svc:
            return f"[ERROR] Service '{target}' not found in system topology."

        old_conns = svc.connection_count
        svc.connection_count = int(svc.max_connections * 0.95)
        svc.response_time_ms = max(svc.response_time_ms, 700)
        svc.request_queue_depth = min(svc.max_queue_depth, svc.request_queue_depth + 100)
        svc.damage_sources.append("exhaust_connections")

        is_monitored = "connection_count" in svc.monitored_metrics
        dependents = [s.name for s in self.services.values() if target in s.dependencies]

        return (
            f"Connection pool flooded on {target}.\n"
            f"  Connections: {old_conns} -> {svc.connection_count}/{svc.max_connections} (95% utilized)\n"
            f"  New incoming connections will be queued or rejected.\n"
            f"  Response time increased to {svc.response_time_ms}ms (connections waiting for pool slots)\n"
            f"  Monitoring: {'connection_count IS monitored' if is_monitored else 'connection_count is NOT monitored — new requests will silently time out'}\n"
            f"  Upstream services affected: {', '.join(dependents) if dependents else 'none'}"
        )

    # ── Tick: System Evolution ─────────────────────────────────────

    def tick(self) -> List[str]:
        """Advance simulation by one tick. Returns events that occurred."""
        self.tick_count += 1
        events: List[str] = []
        self.alerts.clear()

        # Phase 1: Stress tracking and memory leaks
        for svc in self.services.values():
            if svc.status == ServiceState.DOWN:
                continue
            if svc.is_stressed:
                svc.ticks_under_stress += 1
                # Memory leak: sustained stress causes gradual memory growth
                if svc.ticks_under_stress > 2:
                    leak = 1.5 * (svc.ticks_under_stress - 2)
                    svc.memory_usage = min(100.0, svc.memory_usage + leak)
                    if svc.memory_usage > 95:
                        svc.health = max(0, svc.health - 10)
                        events.append(
                            f"{svc.name}: Memory pressure critical ({svc.memory_usage:.0f}%). "
                            f"OOM risk increasing."
                        )
            else:
                svc.ticks_under_stress = max(0, svc.ticks_under_stress - 1)
                # Gradual memory recovery when not stressed
                if svc.memory_usage > svc.base_memory + 5:
                    svc.memory_usage = max(svc.base_memory, svc.memory_usage - 2.0)

        # Phase 2: Dependency cascade propagation (gradual, not instant)
        for svc in self.services.values():
            if svc.status == ServiceState.DOWN:
                continue

            for dep_name in svc.dependencies:
                # Network partition check
                if self.is_partitioned(svc.name, dep_name):
                    if svc.has_circuit_breaker:
                        if svc.cb_state == CircuitBreakerState.CLOSED:
                            svc.cb_state = CircuitBreakerState.OPEN
                            svc.cb_target = dep_name
                            svc.cb_cooldown = 5
                            events.append(
                                f"{svc.name}: Circuit breaker OPEN for {dep_name} (network partition detected)."
                            )
                        svc.error_rate = min(100, svc.error_rate + 3)
                    else:
                        svc.error_rate = min(100, svc.error_rate + 8)
                        svc.health = max(0, svc.health - 4)
                    # Requests back up in queue
                    svc.request_queue_depth = min(svc.max_queue_depth, svc.request_queue_depth + 20)
                    continue

                dep = self.get_service(dep_name)
                if not dep:
                    continue

                # Skip if circuit breaker is open for this dependency
                if svc.cb_state == CircuitBreakerState.OPEN and svc.cb_target == dep_name:
                    continue

                # Dependency DOWN → gradual cascade (not instant)
                if dep.status == ServiceState.DOWN:
                    if svc.has_circuit_breaker and svc.cb_state == CircuitBreakerState.CLOSED:
                        svc.cb_state = CircuitBreakerState.OPEN
                        svc.cb_target = dep_name
                        svc.cb_cooldown = 5
                        events.append(
                            f"{svc.name}: Circuit breaker OPEN — {dep_name} is DOWN."
                        )
                    elif not svc.has_circuit_breaker:
                        svc.error_rate = min(100, svc.error_rate + 8)
                        svc.health = max(0, svc.health - 4)
                        svc.request_queue_depth = min(svc.max_queue_depth, svc.request_queue_depth + 25)

                elif dep.status == ServiceState.CRITICAL:
                    svc.error_rate = min(100, svc.error_rate + 3)
                    svc.health = max(0, svc.health - 2)
                    svc.response_time_ms += 30

                elif dep.status == ServiceState.DEGRADED:
                    svc.response_time_ms += 15
                    svc.health = max(0, svc.health - 1)

                # Data corruption propagates SILENTLY through dependencies
                if dep.data_integrity < 0.9:
                    propagated = min(svc.data_integrity, dep.data_integrity + 0.05)
                    if propagated < svc.data_integrity:
                        svc.data_integrity = propagated

                # Connection pool pressure propagates upstream
                dep_conn_ratio = dep.connection_count / max(dep.max_connections, 1)
                if dep_conn_ratio > 0.8:
                    held = int(5 * (dep_conn_ratio - 0.8) * 10)
                    svc.connection_count = min(svc.max_connections, svc.connection_count + held)
                    svc.response_time_ms += int(held * 8)

                # Request queue backpressure
                if dep.request_queue_depth > dep.max_queue_depth * 0.5:
                    overflow = int((dep.request_queue_depth / dep.max_queue_depth - 0.5) * 30)
                    svc.request_queue_depth = min(svc.max_queue_depth, svc.request_queue_depth + overflow)

        # Phase 3: Self-healing
        for svc in self.services.values():
            # Auto-restart for killed services
            if svc.is_killed and svc.has_auto_restart:
                svc.restart_timer += 1
                if svc.restart_timer >= svc.restart_delay:
                    svc.is_killed = False
                    svc.health = 40.0  # Comes back weak
                    svc.error_rate = max(0, svc.error_rate - 60)
                    svc.cpu_usage = svc.base_cpu * 1.5  # Startup CPU spike
                    svc.connection_count = 5
                    svc.restart_timer = 0
                    svc.update_status()
                    events.append(
                        f"{svc.name}: Auto-restart complete. Status={svc.status.value} "
                        f"(health={svc.health:.0f}/100 — warming up)."
                    )

            # Gradual recovery for non-killed degraded services
            if not svc.is_killed and svc.status in (ServiceState.DEGRADED, ServiceState.CRITICAL):
                if not svc.is_stressed:
                    svc.health = min(100, svc.health + 3)
                    svc.error_rate = max(0, svc.error_rate - 2)

            # Recovering services ramp up gradually
            if not svc.is_killed and svc.health > 0 and svc.health < 80:
                svc.health = min(100, svc.health + 1)

            # Circuit breaker state machine
            if svc.has_circuit_breaker and svc.cb_state != CircuitBreakerState.CLOSED:
                if svc.cb_state == CircuitBreakerState.OPEN:
                    svc.cb_cooldown -= 1
                    if svc.cb_cooldown <= 0:
                        svc.cb_state = CircuitBreakerState.HALF_OPEN
                        events.append(
                            f"{svc.name}: Circuit breaker HALF-OPEN — sending probe traffic to {svc.cb_target}."
                        )

                elif svc.cb_state == CircuitBreakerState.HALF_OPEN:
                    target = self.get_service(svc.cb_target) if svc.cb_target else None
                    if target and target.status in (ServiceState.HEALTHY, ServiceState.DEGRADED):
                        if not self.is_partitioned(svc.name, svc.cb_target):
                            svc.cb_state = CircuitBreakerState.CLOSED
                            svc.cb_target = None
                            events.append(f"{svc.name}: Circuit breaker CLOSED — traffic restored.")
                    else:
                        svc.cb_state = CircuitBreakerState.OPEN
                        svc.cb_cooldown = 3

            # Auto-scaling
            if svc.has_auto_scaling and svc.cpu_usage > 80 and svc.instances < 5 and not svc.is_killed:
                svc.instances += 1
                svc.cpu_usage = max(20, svc.cpu_usage * 0.6)
                svc.connection_count = max(10, int(svc.connection_count * 0.7))
                svc.request_queue_depth = max(0, svc.request_queue_depth - 50)
                events.append(
                    f"{svc.name}: Scaled to {svc.instances} instances. CPU={svc.cpu_usage:.0f}%."
                )

            # Natural drain: connections and queues slowly drain
            if not svc.is_killed:
                if svc.connection_count > 15 and "exhaust_connections" not in svc.damage_sources:
                    svc.connection_count = max(10, svc.connection_count - 4)
                if svc.request_queue_depth > 0:
                    drain = max(5, svc.request_queue_depth // 8)
                    svc.request_queue_depth = max(0, svc.request_queue_depth - drain)
                # Response time recovery toward baseline
                if svc.response_time_ms > svc.base_response_time and not svc.is_stressed:
                    svc.response_time_ms = max(
                        svc.base_response_time,
                        svc.response_time_ms - 20,
                    )

            svc.update_status()

        # Phase 4: Monitoring alerts
        for svc in self.services.values():
            self._check_alerts(svc)

        # Phase 5: Anomaly detection (sophisticated services only)
        for svc in self.services.values():
            if svc.has_anomaly_detection and not svc.is_killed:
                if svc.cpu_usage > svc.base_cpu * 2.5:
                    self._fire_alert(svc.name, "anomaly_cpu", svc.cpu_usage, svc.base_cpu * 2.5,
                                     f"Anomaly: CPU spike on {svc.name} ({svc.cpu_usage:.0f}%)", "critical")
                    events.append(f"ANOMALY DETECTION: Unusual CPU pattern on {svc.name}.")
                if svc.error_rate > 15 and svc.error_rate > 0:
                    self._fire_alert(svc.name, "anomaly_errors", svc.error_rate, 15,
                                     f"Anomaly: Error spike on {svc.name} ({svc.error_rate:.0f}%)", "warning")
                    events.append(f"ANOMALY DETECTION: Unusual error pattern on {svc.name}.")

        # Phase 6: Compound effects — emergent failures from action combinations
        events.extend(self._check_compound_effects())

        return events

    def _check_compound_effects(self) -> List[str]:
        """Detect emergent failures that only occur when multiple conditions combine.

        These are the hardest vulnerabilities to find because no single action
        triggers them. The agent must discover that combining sub-threshold
        attacks creates catastrophic silent failures.
        """
        events: List[str] = []

        # Compound 1: latency + traffic spike on dependent services
        # When service A has added latency AND service B (which depends on A)
        # has spiked traffic, the combination causes connection pool exhaustion
        # on service A that neither action alone would cause.
        for svc in self.services.values():
            if svc.status == ServiceState.DOWN:
                continue
            has_latency = any("latency" in d for d in svc.damage_sources)
            # Check if any service that depends on this one has been traffic-spiked
            upstream_spiked = False
            for other in self.services.values():
                if svc.name in other.dependencies:
                    if any("spike" in d for d in other.damage_sources):
                        upstream_spiked = True
                        break

            if has_latency and upstream_spiked:
                # Compound effect: latency + upstream traffic = connection exhaustion
                compound_conns = int(svc.max_connections * 0.3)
                svc.connection_count = min(svc.max_connections, svc.connection_count + compound_conns)
                svc.response_time_ms += 200
                svc.request_queue_depth = min(svc.max_queue_depth, svc.request_queue_depth + 80)
                if not any("compound_latency_spike" in d for d in svc.damage_sources):
                    svc.damage_sources.append("compound_latency_spike")
                    conn_pct = svc.connection_count / max(svc.max_connections, 1) * 100
                    if "connection_count" not in svc.monitored_metrics:
                        events.append(
                            f"COMPOUND EFFECT on {svc.name}: Latency + upstream traffic spike "
                            f"causing connection pool pressure ({conn_pct:.0f}% utilized). "
                            f"NOT DETECTED by monitoring."
                        )
                    else:
                        events.append(
                            f"COMPOUND EFFECT on {svc.name}: Latency + upstream traffic spike "
                            f"causing connection pool pressure ({conn_pct:.0f}% utilized)."
                        )

        # Compound 2: partition + traffic spike = memory buildup from queued requests
        for svc in self.services.values():
            if svc.status == ServiceState.DOWN:
                continue
            has_partition = any("partition" in d for d in svc.damage_sources)
            has_spike = any("spike" in d for d in svc.damage_sources)

            if has_partition and has_spike:
                # Requests can't reach dependency, pile up in memory
                svc.memory_usage = min(100.0, svc.memory_usage + 15)
                svc.request_queue_depth = min(svc.max_queue_depth,
                                              svc.request_queue_depth + 60)
                if not any("compound_partition_spike" in d for d in svc.damage_sources):
                    svc.damage_sources.append("compound_partition_spike")
                    if "memory_usage" not in svc.monitored_metrics:
                        events.append(
                            f"COMPOUND EFFECT on {svc.name}: Network partition + traffic spike "
                            f"causing memory buildup ({svc.memory_usage:.0f}%) from queued requests. "
                            f"Memory NOT MONITORED — silent OOM risk."
                        )

        return events

    def _check_alerts(self, svc: Service) -> None:
        for metric in svc.monitored_metrics:
            threshold = ALERT_THRESHOLDS.get(metric)
            if threshold is None:
                if metric == "status" and svc.status in (ServiceState.DOWN, ServiceState.CRITICAL):
                    self._fire_alert(svc.name, "status", 0, 0,
                                     f"[CRITICAL] {svc.name} is {svc.status.value}", "critical")
                continue

            if metric == "connection_count":
                value = svc.connection_count / max(svc.max_connections, 1)
            elif metric == "request_queue_depth":
                value = svc.request_queue_depth / max(svc.max_queue_depth, 1)
            else:
                value = getattr(svc, metric, 0)

            if value > threshold:
                severity = "critical" if value > threshold * 1.2 else "warning"
                self._fire_alert(svc.name, metric, round(value, 1), threshold,
                                 f"[{severity.upper()}] {svc.name}: {metric}={value:.1f} (threshold: {threshold})",
                                 severity)

    def _fire_alert(self, service: str, metric: str, value: float, threshold: float, message: str, severity: str) -> None:
        alert = AlertRecord(service_name=service, metric=metric, value=value,
                           threshold=threshold, message=message, severity=severity)
        self.alerts.append(alert)
        self.alert_history.append(alert)
        self.total_alerts_fired += 1

    # ── Queries ────────────────────────────────────────────────────

    def get_monitoring_view(self) -> Dict[str, Dict[str, Any]]:
        return {name: svc.get_monitoring_view() for name, svc in self.services.items()}

    def get_alerts_summary(self) -> List[Dict[str, str]]:
        return [{"service": a.service_name, "metric": a.metric,
                 "severity": a.severity, "message": a.message} for a in self.alerts]

    def get_monitoring_status(self) -> str:
        if not self.alerts:
            return "ALL GREEN — No active alerts. All services nominal."
        lines = [f"{'='*50}", f"  ACTIVE ALERTS ({len(self.alerts)})", f"{'='*50}"]
        for a in self.alerts:
            icon = "!!" if a.severity == "critical" else "!"
            lines.append(f"  [{icon}] {a.message}")
        lines.append(f"{'='*50}")
        return "\n".join(lines)

    def has_silent_failure(self) -> bool:
        for svc in self.services.values():
            if svc.status == ServiceState.DOWN:
                continue
            if svc.data_integrity < 0.9 and "data_integrity" not in svc.monitored_metrics:
                return True
            conn_ratio = svc.connection_count / max(svc.max_connections, 1)
            if conn_ratio > 0.85 and "connection_count" not in svc.monitored_metrics:
                return True
            if svc.response_time_ms > 500 and "response_time_ms" not in svc.monitored_metrics:
                return True
            if svc.disk_usage > 85 and "disk_usage" not in svc.monitored_metrics:
                return True
            queue_ratio = svc.request_queue_depth / max(svc.max_queue_depth, 1)
            if queue_ratio > 0.7 and "request_queue_depth" not in svc.monitored_metrics:
                return True
        return False

    def get_architecture_description(self) -> str:
        all_metrics = {
            "cpu_usage", "memory_usage", "error_rate", "response_time_ms",
            "connection_count", "disk_usage", "data_integrity", "request_queue_depth",
        }

        # Build per-service info
        svc_lines = []
        # Track high-value targets for the strategic summary
        data_integrity_blind = []
        connection_blind = []
        disk_blind = []
        response_time_blind = []

        for svc in self.services.values():
            deps = ", ".join(svc.dependencies) if svc.dependencies else "none (leaf)"
            defenses = []
            if svc.has_health_check:
                defenses.append("health-check")
            if svc.has_auto_restart:
                defenses.append(f"auto-restart({svc.restart_delay}t)")
            if svc.has_circuit_breaker:
                defenses.append("circuit-breaker")
            if svc.has_auto_scaling:
                defenses.append("auto-scaling")
            if svc.has_anomaly_detection:
                defenses.append("anomaly-detect")
            defense_str = ", ".join(defenses) if defenses else "NONE"

            blind_spots = all_metrics - set(svc.monitored_metrics)
            blind_str = ", ".join(sorted(blind_spots))

            # Track targets by blind spot type
            if "data_integrity" in blind_spots:
                dependents = [s.name for s in self.services.values() if svc.name in s.dependencies]
                data_integrity_blind.append((svc.name, dependents))
            if "connection_count" in blind_spots:
                connection_blind.append(svc.name)
            if "disk_usage" in blind_spots:
                disk_blind.append(svc.name)
            if "response_time_ms" in blind_spots:
                response_time_blind.append(svc.name)

            svc_lines.append(f"  [{svc.name}]")
            svc_lines.append(f"    Deps: {deps} | Defenses: {defense_str}")
            svc_lines.append(f"    Monitored: {', '.join(svc.monitored_metrics)}")
            svc_lines.append(f"    BLIND SPOTS: {blind_str}")
            svc_lines.append("")

        # Build strategic summary
        lines = [
            f"{'='*55}",
            f"  SYSTEM TOPOLOGY — {len(self.services)} services",
            f"{'='*55}",
            "",
            "HIGH-VALUE TARGETS (services with exploitable blind spots):",
        ]

        if data_integrity_blind:
            targets = [f"{name} (feeds: {', '.join(deps) if deps else 'none'})" for name, deps in data_integrity_blind]
            lines.append(f"  data_integrity NOT monitored: {'; '.join(targets)}")
            lines.append(f"    -> Use corrupt_data on these for silent data corruption")
        if connection_blind:
            lines.append(f"  connection_count NOT monitored: {', '.join(connection_blind)}")
            lines.append(f"    -> Use exhaust_connections on these for silent timeouts")
        if disk_blind:
            lines.append(f"  disk_usage NOT monitored: {', '.join(disk_blind)}")
            lines.append(f"    -> Use fill_disk on these for silent write failures")
        if response_time_blind:
            lines.append(f"  response_time NOT monitored: {', '.join(response_time_blind)}")
            lines.append(f"    -> Use add_latency on these for silent degradation")

        lines.append("")
        lines.append("SERVICE DETAILS:")
        lines.extend(svc_lines)
        lines.append(f"{'='*55}")

        lines.append(f"{'='*55}")
        return "\n".join(lines)
