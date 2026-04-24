"""
Task scenario definitions for the Chaos Auditor environment.

Each scenario defines:
- System topology (services and dependencies)
- Self-healing configuration
- Monitoring coverage (and blind spots)
- Ground truth vulnerabilities with expected findings
- Scoring parameters
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from chaos_auditor.server.simulation import Service, ServiceGraph


@dataclass
class GroundTruthVulnerability:
    """A known vulnerability the agent should discover."""

    name: str
    finding_type: str  # e.g., "single_point_of_failure", "silent_data_corruption"
    severity: str  # "low", "medium", "high", "critical"
    is_silent: bool
    affected_services: List[str]
    root_cause: str
    how_to_trigger: str  # Description for grading reference
    weight: float  # Fraction of total score this vulnerability is worth


@dataclass
class Scenario:
    name: str
    description: str
    chaos_budget: int
    max_steps: int
    vulnerabilities: List[GroundTruthVulnerability] = field(default_factory=list)

    def build_graph(self) -> ServiceGraph:
        raise NotImplementedError


class EasyScenario(Scenario):
    """4-service linear system with basic defenses."""

    def __init__(self) -> None:
        super().__init__(
            name="easy",
            description=(
                "A simple web application with 4 services. "
                "Basic health checks and auto-restart only. "
                "Find the vulnerabilities — especially the ones monitoring can't see."
            ),
            chaos_budget=8,
            max_steps=15,
            vulnerabilities=[
                GroundTruthVulnerability(
                    name="database_spof",
                    finding_type="single_point_of_failure",
                    severity="high",
                    is_silent=False,
                    affected_services=["database", "app-server"],
                    root_cause="Database has no replica. Killing it takes down app-server.",
                    how_to_trigger="kill(database)",
                    weight=0.25,
                ),
                GroundTruthVulnerability(
                    name="cache_corruption",
                    finding_type="silent_data_corruption",
                    severity="critical",
                    is_silent=True,
                    affected_services=["redis-cache", "app-server"],
                    root_cause=(
                        "Corrupting redis-cache data causes app-server to serve wrong data. "
                        "data_integrity is not monitored — no alert fires."
                    ),
                    how_to_trigger="corrupt_data(redis-cache, cache)",
                    weight=0.35,
                ),
                GroundTruthVulnerability(
                    name="disk_silent_failure",
                    finding_type="silent_disk_pressure",
                    severity="high",
                    is_silent=True,
                    affected_services=["database"],
                    root_cause=(
                        "Filling database disk causes writes to fail silently. "
                        "disk_usage is not monitored on this service."
                    ),
                    how_to_trigger="fill_disk(database, 95)",
                    weight=0.25,
                ),
                GroundTruthVulnerability(
                    name="connection_exhaustion",
                    finding_type="silent_connection_exhaustion",
                    severity="medium",
                    is_silent=True,
                    affected_services=["database"],
                    root_cause=(
                        "Exhausting database connections causes new queries to time out. "
                        "connection_count is not monitored."
                    ),
                    how_to_trigger="exhaust_connections(database)",
                    weight=0.15,
                ),
            ],
        )

    def build_graph(self) -> ServiceGraph:
        graph = ServiceGraph()
        graph.add_service(
            Service(
                name="api-gateway",
                dependencies=["app-server"],
                has_health_check=True,
                has_auto_restart=True,
                monitored_metrics=["cpu_usage", "error_rate", "status"],
            )
        )
        graph.add_service(
            Service(
                name="app-server",
                dependencies=["database", "redis-cache"],
                has_health_check=True,
                has_auto_restart=True,
                monitored_metrics=["cpu_usage", "error_rate", "status"],
            )
        )
        graph.add_service(
            Service(
                name="database",
                dependencies=[],
                has_health_check=True,
                has_auto_restart=True,
                restart_delay=4,
                max_connections=100,
                monitored_metrics=["cpu_usage", "error_rate", "status"],
                # NOTE: disk_usage, connection_count, data_integrity NOT monitored
            )
        )
        graph.add_service(
            Service(
                name="redis-cache",
                dependencies=[],
                has_health_check=True,
                has_auto_restart=True,
                monitored_metrics=["cpu_usage", "status"],
                # NOTE: data_integrity NOT monitored — this is the blind spot
            )
        )
        return graph


class MediumScenario(Scenario):
    """7-service system with redundancy and circuit breakers."""

    def __init__(self) -> None:
        super().__init__(
            name="medium",
            description=(
                "A web application with 7 services including redundancy (2 API gateways, "
                "redis replica). Circuit breakers and load balancing are enabled. "
                "Find the monitoring blind spots and silent failure paths."
            ),
            chaos_budget=10,
            max_steps=20,
            vulnerabilities=[
                GroundTruthVulnerability(
                    name="dual_gateway_kill",
                    finding_type="redundancy_bypass",
                    severity="high",
                    is_silent=False,
                    affected_services=["api-gateway-1", "api-gateway-2"],
                    root_cause="Killing both gateways simultaneously bypasses load balancer redundancy.",
                    how_to_trigger="kill(api-gateway-1) + kill(api-gateway-2)",
                    weight=0.15,
                ),
                GroundTruthVulnerability(
                    name="connection_pool_cascade",
                    finding_type="silent_connection_exhaustion",
                    severity="critical",
                    is_silent=True,
                    affected_services=["auth-service", "user-db"],
                    root_cause=(
                        "Spiking traffic to auth-service causes user-db connection pool exhaustion. "
                        "connection_count is not monitored — silent timeout cascade."
                    ),
                    how_to_trigger="spike_traffic(auth-service, 4)",
                    weight=0.25,
                ),
                GroundTruthVulnerability(
                    name="corruption_replication",
                    finding_type="silent_data_corruption",
                    severity="critical",
                    is_silent=True,
                    affected_services=["redis-primary", "redis-replica", "cache-layer"],
                    root_cause=(
                        "Corrupting redis-primary data propagates to redis-replica via replication. "
                        "cache-layer serves corrupted data. data_integrity not monitored anywhere."
                    ),
                    how_to_trigger="corrupt_data(redis-primary, cache)",
                    weight=0.30,
                ),
                GroundTruthVulnerability(
                    name="sub_threshold_latency",
                    finding_type="silent_latency_cascade",
                    severity="high",
                    is_silent=True,
                    affected_services=["payment-db", "payment-svc"],
                    root_cause=(
                        "Adding latency to payment-db (below alert threshold) causes "
                        "payment-svc to retry, creating exponential connection growth. "
                        "connection_count not monitored on payment-db."
                    ),
                    how_to_trigger="add_latency(payment-db, 400)",
                    weight=0.30,
                ),
            ],
        )

    def build_graph(self) -> ServiceGraph:
        graph = ServiceGraph()
        graph.add_service(
            Service(
                name="load-balancer",
                dependencies=["api-gateway-1", "api-gateway-2"],
                has_health_check=True,
                has_auto_restart=True,
                monitored_metrics=["cpu_usage", "error_rate", "status"],
            )
        )
        graph.add_service(
            Service(
                name="api-gateway-1",
                dependencies=["auth-service", "payment-svc", "cache-layer"],
                has_health_check=True,
                has_auto_restart=True,
                monitored_metrics=["cpu_usage", "error_rate", "status", "memory_usage"],
            )
        )
        graph.add_service(
            Service(
                name="api-gateway-2",
                dependencies=["auth-service", "payment-svc", "cache-layer"],
                has_health_check=True,
                has_auto_restart=True,
                monitored_metrics=["cpu_usage", "error_rate", "status", "memory_usage"],
            )
        )
        graph.add_service(
            Service(
                name="auth-service",
                dependencies=["user-db"],
                has_health_check=True,
                has_auto_restart=True,
                has_circuit_breaker=True,
                monitored_metrics=["cpu_usage", "error_rate", "status"],
            )
        )
        graph.add_service(
            Service(
                name="user-db",
                dependencies=[],
                has_health_check=True,
                has_auto_restart=True,
                restart_delay=4,
                max_connections=150,
                monitored_metrics=["cpu_usage", "error_rate", "status"],
            )
        )
        graph.add_service(
            Service(
                name="payment-svc",
                dependencies=["payment-db"],
                has_health_check=True,
                has_auto_restart=True,
                has_circuit_breaker=True,
                monitored_metrics=["cpu_usage", "error_rate", "status"],
            )
        )
        graph.add_service(
            Service(
                name="payment-db",
                dependencies=[],
                has_health_check=True,
                has_auto_restart=True,
                restart_delay=4,
                max_connections=100,
                monitored_metrics=["cpu_usage", "error_rate", "status"],
            )
        )
        graph.add_service(
            Service(
                name="cache-layer",
                dependencies=["redis-primary", "redis-replica"],
                has_health_check=True,
                has_auto_restart=True,
                monitored_metrics=["cpu_usage", "status"],
            )
        )
        graph.add_service(
            Service(
                name="redis-primary",
                dependencies=[],
                has_health_check=True,
                has_auto_restart=True,
                monitored_metrics=["cpu_usage", "status"],
            )
        )
        graph.add_service(
            Service(
                name="redis-replica",
                dependencies=["redis-primary"],
                has_health_check=True,
                has_auto_restart=True,
                monitored_metrics=["cpu_usage", "status"],
            )
        )
        return graph


class HardScenario(Scenario):
    """12-service fortress with full self-healing and anomaly detection."""

    def __init__(self) -> None:
        super().__init__(
            name="hard",
            description=(
                "A heavily defended production system with 12 services. "
                "Full self-healing: auto-scaling, circuit breakers, anomaly detection, "
                "clustered redis, database replicas. "
                "Finding vulnerabilities requires chaining subtle attacks that stay below "
                "alert thresholds. The highest-value findings are silent failures."
            ),
            chaos_budget=12,
            max_steps=25,
            vulnerabilities=[
                GroundTruthVulnerability(
                    name="cluster_quorum_loss",
                    finding_type="cluster_quorum_failure",
                    severity="high",
                    is_silent=True,
                    affected_services=["redis-1", "redis-2", "cache-cluster"],
                    root_cause=(
                        "Killing redis-1 + redis-2 causes quorum loss in the 3-node cluster. "
                        "redis-3 stays 'healthy' but can't form consensus. "
                        "cache-cluster serves stale data. No alert fires because individual "
                        "node health checks pass."
                    ),
                    how_to_trigger="kill(redis-1) + kill(redis-2)",
                    weight=0.15,
                ),
                GroundTruthVulnerability(
                    name="combined_sub_threshold",
                    finding_type="compound_silent_cascade",
                    severity="critical",
                    is_silent=True,
                    affected_services=[
                        "elasticsearch",
                        "search-service",
                        "order-service",
                        "order-db",
                    ],
                    root_cause=(
                        "Adding latency to elasticsearch (below threshold) + spiking traffic "
                        "to search-service (below auto-scale trigger) = combined effect causes "
                        "search-service thread exhaustion -> order-service timeouts -> "
                        "order-db connection pool creep. connection_count not monitored."
                    ),
                    how_to_trigger="add_latency(elasticsearch, 300) + spike_traffic(search-service, 1.8)",
                    weight=0.25,
                ),
                GroundTruthVulnerability(
                    name="ml_model_staleness",
                    finding_type="silent_data_corruption",
                    severity="high",
                    is_silent=True,
                    affected_services=["ml-model-cache", "recommend-svc"],
                    root_cause=(
                        "Corrupting ml-model-cache causes recommend-svc to serve stale/wrong "
                        "recommendations. No metric tracks recommendation quality. "
                        "All monitoring stays GREEN."
                    ),
                    how_to_trigger="corrupt_data(ml-model-cache, model)",
                    weight=0.20,
                ),
                GroundTruthVulnerability(
                    name="queue_overflow",
                    finding_type="silent_resource_exhaustion",
                    severity="high",
                    is_silent=True,
                    affected_services=["message-queue", "notification-svc"],
                    root_cause=(
                        "Spiking traffic to notification-svc fills message-queue beyond "
                        "the monitored threshold. Queue depth monitoring only checks up to "
                        "a fixed value. Overflow goes undetected until OOM."
                    ),
                    how_to_trigger="spike_traffic(notification-svc, 3)",
                    weight=0.20,
                ),
                GroundTruthVulnerability(
                    name="circuit_breaker_masking",
                    finding_type="defense_masking_failure",
                    severity="critical",
                    is_silent=True,
                    affected_services=["order-service", "order-db"],
                    root_cause=(
                        "Network partition between order-service and order-db triggers "
                        "circuit breaker. Orders queue in memory. Traffic spike causes "
                        "memory growth. Circuit breaker masks the failure from monitoring — "
                        "load balancer still sends traffic. Orders silently dropped."
                    ),
                    how_to_trigger="partition_network(order-service, order-db) + spike_traffic(order-service, 2)",
                    weight=0.20,
                ),
            ],
        )

    def build_graph(self) -> ServiceGraph:
        graph = ServiceGraph()

        graph.add_service(
            Service(
                name="load-balancer",
                dependencies=["api-gateway-1", "api-gateway-2", "api-gateway-3"],
                has_health_check=True,
                has_auto_restart=True,
                monitored_metrics=[
                    "cpu_usage", "error_rate", "status", "memory_usage",
                ],
            )
        )
        for i in range(1, 4):
            graph.add_service(
                Service(
                    name=f"api-gateway-{i}",
                    dependencies=[
                        "auth-service", "search-service", "recommend-svc",
                        "order-service", "notification-svc", "cache-cluster",
                    ],
                    has_health_check=True,
                    has_auto_restart=True,
                    has_auto_scaling=True,
                    has_anomaly_detection=True,
                    restart_delay=1,
                    monitored_metrics=[
                        "cpu_usage", "error_rate", "status", "memory_usage",
                    ],
                )
            )

        graph.add_service(
            Service(
                name="auth-service",
                dependencies=["user-db-primary"],
                has_health_check=True,
                has_auto_restart=True,
                has_circuit_breaker=True,
                restart_delay=1,
                monitored_metrics=[
                    "cpu_usage", "error_rate", "status", "memory_usage",
                ],
            )
        )
        graph.add_service(
            Service(
                name="user-db-primary",
                dependencies=[],
                has_health_check=True,
                has_auto_restart=True,
                restart_delay=2,
                max_connections=200,
                monitored_metrics=["cpu_usage", "error_rate", "status", "disk_usage"],
            )
        )
        graph.add_service(
            Service(
                name="search-service",
                dependencies=["elasticsearch"],
                has_health_check=True,
                has_auto_restart=True,
                has_auto_scaling=True,
                max_connections=150,
                restart_delay=1,
                monitored_metrics=[
                    "cpu_usage", "error_rate", "status", "memory_usage",
                ],
            )
        )
        graph.add_service(
            Service(
                name="elasticsearch",
                dependencies=[],
                has_health_check=True,
                has_auto_restart=True,
                restart_delay=3,
                monitored_metrics=["cpu_usage", "error_rate", "status"],
            )
        )
        graph.add_service(
            Service(
                name="recommend-svc",
                dependencies=["ml-model-cache"],
                has_health_check=True,
                has_auto_restart=True,
                has_circuit_breaker=True,
                restart_delay=1,
                monitored_metrics=["cpu_usage", "error_rate", "status"],
            )
        )
        graph.add_service(
            Service(
                name="ml-model-cache",
                dependencies=[],
                has_health_check=True,
                has_auto_restart=True,
                monitored_metrics=["cpu_usage", "status"],
                # data_integrity NOT monitored — blind spot
            )
        )
        graph.add_service(
            Service(
                name="order-service",
                dependencies=["order-db"],
                has_health_check=True,
                has_auto_restart=True,
                has_circuit_breaker=True,
                max_connections=150,
                restart_delay=1,
                monitored_metrics=[
                    "cpu_usage", "error_rate", "status", "memory_usage",
                ],
            )
        )
        graph.add_service(
            Service(
                name="order-db",
                dependencies=[],
                has_health_check=True,
                has_auto_restart=True,
                restart_delay=3,
                max_connections=200,
                monitored_metrics=["cpu_usage", "error_rate", "status"],
                # connection_count NOT monitored
            )
        )
        graph.add_service(
            Service(
                name="notification-svc",
                dependencies=["message-queue"],
                has_health_check=True,
                has_auto_restart=True,
                restart_delay=1,
                monitored_metrics=["cpu_usage", "error_rate", "status"],
            )
        )
        graph.add_service(
            Service(
                name="message-queue",
                dependencies=[],
                has_health_check=True,
                has_auto_restart=True,
                max_connections=300,
                monitored_metrics=["cpu_usage", "status"],
                # memory_usage, connection_count NOT monitored
            )
        )
        graph.add_service(
            Service(
                name="cache-cluster",
                dependencies=["redis-1", "redis-2", "redis-3"],
                has_health_check=True,
                has_auto_restart=True,
                monitored_metrics=["cpu_usage", "status"],
            )
        )
        for i in range(1, 4):
            graph.add_service(
                Service(
                    name=f"redis-{i}",
                    dependencies=[],
                    has_health_check=True,
                    has_auto_restart=True,
                    restart_delay=2,
                    monitored_metrics=["cpu_usage", "status"],
                )
            )

        return graph


class RandomScenario(Scenario):
    """
    Procedurally generated scenario — different every episode.

    Randomly creates a service graph with:
    - 4 to 12 services in a layered DAG
    - Random monitoring blind spots per service
    - Random defense configurations
    - Ground truth vulnerabilities computed from graph structure

    This makes Chaos Auditor an RLVE-compliant environment:
    infinite tasks, adaptive difficulty, no saturation.
    """

    # Service name pools by role
    _FRONTEND = ["api-gateway", "load-balancer", "nginx-proxy", "cdn-edge"]
    _MIDTIER  = ["app-server", "auth-service", "payment-svc", "order-service",
                 "search-service", "recommend-svc", "notification-svc", "user-service"]
    _BACKEND  = ["database", "user-db", "order-db", "payment-db",
                 "redis-cache", "redis-primary", "elasticsearch", "message-queue",
                 "ml-model-cache", "blob-storage"]

    # Metrics that can be left unmonitored (blind spots)
    _BLINDABLE = [
        "data_integrity", "connection_count", "disk_usage",
        "memory_usage", "response_time_ms", "request_queue_depth",
    ]
    # Always monitored — removing these would make the env trivial
    _ALWAYS_MONITORED = ["cpu_usage", "error_rate", "status"]

    def __init__(self, seed: Optional[int] = None) -> None:
        rng = random.Random(seed)
        n_services = rng.randint(4, 12)

        # Scale budget and steps with complexity
        chaos_budget = max(6, n_services)
        max_steps = max(12, n_services * 2)

        super().__init__(
            name="random",
            description=(
                f"Procedurally generated system with {n_services} services. "
                "Topology, blind spots, and defenses are randomized each episode. "
                "Discover the monitoring gaps and exploit them silently."
            ),
            chaos_budget=chaos_budget,
            max_steps=max_steps,
        )
        self._rng = rng
        self._n_services = n_services
        self._built_graph: Optional[ServiceGraph] = None
        self._built_vulns: Optional[List[GroundTruthVulnerability]] = None

    def build_graph(self) -> ServiceGraph:
        graph = ServiceGraph()
        rng = self._rng

        # Pick service names without repetition
        frontends = rng.sample(self._FRONTEND, min(2, self._n_services))
        remaining = self._n_services - len(frontends)
        midtier_n = max(0, min(remaining // 2, len(self._MIDTIER)))
        backend_n = max(1, remaining - midtier_n)

        midtiers = rng.sample(self._MIDTIER, min(midtier_n, len(self._MIDTIER)))
        backends = rng.sample(self._BACKEND, min(backend_n, len(self._BACKEND)))

        all_names: List[str] = frontends + midtiers + backends

        # Build layered dependency graph: frontend → midtier → backend
        service_objects: Dict[str, Service] = {}

        for name in all_names:
            # Assign blind spots: randomly drop 2-4 metrics from monitoring
            n_blind = rng.randint(2, 4)
            blind = set(rng.sample(self._BLINDABLE, min(n_blind, len(self._BLINDABLE))))
            monitored = self._ALWAYS_MONITORED + [
                m for m in ["memory_usage", "response_time_ms", "connection_count",
                            "disk_usage", "data_integrity", "request_queue_depth"]
                if m not in blind
            ]

            # Random defenses
            has_cb = rng.random() < 0.4
            has_as = rng.random() < 0.3
            has_ad = rng.random() < 0.2
            restart_delay = rng.choice([2, 3, 4])
            max_conn = rng.choice([100, 150, 200, 300])

            service_objects[name] = Service(
                name=name,
                has_health_check=True,
                has_auto_restart=True,
                has_circuit_breaker=has_cb,
                has_auto_scaling=has_as,
                has_anomaly_detection=has_ad,
                restart_delay=restart_delay,
                max_connections=max_conn,
                monitored_metrics=monitored,
            )

        # Wire dependencies: frontend → midtier, midtier → backend
        for fname in frontends:
            targets = midtiers if midtiers else backends
            n_deps = min(rng.randint(1, 2), len(targets))
            service_objects[fname].dependencies = rng.sample(targets, n_deps)

        for mname in midtiers:
            if backends:
                n_deps = min(rng.randint(1, 2), len(backends))
                service_objects[mname].dependencies = rng.sample(backends, n_deps)

        for svc in service_objects.values():
            graph.add_service(svc)

        # Compute ground truth vulnerabilities from graph structure
        # Cap at 5 vulnerabilities max — keeps grading clean and rewards meaningful
        MAX_VULNS = 5
        vulns: List[GroundTruthVulnerability] = []
        weight_pool = 1.0
        seen_types: Dict[str, int] = {}  # cap 2 per type for variety

        for name, svc in service_objects.items():
            if len(vulns) >= MAX_VULNS or weight_pool <= 0.05:
                break

            blind_set = set(self._BLINDABLE) - set(svc.monitored_metrics)

            # Silent data corruption: data_integrity blind + has downstream consumers
            consumers = [
                n for n, s in service_objects.items()
                if name in s.dependencies
            ]
            if (
                "data_integrity" in blind_set
                and consumers
                and seen_types.get("silent_data_corruption", 0) < 2
                and len(vulns) < MAX_VULNS
            ):
                w = round(min(0.30, weight_pool * 0.35), 2)
                weight_pool -= w
                seen_types["silent_data_corruption"] = seen_types.get("silent_data_corruption", 0) + 1
                vulns.append(GroundTruthVulnerability(
                    name=f"corruption_{name}",
                    finding_type="silent_data_corruption",
                    severity="critical",
                    is_silent=True,
                    affected_services=[name] + consumers[:2],
                    root_cause=(
                        f"Corrupting {name} data propagates to {', '.join(consumers[:2])}. "
                        f"data_integrity is not monitored — no alert fires."
                    ),
                    how_to_trigger=f"corrupt_data({name})",
                    weight=w,
                ))

            # Silent connection exhaustion: connection_count blind
            if (
                "connection_count" in blind_set
                and seen_types.get("silent_connection_exhaustion", 0) < 2
                and len(vulns) < MAX_VULNS
            ):
                w = round(min(0.25, weight_pool * 0.30), 2)
                weight_pool -= w
                upstream = [n for n, s in service_objects.items() if name in s.dependencies]
                seen_types["silent_connection_exhaustion"] = seen_types.get("silent_connection_exhaustion", 0) + 1
                vulns.append(GroundTruthVulnerability(
                    name=f"conn_exhaust_{name}",
                    finding_type="silent_connection_exhaustion",
                    severity="high",
                    is_silent=True,
                    affected_services=[name] + upstream[:1],
                    root_cause=(
                        f"Exhausting {name} connection pool causes new requests to time out. "
                        f"connection_count is not monitored."
                    ),
                    how_to_trigger=f"exhaust_connections({name})",
                    weight=w,
                ))

            # Silent disk pressure: disk_usage blind
            if (
                "disk_usage" in blind_set
                and weight_pool > 0.1
                and seen_types.get("silent_disk_pressure", 0) < 1
                and len(vulns) < MAX_VULNS
            ):
                w = round(min(0.20, weight_pool * 0.25), 2)
                weight_pool -= w
                seen_types["silent_disk_pressure"] = seen_types.get("silent_disk_pressure", 0) + 1
                vulns.append(GroundTruthVulnerability(
                    name=f"disk_{name}",
                    finding_type="silent_disk_pressure",
                    severity="high",
                    is_silent=True,
                    affected_services=[name],
                    root_cause=(
                        f"Filling {name} disk causes write failures. "
                        f"disk_usage is not monitored."
                    ),
                    how_to_trigger=f"fill_disk({name}, 95)",
                    weight=w,
                ))

            # Single point of failure: multiple upstream dependents, no replica
            upstream = [n for n, s in service_objects.items() if name in s.dependencies]
            if (
                len(upstream) >= 2
                and weight_pool > 0.1
                and seen_types.get("single_point_of_failure", 0) < 1
                and len(vulns) < MAX_VULNS
            ):
                w = round(min(0.20, weight_pool * 0.25), 2)
                weight_pool -= w
                seen_types["single_point_of_failure"] = seen_types.get("single_point_of_failure", 0) + 1
                vulns.append(GroundTruthVulnerability(
                    name=f"spof_{name}",
                    finding_type="single_point_of_failure",
                    severity="high",
                    is_silent=False,
                    affected_services=[name] + upstream[:2],
                    root_cause=(
                        f"{name} has no replica. Killing it takes down "
                        f"{', '.join(upstream[:2])}."
                    ),
                    how_to_trigger=f"kill({name})",
                    weight=w,
                ))

        # Ensure at least 2 vulnerabilities and weights sum correctly
        if not vulns:
            # Fallback: guarantee at least one silent finding
            first_backend = backends[0] if backends else all_names[-1]
            vulns.append(GroundTruthVulnerability(
                name=f"fallback_corruption_{first_backend}",
                finding_type="silent_data_corruption",
                severity="high",
                is_silent=True,
                affected_services=[first_backend],
                root_cause=f"data_integrity not monitored on {first_backend}.",
                how_to_trigger=f"corrupt_data({first_backend})",
                weight=0.50,
            ))
            vulns.append(GroundTruthVulnerability(
                name=f"fallback_conn_{first_backend}",
                finding_type="silent_connection_exhaustion",
                severity="medium",
                is_silent=True,
                affected_services=[first_backend],
                root_cause=f"connection_count not monitored on {first_backend}.",
                how_to_trigger=f"exhaust_connections({first_backend})",
                weight=0.50,
            ))

        self.vulnerabilities = vulns
        self._built_graph = graph
        self._built_vulns = vulns
        return graph


SCENARIOS: Dict[str, Scenario] = {
    "easy": EasyScenario(),
    "medium": MediumScenario(),
    "hard": HardScenario(),
}


def get_scenario(name: str) -> Scenario:
    if name == "random":
        return RandomScenario()
    if name not in SCENARIOS:
        raise ValueError(
            f"Unknown scenario: '{name}'. Available: {list(SCENARIOS.keys()) + ['random']}"
        )
    return SCENARIOS[name]
