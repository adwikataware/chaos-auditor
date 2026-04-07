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

from dataclasses import dataclass, field
from typing import Any, Dict, List

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


SCENARIOS: Dict[str, Scenario] = {
    "easy": EasyScenario(),
    "medium": MediumScenario(),
    "hard": HardScenario(),
}


def get_scenario(name: str) -> Scenario:
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}. Available: {list(SCENARIOS.keys())}")
    return SCENARIOS[name]
