"""
Fleet health monitor: machines report healthy/unhealthy; also stale if
no message for > stale_duration since last heartbeat.
"""
from dataclasses import dataclass


@dataclass
class HealthMessage:
    machine_id: int
    timestamp: int
    is_healthy: bool


class FleetHealthMonitor:
    def __init__(self, stale_duration: int):
        self.stale_duration = stale_duration
        self.last_timestamp: dict[int, int] = {}
        self.last_healthy: dict[int, bool] = {}
        self.current_time = 0

    def process_message(self, message: HealthMessage) -> None:
        self.current_time = max(self.current_time, message.timestamp)
        self.last_timestamp[message.machine_id] = message.timestamp
        self.last_healthy[message.machine_id] = message.is_healthy

    def is_machine_healthy(self, machine_id: int) -> bool:
        if machine_id not in self.last_timestamp:
            return False
        if self.current_time - self.last_timestamp[machine_id] > self.stale_duration:
            return False
        return self.last_healthy[machine_id]


if __name__ == "__main__":
    monitor = FleetHealthMonitor(stale_duration=10)
    monitor.process_message(HealthMessage(machine_id=1, timestamp=1, is_healthy=True))
    monitor.process_message(HealthMessage(machine_id=1, timestamp=11, is_healthy=False))
    print(monitor.is_machine_healthy(1))  # False (last report: unhealthy)

    # Stale: machine 1 last heard at t=1; clock advances to t=12 via another message
    monitor2 = FleetHealthMonitor(stale_duration=10)
    monitor2.process_message(HealthMessage(machine_id=1, timestamp=1, is_healthy=True))
    monitor2.process_message(HealthMessage(machine_id=2, timestamp=12, is_healthy=True))
    print(monitor2.is_machine_healthy(1))  # False (12 - 1 > 10)
