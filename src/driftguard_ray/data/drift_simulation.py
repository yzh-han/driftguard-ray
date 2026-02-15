"""Generate drift events for federated learning simulations."""

from collections import deque
import random
from dataclasses import dataclass
from typing import List
from driftguard_ray.config import get_logger

logger = get_logger("drift_simulation")

@dataclass(frozen=True)
class DriftEvent:
    """Single drift event; duration=0.0 indicates sudden drift.

    Attributes:
        time_step: Time step when drift starts.
        clients: List of client IDs affected.
        duration: Drift duration in steps; 0 means sudden drift.
        drift_dist: Drift distance (integer step count).
    """

    id: int
    time_step: int
    clients: List[int]
    duration: int
    drift_dist: int


class ClientState:
    """Client state holding domain probabilities.

    Attributes:
        num_domains: Number of domains.
        num_clients: Number of clients.
        rng: Internal random generator.
        c_to_dist: Mapping of client_id -> distribution (list of probabilities).
        c_to_dest: Mapping of client_id -> current destination domain index.
    """

    def __init__(
        self, num_domains: int, num_clients: int, seed: int | None = None
    ) -> None:
        self.rng: random.Random = random.Random(seed)

        self.num_domains: int = num_domains

        self.c_to_dist: dict[int, list[float]] = {
            cid: [1.0] + [0.0] * (num_domains - 1) for cid in range(num_clients)
        }  # cid -> distribution (list of probabilities)

        self.c_to_dest: dict[int, int] = {
            cid: 0 for cid in range(num_clients)
        }  # cid -> destination domain index

    def update(self, event: DriftEvent) -> None:
        """Update client domain probabilities based on a drift event.

        Args:
            event: DriftEvent instance.
        """
        for cid in event.clients:
            _old_dist = self.c_to_dist[cid].copy()

            dest = self.c_to_dest[cid] = (
                self.c_to_dest[cid] + event.drift_dist
            ) % self.num_domains  # new domain destination index

            dist = self.c_to_dist[cid]
            dest_prob = dist[dest] + 1.0 / event.duration

            remain_other = 1.0 - dest_prob
            

            if remain_other <= 0.0:
                dist[:] = [0.0] * self.num_domains
                dist[dest] = 1.0
            else:
                total_other = sum(dist) - dist[dest]
                assert total_other > 0.0, "total_other must > 0.0"
                for d in range(self.num_domains):
                    dist[d] = (
                        dist[d] * remain_other / total_other if d != dest else dest_prob
                    )
            logger.info(f" [Drift] cid: {cid}, time_step: {event.time_step} - " \
                        f"old: {[f'{f:.2f}' for f in _old_dist]}, -> "\
                        f"new: {[f'{f:.2f}' for f in self.c_to_dist[cid]]}")
            assert abs(sum(dist) - 1.0) < 1e-6, "Probabilities must sum to 1.0"

    def get_distribution(self, cid: int) -> list[float]:
        """Get the current domain distribution for a client.

        Args:
            cid: Client identifier.
        Returns:
            List of probabilities for each domain.
        """
        return self.c_to_dist[cid]


@dataclass
class DriftEventArgs:
    n_time_steps: int = 20
    n_clients: int = 10
    n_sudden: int = 2
    n_gradual: int = 2
    n_stage: int = 3
    aff_client_ratio_range: tuple[float, float] = (0.1, 0.5)
    start: float = 0.0
    end: float = 0.85
    dist_range: tuple[int, int] = (1, 3)
    gradual_duration_ratio: float = 0.15
    seed: int | None = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.start < self.end <= 1.0):
            raise ValueError("start/end must satisfy 0.0 <= start < end <= 1.0")
        if self.n_stage <= 0:
            raise ValueError("n_stage must be positive")
        if not (
            0.0
            <= self.aff_client_ratio_range[0]
            <= self.aff_client_ratio_range[1]
            <= 1.0
        ):
            raise ValueError("aff_client_ratio_range must be within [0.0, 1.0]")
        if self.dist_range[0] <= 0 or self.dist_range[1] < self.dist_range[0]:
            raise ValueError("dist_range must be positive and valid")
        if not (0.0 <= self.gradual_duration_ratio <= 1.0):
            raise ValueError("gradual_duration_ratio must be within [0.0, 1.0]")


def generate_drift_events(
    args: DriftEventArgs,
) -> deque[DriftEvent]:
    """Generate drift events for federated learning simulation.
    Args:
        args: DriftEventArgs instance containing configuration.
    Returns:
        List of DriftEvent instances.
    """
    rng = random.Random(args.seed)
    events: list[DriftEvent] = []
    stage_length = (args.end - args.start) / args.n_stage
    total_events = args.n_sudden + args.n_gradual
    max_duration_steps = max(1, int(args.gradual_duration_ratio * args.n_time_steps))
    # Generate events per stage.
    for stage_idx in range(args.n_stage):
        stage_start_step = (
            int(args.n_time_steps * (args.start + stage_idx * stage_length)) + 1
        )
        stage_end_step = (
            int(args.n_time_steps * (args.start + (stage_idx + 1) * stage_length)) + 1
        )
        stage_span = stage_end_step - stage_start_step
        if stage_span <= 0:
            raise ValueError("stage length must be positive")
        if total_events > stage_span:
            raise ValueError("stage is too small for the requested events")

        # Generate drift events.
        time_steps = rng.sample(range(stage_start_step, stage_end_step), k=total_events)

        for idx, t in enumerate(time_steps):
            # Select affected clients
            n_aff_client = rng.randint(
                int(args.aff_client_ratio_range[0] * args.n_clients),
                int(args.aff_client_ratio_range[1] * args.n_clients),
            )

            clients = rng.sample(range(args.n_clients), k=n_aff_client)
            # drift distance
            drift_dist = rng.randint(args.dist_range[0], args.dist_range[1])

            # duration
            if idx < args.n_sudden:
                duration = 1
            else:
                duration = rng.randint(2, 2 + max_duration_steps)
            for d in range(duration):
                events.append(
                    DriftEvent(
                        id=idx,
                        time_step=t + d,
                        clients=clients,
                        duration=duration,
                        drift_dist=drift_dist if d == 0 else 0,
                    )
                )

    events.sort(key=lambda e: e.time_step)

    return deque(events)

# event_args = DriftEventArgs(
#             n_time_steps=30,
#             n_clients=20,
#             n_sudden=3,
#             n_gradual=3,
#             n_stage=1,
#             aff_client_ratio_range=(0.1, 0.15),
#             start=0.05,
#             end=0.8,
#             dist_range=(1, 3),
#             gradual_duration_ratio=0.15,
#             seed=42,
#         )
# es = generate_drift_events(event_args)
# for e in es:
#     print(e)