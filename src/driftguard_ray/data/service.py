"""Data service for domain-sampled data."""

from collections import deque
from dataclasses import dataclass
import random
from pathlib import Path
from typing import List, Tuple

from driftguard_ray.data.drift_simulation import (
    ClientState,
    DriftEvent,
    DriftEventArgs,
    generate_drift_events,
)
from driftguard_ray.data.domain_dataset import DomainDataset
from driftguard_ray import config

logger = config.get_logger("data.service")


@dataclass
class DataServiceArgs:
    meta_path: Path | str
    num_clients: int
    sample_size: int
    drift_event_args: DriftEventArgs
    seed: int | None


class DataService:
    """Data service for sampling data by domain.

    Attributes:
        dataset: DomainDataset instance.
        client_states: ClientState instance.
        sample_size: Number of samples per client request.
    """

    def __init__(
        self,
        args: DataServiceArgs,
    ) -> None:
        """Initialize the data service with dataset and drift events.
        Args:
            args: DataServiceArgs instance.
        """
        self.dataset = DomainDataset(args.meta_path, seed=args.seed)
        self.client_states: ClientState = ClientState(
            num_domains=len(self.dataset.domains),
            num_clients=args.num_clients,
            seed=args.seed,
        )
        self.sample_size: int = args.sample_size

        self._time_step: int = 1
        self._events: deque[DriftEvent] = generate_drift_events(
            args=args.drift_event_args
        )
        self._stopped: bool = False
        self._rng: random.Random = random.Random(args.seed)

        logger.info(f"Data service starting with {args.num_clients} clients...")
        logger.debug(f"Dataset domains: {self.dataset.domains}")

    def get_data(self, args: Tuple[int, int]) -> List[Tuple[bytes, int]]:
        """Get a batch of data samples for a client at a given time step.

        Args:
            args: Tuple containing (client_id, time_step).
        Returns:
            List of samples: List[Tuple[bytes, int]]
        """
        if self._stopped:
            raise RuntimeError("Data service is stopped.")

        # args: (cid, time_step) in this minimal RPC contract.
        cid, time_step = args
        try:
            if self._time_step != time_step:
                # Advance all client states to the new time step.
                self._time_step = time_step
                while self._events and self._events[0].time_step <= time_step:
                    event = self._events.popleft()
                    self.client_states.update(event)

            samples = self.dataset.get(
                self.sample_size, self.client_states.get_distribution(cid)
            )
            # logger.info(
            #     "[get_data] cid: %s\ttime_step: %s\tdistribution: %s",
            #     cid,
            #     time_step,
            #     self.client_states.get_distribution(cid),
            # )
            return samples
        except Exception:
            logger.exception(
                "get_data failed (cid=%s, time_step=%s).", cid, time_step
            )
            raise

    def stop(self) -> bool:
        """Stop the service.

        Returns:
            True if shutdown was triggered.
        """
        logger.info("Stopping data service...")
        self._stopped = True
        return True
