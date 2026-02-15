"""Domain-level dataset reader backed by precomputed metadata."""

import json
from collections.abc import Callable
from pathlib import Path
import random
from PIL import Image
import io
import torch
from torchvision import transforms
from driftguard_ray import config

logger = config.get_logger("data.domain_dataset")

class DomainDataset:
    """Serve samples by domain using precomputed metadata.

    Attributes:
        domains: Ordered list of available domains.
    """
    # static method to convert PIL Image to tensor
    @staticmethod
    def toTensor(b_image: bytes) -> torch.Tensor:
        pil_image = Image.open(io.BytesIO(b_image))
        tensor_transform = transforms.ToTensor()
        return tensor_transform(pil_image)
    
    # static method to convert bytes to PIL Image
    @staticmethod
    def toImage(b_image: bytes) -> Image.Image:
        return Image.open(io.BytesIO(b_image))

    def __init__(
        self,
        meta_path: Path | str,
        transform: Callable[[bytes], bytes] | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize the dataset with metadata and buffering.

        Args:
            meta_path: Path to `datasets/<name>/_meta.json`.
            transform: Optional transform applied to raw bytes.
        """
        self.meta_path = Path(meta_path)
        self.dataset_root = self.meta_path.parent
        self.transform = transform

        self.rng = random.Random(seed)
        meta = self._load_meta(self.meta_path)

        self.domains: list[str] = list(meta["domains"])
        """Ordered list of available domains."""

        domain_to_files: dict[str, list[tuple[Path, int]]] = {
            domain: [
                (self.dataset_root / rel_path, label_idx)
                for rel_path, label_idx in entries
            ]
            for domain, entries in meta["domain_to_files"].items()
        } # Mapping of domain -> list of (path, label idx).

        for entries in domain_to_files.values():
            # Shuffle entries for randomness
            self.rng.shuffle(entries)

        self.buffer: dict[str, list[tuple[Path, int]]] \
            = {domain: list(entries) for domain, entries in domain_to_files.items()}
        """Mapping of domain -> list of (path, label idx)."""
        logger.info(f"DomainDataset buffer sizes: " +
            ", ".join(
                f"{domain}: {len(entries)}"
                for domain, entries in self.buffer.items()
            )
        )
    # sample number of samples based on domain probabilities
    def get(self, n: int, distribution: list[float]) -> list[tuple[bytes, int]]:
        assert len(distribution) == len(self.domains), (
            "Distribution length must match number of domains"
        )
        samples = self.rng.choices(
            range(len(distribution)), weights=distribution, k=n
        )

        return [self.get_one(d) for d in samples]

    def get_one(self, domain_idx: int) -> tuple[bytes, int]:
        """Return a single sample for the requested domain index.

        Args:
            domain_idx: Index of the domain to sample from.

        Returns:
            Tuple of (raw bytes, label_idx) or None if unavailable.
        """
        if domain_idx >= len(self.domains):
            raise ValueError(f"Unknown domain: {domain_idx}")
        domain = self.domains[domain_idx]
        entries = self.buffer[domain]
        if not entries:
            raise ValueError(f"No samples available for domain: {domain}")
        
        path, label_idx = self.rng.choice(entries)  # 随机采样 with replacement <-------
        b_image = path.read_bytes()
        if self.transform:
            b_image = self.transform(b_image)
        return b_image, label_idx

    @staticmethod
    def _load_meta(meta_path: Path) -> dict:
        """Load metadata content from disk.

        Args:
            meta_path: Path to `_meta.json`.

        Returns:
            Parsed metadata dictionary.
        """
        return json.loads(meta_path.read_text())
