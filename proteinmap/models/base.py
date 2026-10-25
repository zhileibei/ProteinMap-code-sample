"""Abstract interface for all protein localization predictors."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BasePredictor(ABC):
    """
    All predictors must implement this interface so the Evaluator can
    drive PUPS and ProtGPS interchangeably.

    Inputs follow the PUPS convention:
        esm2_embedding  : np.ndarray, shape (1, 2000, 1280) float32
        protein_length  : int
        landmark_stains : np.ndarray, shape (3, 128, 128) float32
                          channels: [nuclei, microtubule/mito, ER]
    """

    @abstractmethod
    def predict_compartment(
        self,
        esm2_embedding: np.ndarray,
        protein_length: int,
        landmark_stains: np.ndarray,
    ) -> dict:
        """
        Returns:
            {
                "proba":  np.ndarray shape (29,) float32  — sigmoid scores
                "binary": np.ndarray shape (29,) int32    — thresholded labels
            }
        """

    @abstractmethod
    def predict_inpainting(
        self,
        esm2_embedding: np.ndarray,
        protein_length: int,
        landmark_stains: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Returns:
            np.ndarray shape (128, 128) float32, or None if not supported.
        """

    @property
    @abstractmethod
    def supports_inpainting(self) -> bool:
        """True if the model produces antibody-stain reconstructions."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model identifier used in result tables."""
