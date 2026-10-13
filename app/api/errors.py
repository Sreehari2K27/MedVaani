"""Shared Prisma/Timbre error to HTTP status mapping for voice endpoints."""

from __future__ import annotations

from app.services.prisma import (
    PrismaAPIError,
    PrismaConfigurationError,
    PrismaFileError,
    PrismaNetworkError,
    PrismaValidationError,
)
from app.services.timbre import (
    TimbreAPIError,
    TimbreConfigurationError,
    TimbreNetworkError,
    TimbreValidationError,
)

VALIDATION_ERRORS = (
    PrismaFileError,
    PrismaValidationError,
    TimbreValidationError,
)
CONFIGURATION_ERRORS = (
    PrismaConfigurationError,
    TimbreConfigurationError,
)
UPSTREAM_ERRORS = (
    PrismaNetworkError,
    PrismaAPIError,
    TimbreNetworkError,
    TimbreAPIError,
)

__all__ = [
    "VALIDATION_ERRORS",
    "CONFIGURATION_ERRORS",
    "UPSTREAM_ERRORS",
]