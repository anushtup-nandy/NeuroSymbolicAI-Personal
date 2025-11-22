"""Ingestion modules for multi-source data processing."""

from modules.ingestion.base import BaseIngestor
from modules.ingestion.obsidian import ObsidianIngestor
from modules.ingestion.google_takeout import GoogleTakeoutIngestor

__all__ = ['BaseIngestor', 'ObsidianIngestor', 'GoogleTakeoutIngestor']
