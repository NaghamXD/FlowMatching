"""Feature extraction and caching, with a manifest that invalidates on any
(dataset, split, encoder, transform, code version) mismatch."""

from cvlab.features.cache import load, load_manifest
from cvlab.features.extract import extract_split
from cvlab.features.manifest import FeatureManifest, manifest_hash

__all__ = ["load", "load_manifest", "extract_split", "FeatureManifest", "manifest_hash"]
