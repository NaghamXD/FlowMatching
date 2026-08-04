from __future__ import annotations

from pathlib import Path

from cvlab.features.manifest import FeatureManifest, get_code_version, manifest_hash


def _manifest(**overrides) -> FeatureManifest:
    fields = dict(
        dataset="dtd",
        split="train",
        encoder="resnet18",
        checkpoint_id="IMAGENET1K_V1",
        transform_description="Resize(256) CenterCrop(224) Normalize(...)",
        dtype="float32",
        shape=(1880, 512),
        code_version="deadbeef",
    )
    fields.update(overrides)
    return FeatureManifest(**fields)


def test_round_trips_through_dict() -> None:
    m = _manifest()
    assert FeatureManifest.from_dict(m.to_dict()) == m


def test_to_dict_shape_is_json_serializable_list() -> None:
    m = _manifest()
    d = m.to_dict()
    assert d["shape"] == [1880, 512]
    assert isinstance(d["shape"], list)


def test_equality_is_field_wise() -> None:
    assert _manifest() == _manifest()
    assert _manifest(shape=(1880, 384)) != _manifest(shape=(1880, 512))
    assert _manifest(transform_description="different") != _manifest()


def test_get_code_version_is_unknown_outside_own_repo(tmp_path: Path) -> None:
    # tmp_path is not a git repo at all (or not *this* project's repo), so this
    # must not accidentally report an unrelated enclosing repository's commit.
    assert get_code_version(repo_root=tmp_path) == "unknown"


def test_manifest_hash_is_deterministic_and_sensitive_to_every_field() -> None:
    base = _manifest()
    assert manifest_hash(base) == manifest_hash(_manifest())  # same content -> same hash
    assert manifest_hash(base) != manifest_hash(_manifest(shape=(1880, 384)))
    assert manifest_hash(base) != manifest_hash(_manifest(code_version="cafef00d"))
