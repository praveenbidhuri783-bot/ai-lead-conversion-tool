"""Recommended agent pitch generation based on priority and objection."""

from __future__ import annotations

from typing import Dict, Optional

NO_MAJOR_OBJECTION = "No Major Objection"


def generate_pitch(
    priority: str,
    primary_objection: str,
    hard_excluded: bool,
    pitch_config: Dict,
    service_type: Optional[str] = None,
    language: Optional[str] = None,
) -> str:
    """Compose the recommended agent pitch for a lead.

    Excluded leads always receive the standard "no pitch" message. Otherwise,
    an objection-specific pitch takes precedence over the generic
    priority-based pitch, since it is more directly actionable for the agent.
    """
    if hard_excluded:
        return pitch_config.get("exclusion_pitch", "No further sales pitch recommended.")

    objection_pitches = pitch_config.get("pitch_by_objection", {})
    objection_pitch = objection_pitches.get(primary_objection, "").strip()

    if primary_objection != NO_MAJOR_OBJECTION and objection_pitch:
        pitch = objection_pitch
    else:
        pitch = pitch_config.get("pitch_by_priority", {}).get(priority, "")

    service_overrides = pitch_config.get("pitch_by_service", {})
    if service_type and service_overrides.get(service_type):
        pitch = f"{pitch} {service_overrides[service_type]}".strip()

    language_overrides = pitch_config.get("pitch_by_language", {})
    if language and language_overrides.get(language):
        pitch = f"{pitch} {language_overrides[language]}".strip()

    return pitch or "No further sales pitch recommended."
