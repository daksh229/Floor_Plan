"""Versioned prompt store. Prompts are data, not code (proposal §3, §7).

Adding a new variable or tuning instructions is a dict edit, not a code deploy.
In a production build this would be backed by Postgres + an admin UI; here we
keep it inline so the structure is visible during the interview demo.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class VariableSpec:
    key: str
    display_name: str
    unit: str
    instruction: str


@dataclass(frozen=True)
class PromptVersion:
    version: str
    project_type: str
    system: str
    variables: List[VariableSpec]

    def render_user_text(self, page_index: int, page_w: int, page_h: int) -> str:
        var_lines = "\n".join(
            f"  - {v.key} ({v.display_name}, unit={v.unit}): {v.instruction}"
            for v in self.variables
        )
        return (
            f"You are looking at page {page_index} of a residential construction plan set.\n"
            f"The rendered image is {page_w} x {page_h} pixels.\n\n"
            f"Extract the following variables. For each variable, return exactly one entry:\n"
            f"{var_lines}\n\n"
            f"Respond ONLY with a JSON object matching this schema:\n"
            "{\n"
            '  "fields": [\n'
            "    {\n"
            '      "variable": "<key from list above>",\n'
            '      "value": <number | integer | string | null>,\n'
            '      "unit": "<unit string or null>",\n'
            '      "detection_state": "detected_actual" | "detected_placeholder" | "not_detected",\n'
            '      "overlay_coordinates": [\n'
            '        { "x": <px>, "y": <px>, "w": <px>, "h": <px>, "label": "<optional>" }\n'
            "      ],\n"
            '      "confidence_note": "<short reason if placeholder, else null>"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Detection state rules (strict):\n"
            "- detected_actual: you can read the value directly from the page AND can place a tight bounding box on it.\n"
            "- detected_placeholder: you can read or infer the value but the bounding box is approximate "
            "(e.g. centered on the floor-plan region rather than tightly on a number). Still return a box.\n"
            "- not_detected: the variable cannot be determined from this page. overlay_coordinates must be [].\n\n"
            "Coordinates are in image pixel space with origin at top-left. Do not normalise.\n"
            "Do not include any prose outside the JSON object."
        )


_GROUND_FLOOR_V1 = PromptVersion(
    version="ground_floor_v1",
    project_type="new_build_ground_floor",
    system=(
        "You are a quantity-surveying assistant reading Australian residential construction plans. "
        "You extract take-off quantities with strict JSON output and honest confidence labelling. "
        "You never guess: if a value is not visible on this page, mark it not_detected."
    ),
    variables=[
        VariableSpec(
            key="ground_floor_area_m2",
            display_name="Ground Floor Area",
            unit="m2",
            instruction=(
                "Total ground-floor internal floor area in square metres. Prefer a value printed on the "
                "plan (e.g. 'Ground Floor 142.5 m2' in a title block or area schedule). If only the "
                "overall floor plan is visible without a printed total, mark detected_placeholder and "
                "place the bounding box on the floor plan region."
            ),
        ),
        VariableSpec(
            key="room_count",
            display_name="Room Count",
            unit="rooms",
            instruction=(
                "Number of named rooms on the ground floor plan (bedrooms, living, kitchen, bathroom, "
                "laundry, etc.). Return one bounding box per room label you used, with the room name "
                "in the box label."
            ),
        ),
        VariableSpec(
            key="door_openings",
            display_name="Door Openings",
            unit="count",
            instruction=(
                "Count of door openings visible on the ground floor plan (internal + external door "
                "swings). Return one bounding box per detected door swing. If too many to box "
                "individually, return the total in value and mark detected_placeholder."
            ),
        ),
    ],
)


PROMPT_REGISTRY: Dict[str, PromptVersion] = {
    _GROUND_FLOOR_V1.version: _GROUND_FLOOR_V1,
}

DEFAULT_VERSION = _GROUND_FLOOR_V1.version


def get_prompt(version: str = DEFAULT_VERSION) -> PromptVersion:
    if version not in PROMPT_REGISTRY:
        raise KeyError(f"Unknown prompt version: {version}. Available: {list(PROMPT_REGISTRY)}")
    return PROMPT_REGISTRY[version]
