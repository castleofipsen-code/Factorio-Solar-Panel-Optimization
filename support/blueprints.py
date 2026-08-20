import numpy as np
import json
import base64
import zlib


def factorio_blueprint_to_coordinates(blueprint_string):
    """Decode a Factorio blueprint into the five building coordinate arrays."""
    blueprint_string = blueprint_string.strip()
    if not blueprint_string.startswith("0"):
        raise ValueError("A Factorio blueprint string must start with 0.")

    compressed = base64.b64decode(blueprint_string[1:])
    blueprint_data = json.loads(zlib.decompress(compressed))
    entities = blueprint_data["blueprint"].get("entities", [])

    entity_offsets = {
        "solar-panel": (0, 1.5),
        "accumulator": (1, 1.0),
        "substation": (2, 1.0),
        "roboport": (3, 2.0),
        "medium-electric-pole": (4, 0.5),
    }
    coordinates = [[] for _ in range(5)]

    for entity in entities:
        entity_name = entity.get("name")
        if entity_name not in entity_offsets:
            continue

        layer, center_offset = entity_offsets[entity_name]
        position = entity["position"]
        coordinates[layer].append(
            (
                round(position["x"] - center_offset),
                round(position["y"] - center_offset),
            )
        )

    return tuple(
        np.asarray(layer, dtype=int).reshape(-1, 2)
        for layer in coordinates
    )


def coordinates_to_factorio_blueprint(coordinates, label="my_blueprint", substation_range=18):
    coordinates_solar = coordinates[0] + np.array([1.5, 1.5])
    coordinates_accumulators = coordinates[1] + np.array([1, 1])
    coordinates_substations = coordinates[2] + np.array([1, 1])
    coordinates_roboports = coordinates[3] + np.array([2, 2])
    coordinates_medium_poles = coordinates[4] + np.array([0.5, 0.5])

    entities = []
    substations = []
    entity_number = 1

    for x, y in coordinates_solar:
        entities.append({
            "entity_number": entity_number,
            "name": "solar-panel",
            "position": {"x": float(x), "y": float(y)}
        })
        entity_number += 1

    for x, y in coordinates_accumulators:
        entities.append({
            "entity_number": entity_number,
            "name": "accumulator",
            "position": {"x": float(x), "y": float(y)}
        })
        entity_number += 1

    for x, y in coordinates_roboports:
        entities.append({
            "entity_number": entity_number,
            "name": "roboport",
            "position": {"x": float(x), "y": float(y)},
            "request_filters": {"sections": [{"index": 1}]}
        })
        entity_number += 1

    for x, y in coordinates_substations:
        entities.append({
            "entity_number": entity_number,
            "name": "substation",
            "position": {"x": float(x), "y": float(y)}
        })

        substations.append((entity_number, float(x), float(y)))
        entity_number += 1

        medium_poles = []

    for x, y in coordinates_medium_poles:
        entities.append({
            "entity_number": entity_number,
            "name": "medium-electric-pole",
            "position": {"x": float(x), "y": float(y)}
        })

        medium_poles.append((entity_number, float(x), float(y)))
        entity_number += 1

    wires = []

    # Substation ↔ Substation
    for i in range(len(substations)):
        n1, x1, y1 = substations[i]

        for j in range(i + 1, len(substations)):
            n2, x2, y2 = substations[j]

            distance = np.hypot(x1 - x2, y1 - y2)

            if distance <= substation_range:
                wires.append([n1, 5, n2, 5])

    # Substation ↔ Medium Pole
    for n1, x1, y1 in substations:
        for n2, x2, y2 in medium_poles:

            distance = np.hypot(x1 - x2, y1 - y2)

            if distance <= 9:
                wires.append([n1, 5, n2, 5])

    # Medium Pole ↔ Medium Pole
    for i in range(len(medium_poles)):
        n1, x1, y1 = medium_poles[i]

        for j in range(i + 1, len(medium_poles)):
            n2, x2, y2 = medium_poles[j]

            distance = np.hypot(x1 - x2, y1 - y2)

            if distance <= 9:
                wires.append([n1, 5, n2, 5])

    blueprint = {
        "blueprint": {
            "icons": [
                {"signal": {"name": "solar-panel"}, "index": 1},
                {"signal": {"name": "accumulator"}, "index": 2},
                {"signal": {"name": "roboport"}, "index": 3},
                {"signal": {"name": "substation"}, "index": 4}
            ],
            "entities": entities,
            "wires": wires,
            "item": "blueprint",
            "label": label,
            "version": 562949958402048
        }
    }

    json_string = json.dumps(blueprint, separators=(",", ":"))
    compressed = zlib.compress(json_string.encode("utf-8"))
    encoded = base64.b64encode(compressed).decode("utf-8")

    return "0" + encoded
