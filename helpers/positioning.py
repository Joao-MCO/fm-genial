from __future__ import annotations

import re


# Posições que realmente terão modelos.
#
# WB é agrupado com RB/LB conforme definido no experimento.
# M(R)/M(L) (meio-campo lateral clássico) é agrupado com RW/LW,
# pois LM/RM não é usado como posição própria no experimento.
CANONICAL_POSITIONS = [
    "GK",
    "LB",
    "CB",
    "RB",
    "DM",
    "CM",
    "LW",
    "AM",
    "RW",
    "ST",
]


def parse_position_groups(
    raw: str | None,
) -> list[tuple[str, str | None]]:
    """
    Converte a posição original do FM em:

        (linha, lado)

    Exemplos:

        D (R)
            -> [("D", "R")]

        D/WB (R)
            -> [("D", "R"), ("WB", "R")]

        D (RLC)
            -> [
                ("D", "R"),
                ("D", "L"),
                ("D", "C"),
            ]

        DM
            -> [("DM", None)]
    """

    if raw is None:
        return []

    raw = str(raw).strip()

    if not raw or raw in {
        "-",
        "Unknown",
        "nan",
        "None",
    }:
        return []

    result: list[
        tuple[str, str | None]
    ] = []

    for chunk in raw.split(","):

        chunk = chunk.strip()

        if not chunk:
            continue

        match = re.fullmatch(
            r"([A-Za-z/]+)\s*(?:\(([A-Za-z]+)\))?",
            chunk,
        )

        if not match:
            continue

        lines_raw, zones_raw = match.groups()

        lines = [
            line.upper()
            for line in lines_raw.split("/")
            if line
        ]

        zones = (
            list(zones_raw.upper())
            if zones_raw
            else [None]
        )

        for line in lines:
            for zone in zones:
                result.append(
                    (line, zone)
                )

    return result


def to_canonical_position(
    line: str,
    side: str | None,
) -> str | None:

    line = line.upper()
    side = (
        side.upper()
        if side
        else None
    )

    if line == "GK":
        return "GK"

    # Defensor e ala.
    #
    # WB(R) também é considerado RB.
    if line in {"D", "WB"}:

        return {
            "R": "RB",
            "L": "LB",
            "C": "CB",
        }.get(side)

    if line == "DM":
        return "DM"

    if line == "M":

        # M(R)/M(L) (meio-campo lateral clássico) é absorvido em
        # RW/LW: LM/RM não existe como posição própria no experimento.
        return {
            "R": "RW",
            "L": "LW",
            "C": "CM",
            None: "CM",
        }[side]

    if line == "AM":

        return {
            "R": "RW",
            "L": "LW",
            "C": "AM",
            None: "AM",
        }[side]

    if line == "ST":
        return "ST"

    return None


def get_canonical_positions(
    raw: str | None,
) -> list[str]:

    positions: list[str] = []

    for line, side in parse_position_groups(raw):

        position = to_canonical_position(
            line,
            side,
        )

        if (
            position
            and position not in positions
        ):
            positions.append(position)

    return positions


def get_primary_position(
    raw: str | None,
) -> str | None:

    positions = get_canonical_positions(
        raw
    )

    return (
        positions[0]
        if positions
        else None
    )


def get_secondary_positions(
    raw: str | None,
) -> list[str]:

    positions = get_canonical_positions(
        raw
    )

    return positions[1:]


def position_matches(
    raw: str | None,
    canonical_position: str,
) -> bool:

    return (
        canonical_position
        in get_canonical_positions(raw)
    )