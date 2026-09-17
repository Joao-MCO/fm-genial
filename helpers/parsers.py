from __future__ import annotations

import re


_MONEY_MULTIPLIERS = {
    "K": 1_000,
    "M": 1_000_000,
    "B": 1_000_000_000,
}


def _normalize_number(value: str) -> float:
    """
    Converte números nos formatos mais comuns:

        1234
        1,234
        8,25
        1.234,56
        1,234.56
    """

    value = value.strip().replace(" ", "")

    if not value:
        raise ValueError("Número vazio")

    if "," in value and "." in value:
        # Último separador é o decimal.
        if value.rfind(",") > value.rfind("."):
            value = value.replace(".", "").replace(",", ".")
        else:
            value = value.replace(",", "")

    elif "," in value:
        parts = value.split(",")

        # 8,25 -> decimal
        # 1,234 -> milhares
        if len(parts) == 2 and len(parts[1]) != 3:
            value = value.replace(",", ".")
        else:
            value = value.replace(",", "")

    elif value.count(".") > 1:
        value = value.replace(".", "")

    return float(value)


def _money_token_to_number(token: str) -> float:
    token = token.strip().upper()

    match = re.fullmatch(
        r"([\d.,]+)\s*([KMB])?",
        token,
    )

    if not match:
        raise ValueError(
            f"Valor monetário inválido: {token!r}"
        )

    number = _normalize_number(
        match.group(1)
    )

    multiplier = _MONEY_MULTIPLIERS.get(
        match.group(2) or "",
        1,
    )

    return number * multiplier


def parse_money_range(value) -> tuple[float, float]:
    """
    Converte valores como:

        8,25K
        8,25K - 10K
        R$ 1,5M - R$ 2M

    para:

        (8250, 8250)
        (8250, 10000)
        (1500000, 2000000)
    """

    if value is None:
        return (float("nan"), float("nan"))

    text = str(value).strip()

    if not text or text in {
        "-",
        "—",
        "nan",
        "NaN",
        "None",
    }:
        return (
            float("nan"),
            float("nan"),
        )

    text = re.sub(
        r"R\$\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    tokens = re.findall(
        r"[\d.,]+\s*[KMB]?",
        text,
        flags=re.IGNORECASE,
    )

    if not tokens:
        return (
            float("nan"),
            float("nan"),
        )

    try:
        numbers = [
            _money_token_to_number(token)
            for token in tokens
        ]
    except ValueError:
        return (
            float("nan"),
            float("nan"),
        )

    if len(numbers) == 1:
        return (
            numbers[0],
            numbers[0],
        )

    return (
        numbers[0],
        numbers[1],
    )


def parse_distance_km(value) -> float:
    if value is None:
        return float("nan")

    text = str(value).strip().lower()

    if not text or text in {
        "-",
        "nan",
        "none",
    }:
        return float("nan")

    match = re.search(
        r"([\d.,]+)",
        text,
    )

    if not match:
        return float("nan")

    try:
        return _normalize_number(
            match.group(1)
        )
    except ValueError:
        return float("nan")


def parse_height(value) -> float:
    """
    Converte:

        185 cm -> 185
        1.85 m -> 185
        6'2"   -> 187.96
    """

    if value is None:
        return float("nan")

    text = str(value).strip().lower()

    if not text or text in {
        "-",
        "nan",
        "none",
    }:
        return float("nan")

    meter_match = re.search(
        r"([\d.,]+)\s*m\b",
        text,
    )

    if meter_match:
        try:
            return (
                _normalize_number(
                    meter_match.group(1)
                )
                * 100
            )
        except ValueError:
            return float("nan")

    feet_match = re.search(
        r"(\d+)\s*(?:'|ft)\s*(\d+)?",
        text,
    )

    if feet_match:
        feet = int(feet_match.group(1))
        inches = int(
            feet_match.group(2) or 0
        )

        return (
            (feet * 12 + inches)
            * 2.54
        )

    cm_match = re.search(
        r"([\d.,]+)\s*cm\b",
        text,
    )

    if cm_match:
        try:
            return _normalize_number(
                cm_match.group(1)
            )
        except ValueError:
            return float("nan")

    try:
        number = _normalize_number(text)

        # Número acima de 100 provavelmente já está em cm.
        if number > 100:
            return number

        return number * 100

    except ValueError:
        return float("nan")


def parse_positions(value) -> dict[str, float]:
    """
    Gera features one-hot das posições originais.

    Exemplo:

        D/WB (R)

    gera:

        pos_D  = 1
        pos_WB = 1
        pos_zone_R = 1
    """

    lines = [
        "GK",
        "D",
        "WB",
        "DM",
        "M",
        "AM",
        "ST",
    ]

    zones = [
        "L",
        "C",
        "R",
    ]

    result = {
        f"pos_{line}": 0.0
        for line in lines
    }

    result.update({
        f"pos_zone_{zone}": 0.0
        for zone in zones
    })

    if value is None:
        return result

    text = str(value).upper()

    for line in lines:
        if re.search(
            rf"(?<![A-Z]){re.escape(line)}(?![A-Z])",
            text,
        ):
            result[f"pos_{line}"] = 1.0

    for zone in zones:
        if re.search(
            rf"\({zone}+\)",
            text,
        ):
            result[f"pos_zone_{zone}"] = 1.0

    return result


def parse_preferred_foot(value) -> dict[str, float]:
    result = {
        "foot_right": 0.0,
        "foot_left": 0.0,
        "foot_only": 0.0,
    }

    if value is None:
        return result

    text = str(value).strip().lower()

    if "right" in text:
        result["foot_right"] = 1.0

    elif "left" in text:
        result["foot_left"] = 1.0

    elif "only" in text:
        result["foot_only"] = 1.0

    return result


def parse_appearances(value) -> tuple[float, float]:
    """
    Retorna:

        (titularidades, entradas como substituto)
    """

    if value is None:
        return (
            float("nan"),
            float("nan"),
        )

    text = str(value).strip()

    if not text or text in {
        "-",
        "nan",
        "None",
    }:
        return (
            float("nan"),
            float("nan"),
        )

    numbers = re.findall(
        r"\d+(?:[.,]\d+)?",
        text,
    )

    if not numbers:
        return (
            float("nan"),
            float("nan"),
        )

    try:
        starts = _normalize_number(
            numbers[0]
        )

        subs = (
            _normalize_number(numbers[1])
            if len(numbers) > 1
            else 0.0
        )

        return (
            starts,
            subs,
        )

    except ValueError:
        return (
            float("nan"),
            float("nan"),
        )


def parse_percentage(value) -> float:
    if value is None:
        return float("nan")

    text = str(value).strip().replace(
        "%",
        "",
    )

    if not text or text in {
        "-",
        "nan",
        "None",
    }:
        return float("nan")

    try:
        return _normalize_number(text)

    except ValueError:
        return float("nan")