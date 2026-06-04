"""
Franchised VCT team allowlist with alias support.

TEAM_ALIASES maps every known name variant (as it appears on VLR.gg) to a
canonical team name. Add new rows whenever a team rebrands or gets a new
sponsor prefix — the canonical name is what gets stored in the dataset.

Only matches where BOTH teams resolve to a canonical name are kept.
"""

# Format: "VLR.gg display name": "Canonical name"
TEAM_ALIASES: dict[str, str] = {

    # ── VCT Americas ────────────────────────────────────────────────────
    "Cloud9":                   "Cloud9",
    "Evil Geniuses":            "Evil Geniuses",
    "EG":                       "Evil Geniuses",
    "100 Thieves":              "100 Thieves",
    "Sentinels":                "Sentinels",
    "NRG":                      "NRG",
    "NRG Esports":              "NRG",
    "LOUD":                     "LOUD",
    "LEVIATÁN":                 "LEVIATÁN",
    "KRÜ Esports":              "KRÜ Esports",
    "G2 Esports":               "G2 Esports",
    "MIBR":                     "MIBR",
    "ENVY":                     "ENVY",
    "FURIA":                    "FURIA",

    # ── VCT EMEA ────────────────────────────────────────────────────────
    "Team Liquid":              "Team Liquid",
    "Fnatic":                   "Fnatic",
    "Natus Vincere":            "Natus Vincere",
    "NAVI":                     "Natus Vincere",
    "Team Vitality":            "Team Vitality",
    "Karmine Corp":             "Karmine Corp",
    "Gentle Mates":             "Gentle Mates",
    "BBL Esports":              "BBL Esports",
    "GIANTX":                   "GIANTX",
    "FUT Esports":              "FUT Esports",
    "Team Heretics":            "Team Heretics",
    "Eternal Fire":             "Eternal Fire",
    "PCIFIC Esports":           "PCIFIC Esports",

    # ── VCT Pacific ─────────────────────────────────────────────────────
    "T1":                       "T1",
    "Gen.G":                    "Gen.G",
    "DRX":                      "DRX",
    "Kiwoon DRX":               "DRX",
    "ZETA DIVISION":            "ZETA DIVISION",
    "Paper Rex":                "Paper Rex",
    "BOOM Esports":             "BOOM Esports",
    "Team Secret":              "Team Secret",
    "Global Esports":           "Global Esports",
    "Rex Regum Qeon":           "Rex Regum Qeon",
    "RRQ":                      "Rex Regum Qeon",
    "FULL SENSE":               "FULL SENSE",
    "Talon Esports":            "FULL SENSE",

    # ── VCT China ───────────────────────────────────────────────────────
    "EDward Gaming":            "EDward Gaming",
    "EDG":                      "EDward Gaming",
    "FunPlus Phoenix":          "FunPlus Phoenix",
    "FPX":                      "FunPlus Phoenix",
    "Bilibili Gaming":          "Bilibili Gaming",
    "BLG":                      "Bilibili Gaming",
    "NOVA Esports":             "NOVA Esports",
    "Wolves Esports":           "Wolves Esports",
    "Dragon Rangers Gaming":    "Dragon Rangers Gaming",
    "DRG":                      "Dragon Rangers Gaming",
    "Trace Esports":            "Trace Esports",
    "Attacking Soul Esports":   "Attacking Soul Esports",
    "All Gamers":               "All Gamers",
    "Titan Esports Club":       "Titan Esports Club",
    "TEC":                      "Titan Esports Club",
    "Xi Lai Gaming":            "Xi Lai Gaming",
}

# Derived set of all canonical names (used for quick membership checks)
CANONICAL_TEAMS: set[str] = set(TEAM_ALIASES.values())

# ── Masters London 2026 qualified teams ─────────────────────────────────────
# Fill these in manually — use canonical names from TEAM_ALIASES above
MASTERS_LONDON_TEAMS: set[str] = {
    # Americas
    "G2 Esports",
    "LEVIATÁN",
    "NRG",

    # EMEA
    "Team Heretics",
    "Team Vitality",
    "FUT Esports",

    # Pacific
    "Paper Rex",
    "FULL SENSE",
    "Global Esports",

    # China
    "EDward Gaming",
    "Xi Lai Gaming",
    "Dragon Ranger Gaming",
}


def is_masters_london_match(team_a: str, team_b: str) -> bool:
    """Return True only if both teams qualified for Masters London."""
    a = resolve_team(team_a) or team_a
    b = resolve_team(team_b) or team_b
    return a in MASTERS_LONDON_TEAMS and b in MASTERS_LONDON_TEAMS


def involves_masters_london_team(team_a: str, team_b: str) -> bool:
    """Return True if at least one team qualified for Masters London."""
    a = resolve_team(team_a) or team_a
    b = resolve_team(team_b) or team_b
    return a in MASTERS_LONDON_TEAMS or b in MASTERS_LONDON_TEAMS


def resolve_team(name: str) -> str | None:
    """
    Return the canonical team name for a given VLR.gg display name.
    Returns None if the team is not a franchised VCT team.
    """
    return TEAM_ALIASES.get(name)


def is_franchised_match(team_a: str, team_b: str) -> bool:
    """Return True only if both teams resolve to franchised VCT teams."""
    return resolve_team(team_a) is not None and resolve_team(team_b) is not None
