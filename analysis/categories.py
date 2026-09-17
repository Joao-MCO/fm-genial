from __future__ import annotations


ATTRIBUTE_CATEGORIES: dict[str, str] = {
    # Technical
    "Corners": "Technical",
    "Crossing": "Technical",
    "Dribbling": "Technical",
    "Finishing": "Technical",
    "First Touch": "Technical",
    "Free Kick Taking": "Technical",
    "Heading": "Technical",
    "Long Shots": "Technical",
    "Long Throws": "Technical",
    "Marking": "Technical",
    "Passing": "Technical",
    "Penalty Taking": "Technical",
    "Tackling": "Technical",
    "Technique": "Technical",

    # Mental
    "Aggression": "Mental",
    "Anticipation": "Mental",
    "Bravery": "Mental",
    "Composure": "Mental",
    "Concentration": "Mental",
    "Decisions": "Mental",
    "Determination": "Mental",
    "Flair": "Mental",
    "Leadership": "Mental",
    "Off The Ball": "Mental",
    "Positioning": "Mental",
    "Team Work": "Mental",
    "Vision": "Mental",
    "Work Rate": "Mental",

    # Physical
    "Acceleration": "Physical",
    "Agility": "Physical",
    "Balance": "Physical",
    "Jumping Reach": "Physical",
    "Natural Fitness": "Physical",
    "Pace": "Physical",
    "Stamina": "Physical",
    "Strength": "Physical",
}


STAT_CATEGORIES: dict[str, str] = {
    # Passing / creation
    "Assists": "Creation",
    "Key Passes": "Creation",
    "Chances Created": "Creation",
    "Clear Cut Chances Created": "Creation",
    "Open Play Key Passes": "Creation",
    "xA": "Creation",

    # Crossing
    "Crosses Attempted": "Crossing",
    "Crosses Completed": "Crossing",
    "Crosses Completed Ratio": "Crossing",
    "Open Play Crosses Attempted": "Crossing",
    "Open Play Crosses Completed": "Crossing",
    "Open Play Cross Completion Percentage": "Crossing",

    # Passing
    "Passes Attempted": "Passing",
    "Passes Completed": "Passing",
    "Pass Completion Percentage": "Passing",
    "PsP": "Passing",

    # Shooting
    "Goals": "Shooting",
    "Goals/90": "Shooting",
    "xG": "Shooting",
    "xG/90": "Shooting",
    "NP-xG/90": "Shooting",
    "xG-OP": "Shooting",
    "xG-OP/90": "Shooting",
    "Shots": "Shooting",
    "Shots/90": "Shooting",
    "Shots on Target": "Shooting",
    "Shots on Target/90": "Shooting",
    "Shots on Target Percentage": "Shooting",
    "xG/shot": "Shooting",
    "xG/shot/90": "Shooting",
    "Conv %": "Shooting",
    "Goals From Outside The Box": "Shooting",
    "Goals From Outside The Box/90": "Shooting",
    "Shots From Outside The Box Per 90 minutes": "Shooting",
    "Free Kick Shots": "Shooting",
    "Free Kick Shots/90": "Shooting",
    "Penalties Taken": "Shooting",
    "Penalties Taken/90": "Shooting",
    "Penalties Scored Ratio": "Shooting",

    # Dribbling / ball progression
    "Dribbles": "Dribbling",
    "Dribbles/90": "Dribbling",
    "Sprints": "Dribbling",
    "Sprints/90": "Dribbling",

    # Defensive
    "Tackles Attempted": "Defending",
    "Tackled Completed": "Defending",
    "Tackled Completed/90": "Defending",
    "Tackle Completion Percentage": "Defending",
    "Key Tackles": "Defending",
    "Key Tackles/90": "Defending",
    "Interceptions": "Defending",
    "Clearances": "Defending",
    "Clearances/90": "Defending",
    "Blk": "Defending",
    "Blk/90": "Defending",
    "Shts Blckd": "Defending",
    "Shts Blckd/90": "Defending",
    "Mistakes Leading to Goals": "Defending",
    "Mistakes Leading to Goals/90": "Defending",

    # Aerial
    "Headers Attempted": "Aerial",
    "Headers Attempted/90": "Aerial",
    "Headers Won": "Aerial",
    "Headers Won/90": "Aerial",
    "Headers Won Percentage": "Aerial",
    "Key Headers per 90": "Aerial",
    "Key Headers per 90/90": "Aerial",

    # Possession
    "Possession Won": "Possession",
    "Possession Won per 90": "Possession",
    "Possession Lost": "Possession",
    "Possession Lost per 90": "Possession",

    # Discipline
    "Fouls Made": "Discipline",
    "Fouls Made/90": "Discipline",
    "Fouls Against": "Discipline",
    "Fouls Against/90": "Discipline",
    "Yellow Cards": "Discipline",
    "Yellow Cards/90": "Discipline",
    "Red cards": "Discipline",
    "Red cards/90": "Discipline",
    "Off": "Discipline",
    "Off/90": "Discipline",
    "Pres A": "Discipline",
    "Pres C": "Discipline",

    # Playing time / volume
    "Starts": "Playing Time",
    "Starts/90": "Playing Time",
    "Minutes": "Playing Time",

    # Physical activity
    "Distance": "Physical Activity",
    "Distance/90": "Physical Activity",
}


def get_attribute_category(attribute: str) -> str:
    return ATTRIBUTE_CATEGORIES.get(attribute, "Other")


def get_stat_category(stat: str) -> str:
    base = stat.removesuffix("/90")
    return STAT_CATEGORIES.get(stat, STAT_CATEGORIES.get(base, "Other"))