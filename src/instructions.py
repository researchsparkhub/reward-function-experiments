"""Natural-language instructions for E2/E3 (the goal is described in
language instead of being handed to the reward as coordinates)."""

TEMPLATES = [
    "Go to the sofa. Avoid bumping into walls and take the shortest path you can.",
    "Move toward the sofa as directly as possible without hitting obstacles.",
    "Reach the sofa, staying away from the chair and the cube on the way.",
    "Navigate to the sofa efficiently; try not to waste moves walking into walls.",
]


def instruction_for(seed: int) -> str:
    return TEMPLATES[seed % len(TEMPLATES)]
