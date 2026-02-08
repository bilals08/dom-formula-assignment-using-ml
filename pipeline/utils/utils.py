import os
import re

def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def calculate_theoretical_mass(formula: str) -> float:
    atomic_masses = {
        'H': 1.00782503223,
        'C': 12.0000000,
        'N': 14.00307400443,
        'O': 15.99491461957,
        'S': 31.9720711744
    }
    pattern = r'([CHOSN])(\d*)'
    matches = re.findall(pattern, formula)
    mass = 0.0
    for element, count in matches:
        if element not in atomic_masses:
            raise ValueError(f"Unknown element: {element}")
        count = int(count) if count else 1
        mass += atomic_masses[element] * count
    mass = mass - atomic_masses["H"]
    mass = round(mass, 4) + 0.000549
    return mass