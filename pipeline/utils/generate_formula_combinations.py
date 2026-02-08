import re
from itertools import product
import csv
import os
from typing import List, Tuple, Dict

def calculate_theoretical_mass(formula: str) -> float:
    atomic_masses = {
        'H': 1.00782503223,
        'C': 12.0000000,
        'N': 14.00307400443,
        'O': 15.99491461957,
        'S': 31.9720711744
    }

    
    # Regex to match elements and their counts
    pattern = r'([CHOSN])(\d*)'
    matches = re.findall(pattern, formula)
    mass = 0.0
    for element, count in matches:
        if element not in atomic_masses:
            raise ValueError(f"Unknown element: {element}")
        count = int(count) if count else 1
        mass += atomic_masses[element] * count
    mass = mass - atomic_masses["H"]
    mass = round(mass, 4) + 0.0006
    return mass

def generate_formula_string(carbon_count: int, hydrogen_count: int, oxygen_count: int, 
                          nitrogen_count: int, sulfur_count: int) -> str:

    formula_parts = []
    
    if carbon_count > 0:
        formula_parts.append(f"C{carbon_count}" if carbon_count > 1 else "C")
    if hydrogen_count > 0:
        formula_parts.append(f"H{hydrogen_count}" if hydrogen_count > 1 else "H")
    if oxygen_count > 0:
        formula_parts.append(f"O{oxygen_count}" if oxygen_count > 1 else "O")
    if sulfur_count > 0:
        formula_parts.append(f"S{sulfur_count}" if sulfur_count > 1 else "S")
    if nitrogen_count > 0:
        formula_parts.append(f"N{nitrogen_count}" if nitrogen_count > 1 else "N")
    
    
    return "".join(formula_parts)

def save_combinations_to_csv_chunks(combinations: List[Tuple[str, float]], output_dir: str, chunk_size: int = 10):

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Group combinations by mass chunks
    chunked_combinations = {}
    
    for formula, mass in combinations:
        chunk_start = int(mass // chunk_size) * chunk_size
        chunk_end = chunk_start + chunk_size
        
        if chunk_start not in chunked_combinations:
            chunked_combinations[chunk_start] = []
        
        chunked_combinations[chunk_start].append((formula, mass))
    
    # Write each chunk to a separate CSV file
    total_files = 0
    total_formulas = 0
    
    for chunk_start in sorted(chunked_combinations.keys()):
        chunk_end = chunk_start + chunk_size
        chunk_data = chunked_combinations[chunk_start]
        
        if not chunk_data:
            continue
            
        # Sort by mass within the chunk
        chunk_data.sort(key=lambda x: x[1])
        
        filename = os.path.join(output_dir, f"formula_combinations_{chunk_start}-{chunk_end}.csv")
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Formula', 'Mass_Daltons'])
            for formula, mass in chunk_data:
                writer.writerow([formula, f"{mass:.4f}"])
        
        total_files += 1
        total_formulas += len(chunk_data)
        print(f"Saved {len(chunk_data)} formulas to {filename}")
    
    print(f"\nTotal files created: {total_files}")
    print(f"Total formulas saved: {total_formulas}")

def generate_all_formula_combinations(min_mass: float = 100.0, max_mass: float = 650.0) -> List[Tuple[str, float]]:
 

        # Define ranges according to requested constraints (inclusive bounds)
        carbon_range = range(4, 51)   # C4 to C50
        hydrogen_range = range(4, 101)  # H4 to H100
        oxygen_range = range(0, 26)   # O0 to O25
        nitrogen_range = range(0, 4)  # N0 to N3
        sulfur_range = range(0, 3)    # S0 to S2
        
        valid_combinations = []
        
        print("Generating formula combinations...")
        total_combinations = len(carbon_range) * len(hydrogen_range) * len(oxygen_range) * len(nitrogen_range) * len(sulfur_range)
        print(f"Total possible combinations: {total_combinations:,}")
        
        count = 0
        for carbon, hydrogen, oxygen, nitrogen, sulfur in product(carbon_range, hydrogen_range, oxygen_range, nitrogen_range, sulfur_range):
            count += 1
            if count % 100000 == 0:
                print(f"Processed {count:,} combinations...")
            # Apply ratio constraints early to reduce candidates
            # O/C must be > 0 and < 1 so oxygen must be >=1

            o_c_ratio = oxygen / carbon
            if not (0 < o_c_ratio < 1):
                continue

            h_c_ratio = hydrogen / carbon
            if not (0.3 < h_c_ratio < 2.5):
                continue

            # Calculate DBE and DBE - O constraint
            dbe = (2 * carbon + 2 - hydrogen + nitrogen) / 2.0
            dbe_minus_o = dbe - oxygen
            if not (-10 <= dbe_minus_o <= 10):
                continue

            # Generate formula string
            formula = generate_formula_string(carbon, hydrogen, oxygen, nitrogen, sulfur)
            
            try:
                mass = calculate_theoretical_mass(formula)
                
                # Check if mass is within range
                if min_mass <= mass <= max_mass:
                    valid_combinations.append((formula, mass))
                    
            except ValueError:
                # Skip invalid formulas
                continue
    
        print(f"Found {len(valid_combinations)} valid combinations within mass range {min_mass}-{max_mass} Daltons")
        
        # Sort by mass for easier analysis
        valid_combinations.sort(key=lambda x: x[1])
        
        return valid_combinations

def analyze_combinations(combinations: List[Tuple[str, float]]):
    """
    Analyze the generated combinations and print statistics.
    """
    if not combinations:
        print("No combinations found!")
        return
    
    masses = [mass for _, mass in combinations]
    formulas = [formula for formula, _ in combinations]
    
    print(f"Total combinations: {len(combinations)}")
    print(f"Mass range: {min(masses):.4f} - {max(masses):.4f} Daltons")
    print(f"Average mass: {sum(masses)/len(masses):.4f} Daltons")
    
    # Count elements
    element_counts = {'C': 0, 'H': 0, 'O': 0, 'N': 0, 'S': 0}
    for formula in formulas:
        for element in element_counts:
            if element in formula:
                element_counts[element] += 1
    
    print("\nElement presence:")
    for element, count in element_counts.items():
        percentage = (count / len(formulas)) * 100
        print(f"  {element}: {count} formulas ({percentage:.1f}%)")
    
    print(f"\nFirst 10 combinations:")
    for i, (formula, mass) in enumerate(combinations[:10]):
        print(f"  {i+1:2d}. {formula:15s} = {mass:.4f} Daltons")
    
    print(f"\nLast 10 combinations:")
    for i, (formula, mass) in enumerate(combinations[-10:]):
        print(f"  {len(combinations)-9+i:2d}. {formula:15s} = {mass:.4f} Daltons")
        
def generate_synthetic_data():
    """
    Main function to generate synthetic data and save to CSV files.
    """
    # Set output directory
    output_directory = "data/synthetic_data"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    
    # Generate all combinations
    combinations = generate_all_formula_combinations(100.0, 650.0)
    
    # Analyze results
    analyze_combinations(combinations)
    
    # Save to CSV files in chunks
    save_combinations_to_csv_chunks(combinations, output_directory, chunk_size=10)
    
    print("\nGeneration complete!")        