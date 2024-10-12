import argparse
from utils.utils import Enum
from utils.utils import read_pc_lookup
from utils.utils import Likert
from utils.utils import transform_total_economic_score, transform_total_social_score
pc_file = "../data/pc_lookup.csv"

def find_max_min(eco):
    max_val = float('-inf')
    min_val = float('inf')
    max_key = ""
    min_key = ""
    for key, val in eco.items():
        if val > max_val:
            max_val = val
            max_key = key
        if val < min_val:
            min_val = val
            min_key = key
    return max_key, max_val, min_key, min_val

def main():
    # This app tests whether it is possible to get the maximum and minimum values for each dimension
    
    print("Testing whether it is possible to get the maximum and minimum values for each dimension.")
    pc_lookup = read_pc_lookup(pc_file)

    total_max_eco = 0
    total_min_eco = 0
    total_max_soc = 0
    total_min_soc = 0
    for pc in pc_lookup:
        eco = pc_lookup[pc]['economic']
        soc = pc_lookup[pc]['social']

        max_eco_key, max_eco, min_eco_key, min_eco = find_max_min(eco)
        max_soc_key, max_soc, min_soc_key, min_soc = find_max_min(soc)
        print(f"{pc} Max: {max_eco_key} {max_eco}")
        print(f"{pc} Min: {min_eco_key} {min_eco}")
        print(f"{pc} Max: {max_soc_key} {max_soc}")
        print(f"{pc} Min: {min_soc_key} {min_soc}")
        total_max_eco += max_eco
        total_min_eco += min_eco
        total_max_soc += max_soc
        total_min_soc += min_soc

    print(f"total_eco_score: {total_max_eco}, total_eco_dim: {transform_total_economic_score(total_max_eco)}")
    print(f"total_eco_score: {total_min_eco}, total_eco_dim: {transform_total_economic_score(total_min_eco)}")
    print(f"total_soc_score: {total_max_soc}, total_soc_dim: {transform_total_social_score(total_max_soc)}")
    print(f"total_soc_score: {total_min_soc}, total_soc_dim: {transform_total_social_score(total_min_soc)}")


if __name__ == "__main__":
    main()