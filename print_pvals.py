import re

# Example: load your log text from a file
with open("psrj2229_allfiles.log", "r") as f:
    log_text = f.read()

# Find each run section
run_blocks = re.split(r"INFO:root:Start analysis of run:", log_text)

for block in run_blocks[1:]:  # Skip first split (text before first run)
    # Get run name
    run_name_match = re.search(r"\s*(\d+)", block)
    if run_name_match:
        run_name = run_name_match.group(1)
        print(f"Run: {run_name}")
        
        # Find all gumball_p_array lists
        arrays = re.findall(r"gumball_p_array: \[(.*?)\]", block, re.DOTALL)
        
        for arr in arrays:
            # Extract float values
            values = re.findall(r"np\.float64\((.*?)\)", arr)
            print(values)
        print()
