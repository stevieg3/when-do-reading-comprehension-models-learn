"""
Script for generating .job files. Adapted version of script written by Max Bartolo.
"""

SEEDS = [28, 29, 30]
SAVE_STEPS_SCHEDULE = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 36, 44, 52, 60, 68, 76, 92, 108, 124, 140, 156, 172, 188, 204, 220, 236, 252, 268, 284, 300, 316, 332, 348, 364, 380, 396, 428, 460, 492, 524, 556, 588, 620, 652, 684, 716, 748, 780, 812, 844, 876, 908, 940, 972, 1004, 1036, 1100, 1164, 1228, 1292, 1356, 1420, 1484, 1548, 1612, 1676, 1804, 1932, 2060, 2188, 2316, 2444, 2572, 2700, 2828, 2956, 3084, 3212, 3340, 3468, 3596, 3724, 3852, 3980, 4108, 4236, 4364, 4492, 4620, 4748, 4876, 5004, 5132, 5260, 5388, 5516, 5644, 5772, 5900, 6028, 6156, 6284, 6412, 6540, 6668, 6796, 6924, 7052, 7180, 7308, 7436, 7564, 7692, 7820, 7948]

# Generate the commands
flatten_commands = []
checklist_commands = []
extract_commands = []

for seed in SEEDS:
    for checkpoint in SAVE_STEPS_SCHEDULE:
        flatten_commands.append(
            f"python src/analysis/checklist/flatten_squad_predictions.py '/SAN/intelsys/rclearn/when-do-reading-comprehension-models-learn/predictions/checklist/albert-xlarge-v2-squadv1-wu=100-lr=3e5-bs=32-msl=384-seed={seed}/checkpoint-{checkpoint}/predict_predictions.json'"
        )
        checklist_commands.append(
            f"python src/analysis/checklist/run_squad.py '/SAN/intelsys/rclearn/when-do-reading-comprehension-models-learn/predictions/checklist/albert-xlarge-v2-squadv1-wu=100-lr=3e5-bs=32-msl=384-seed={seed}/checkpoint-{checkpoint}/predict_predictions_flat.txt' > '/SAN/intelsys/rclearn/when-do-reading-comprehension-models-learn/predictions/checklist/albert-xlarge-v2-squadv1-wu=100-lr=3e5-bs=32-msl=384-seed={seed}/checkpoint-{checkpoint}/checklist_results.txt'",
        )
        extract_commands.append(
            f"python src/analysis/checklist/extract_results_from_checklist_summary.py '/SAN/intelsys/rclearn/when-do-reading-comprehension-models-learn/predictions/checklist/albert-xlarge-v2-squadv1-wu=100-lr=3e5-bs=32-msl=384-seed={seed}/checkpoint-{checkpoint}/checklist_results.txt'"
        )


if __name__ == '__main__':

    job_file = ''
    for cmd in flatten_commands:
        job_file += cmd
        job_file += '\n'
    job_file += 'sleep 10\n'
    job_file += "printf 'Finished flattening'\n"

    for cmd in checklist_commands:
        job_file += cmd
        job_file += '\n'
    job_file += 'sleep 10\n'
    job_file += "printf 'Finished CheckListing'\n"

    for cmd in extract_commands:
        job_file += cmd
        job_file += '\n'
    job_file += "printf 'Finished extracting'\n"

    # Save the job file
    with open(f'jobs/checklist/processing.sh', 'w') as f:
        f.write(job_file)
