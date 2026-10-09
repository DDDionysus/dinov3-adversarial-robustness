import csv
from pathlib import Path


def save_csv(results, path):
    """
    Save a list of result dicts to a CSV file.
    """

    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())

        writer.writeheader()
        writer.writerows(results)
