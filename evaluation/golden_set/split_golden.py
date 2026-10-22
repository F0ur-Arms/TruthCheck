"""Split golden_v1.csv into 10 balanced CSV files in this directory."""

import csv
from pathlib import Path


CHUNK_COUNT = 10
SOURCE = Path(__file__).with_name("golden_v1.csv")


def main() -> None:
    with SOURCE.open(encoding="utf-8-sig", newline="") as source_file:
        rows = list(csv.reader(source_file))

    if not rows:
        raise ValueError(f"{SOURCE} is empty and has no CSV header")

    header, data = rows[0], rows[1:]
    chunk_size, remainder = divmod(len(data), CHUNK_COUNT)
    start = 0

    for index in range(CHUNK_COUNT):
        end = start + chunk_size + (index < remainder)
        output = SOURCE.with_name(f"golden_v1_chunk_{index + 1:02}.csv")
        with output.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(header)
            writer.writerows(data[start:end])
        start = end


if __name__ == "__main__":
    main()
