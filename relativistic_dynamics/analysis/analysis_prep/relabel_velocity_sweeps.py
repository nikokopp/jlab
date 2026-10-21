from pathlib import Path
import re
import csv
import shutil

data_dir = Path.home() / "jlab/relativistic_dynamics/data/"

raw_dir = data_dir / "raw/velocity_sweep"
processed_dir = data_dir / "processed/velocity_sweep"

TEST_RUN = False

# use original B label as key
# values: new B label, assigned B (G), current (A), status as usable or not

old_runs = {
    "2026-10-07": {
        "100": (None, None, None, "unused")
    },
    "2026-10-05": {
        "090": (None, None, None, "unused"),
        "100": (None, None, None, "unused")
    },
    "2026-09-30": {
        "099": (None, None, None, "unused")
    },
    "2026-10-09": {
        "080": ("089p4", 89.4, 3.85, "usable"),
        "090": ("091p9", 91.9, 4.00, "usable")
    },
    "2026-10-13": {
        "110": (None, None, None, "unchanged"),
        "120": (None, None, None, "unchanged")
    },
    "2026-10-14": {
        "080": (None, None, None, "unchanged"),
        "100": (None, None, None, "unchanged")
    }
}

pattern = re.compile(
    r"(?P<date>\d{8})_"
    r"(?P<prefix>vsweep_)?"
    r"B(?P<B>\d{3})G_"
    r"(?P<rest>Vsel.+)\.Spe$"
)

changes = []

for date, assignments in old_runs.items():
    date_dir = raw_dir / date

    if not date_dir.is_dir():
        raise FileNotFoundError(date_dir)

    for source in sorted(date_dir.glob("*.Spe")):
        match = pattern.fullmatch(source.name)

        if match is None:
            print(f"Skipping unrecognized filename: {source.name}")
            continue

        if match["date"] != date.replace("-", ""):
            raise ValueError(f"Date mismatch: {source.name}")

        old_B = match["B"]

        if old_B not in assignments:
            continue

        new_B, field, current, status = assignments[old_B]

        if status == "unused":
            destination = processed_dir / "unused" / source.name

        elif status == "unchanged":
            destination = processed_dir / source.name

        else:
            new_name = (
                f"{match['date']}_"
                f"{match['prefix'] or ''}"
                f"B{new_B}G_"
                f"{match['rest']}.Spe"
            )
            destination = processed_dir / new_name

        print(f"{source.name}")
        print(f"  -> {destination.relative_to(data_dir)}")

        changes.append({
            "original_path": str(source.relative_to(data_dir)),
            "processed_path": str(destination.relative_to(data_dir)),
            "original_B_label": old_B,
            "assigned_B_G": field,
            "current_A": current,
            "status": status,
        })

        if not TEST_RUN:
            if destination.exists():
                raise FileExistsError(destination)

            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

if not TEST_RUN and changes:
    changes_path = processed_dir / "relabel_changes.csv"
    with changes_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=changes[0].keys()
        )
        writer.writeheader()
        writer.writerows(changes)

print(f"\nProcessed {len(changes)} files.")
print("Preview only." if TEST_RUN else "Copies created.")

