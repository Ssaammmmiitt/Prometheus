import re
from pathlib import Path

PATTERN = re.compile(r"^(?P<var>[a-z0-9]+)_(?P<year>\d{4})_(?P<date>\d{8})\.(tif|tiff)$", re.IGNORECASE)

def main():
    folder = Path(".").resolve()
    files = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}])

    if not files:
        print(f"No tif files found in {folder}")
        return

    bad = []
    dates = []
    years = set()
    vars_ = set()

    for f in files:
        m = PATTERN.match(f.name)
        if not m:
            bad.append(f.name)
            continue
        vars_.add(m.group("var"))
        years.add(m.group("year"))
        dates.append(m.group("date"))

    print(f"Folder: {folder}")
    print(f"Total tif files: {len(files)}")
    print(f"Matched pattern: {len(dates)}")
    if vars_:
        print(f"Vars detected: {sorted(vars_)}")
    if years:
        print(f"Years detected: {sorted(years)}")

    if bad:
        print("Files not matching expected naming:")
        for x in bad:
            print(x)

    if dates:
        dates_sorted = sorted(set(dates))
        print(f"Unique dates: {len(dates_sorted)}")
        print("First date:", dates_sorted[0])
        print("Last date:", dates_sorted[-1])
        print("All dates:")
        for d in dates_sorted:
            print(d)

if __name__ == "__main__":
    main()
