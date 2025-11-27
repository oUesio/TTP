'''
Entry point
(loading instance → running HGS → saving nd_archive output).
'''

import os, csv, time
from loader import load_ttp
from hgs import run_hgs

def main():
    # a280-n279 default path
    path_candidates = ["gecco19-thief/src/main/resources/a280-n279.txt"]
    PATH=None
    for p in path_candidates:
        if os.path.exists(p):
            PATH=p; break
    if PATH is None:
        raise RuntimeError("Instance file a280-n279.txt not found.")

    prob = load_ttp(PATH)
    nd = run_hgs(prob)

    # dumps: exact competition format (time, profit)
    with open("nd_archive.csv","w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["time","profit"])
        for s in nd:
            w.writerow([f"{s.obj[0]:.6f}", f"{-s.obj[1]:.6f}"])

    with open("nd_archive.f","w",encoding="utf-8") as f:
        for s in nd:
            f.write(f"{s.obj[0]:.9f} {-s.obj[1]:.0f}\n")

    print("\nSaved: nd_archive.csv (time, profit), nd_archive.f (time profit)")

if __name__=="__main__":
    # wallclock anchor for per-gen cap
    last_time = time.time()
    main()