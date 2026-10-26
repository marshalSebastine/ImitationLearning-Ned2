import sys, fitz
import sys; sys.stdout.reconfigure(encoding="utf-8")
for p in sys.argv[1:]:
    d = fitz.open(p)
    print(f'===== {p}  pages={len(d)}')
    for i, page in enumerate(d):
        t = page.get_text()
        print(f'--- page {i+1} ---')
        print(t)
