import gzip, csv

files = ['patients.csv.gz','admissions.csv.gz','diagnoses_icd.csv.gz','labevents.csv.gz','d_icd_diagnoses.csv.gz','d_labitems.csv.gz','prescriptions.csv.gz']
for f in files:
    print(f'\n=== {f} ===')
    with gzip.open(f, 'rt') as fp:
        reader = csv.reader(fp)
        for i, row in enumerate(reader):
            if i >= 3: break
            print(row)