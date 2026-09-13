import csv


def read_rows(path):
    with open(path, newline='') as source:
        return list(csv.DictReader(source))
