import csv

with open("result_ppl.csv", "r") as file:
    ppl = file.readlines()[1].split(",")[1].strip()

with open("result_tput.csv", "r") as file:
    tput = file.readlines()[1].split(",")[1].strip()

with open("result.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Id", "value"])
    writer.writerow([0, ppl])
    writer.writerow([1, tput])
