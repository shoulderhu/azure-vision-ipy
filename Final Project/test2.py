with open("plate/output.txt") as f:
    lines = f.readlines()

with open("output/output.txt") as f:
    lines2 = f.readlines()

for i, l in enumerate(lines):
    line = l.replace("\n", "")
    line2 = lines2[i].replace("\n", "")
    text = line.split(" ")
    text2 = line2.split(" ")

    if len(text) == 1:
        print(text[0])
    elif len(text) == 4:
        for j in range(4):
            d = abs(int(text[j]) - int(text2[j]))
            print(d, end=" ")
            if d > 2:
                print("error", end=" ")
        print()
