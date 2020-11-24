lines = []
with open('./data/BORABIBLE.txt') as file:
    # replace all 14Aabéhjáa kind words into: 14 Aabéhjáa
    for line in file:
        line = line.replace('\n', '').split(' ')
        for i in range(0, 10000):
            if i == len(line):
                break
            if '0' <= line[i][0] <= '9':
                tmp = line[i]
                idx = 0
                while idx < len(tmp) and ('0' <= tmp[idx] <= '9' or tmp[idx] == ',' or tmp[idx] == '-' or tmp[idx] == '.'):
                    idx += 1
                num = tmp[0:idx]
                word = tmp[idx:]
                line[i] = num
                if word != '':
                    line.insert(i + 1, word)
        line = ' '.join(line)
        lines.append(line)

lines_cp = []
punctuations = [',', '.', '?', '—', ';', '(', ')', '¿', '!', ':', '“', '”', '‘', '’', '¡']
for line in lines:
    line = line.replace('\n', '').split(' ')
    for i in range(0, 10000):
        if i == len(line):
            break
        if len(line[i]) == 1:
            continue
        if line[i][0] in punctuations:
            tmp = line[i]
            line[i] = tmp[0]
            line.insert(i + 1, tmp[1:])
        if len(line[i]) == 1:
            continue
        if line[i][len(line[i]) - 1] in punctuations:
            tmp = line[i]
            line[i] = tmp[:len(tmp) - 1]
            line.insert(i + 1, tmp[len(tmp) - 1])
    line = ' '.join(line)
    lines_cp.append(line)

with open('./tokenized/bora.tok', 'w') as file:
    for line in lines_cp:
        file.write(line + '\n')
