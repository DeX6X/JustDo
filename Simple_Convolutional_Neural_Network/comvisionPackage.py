def stringToVector(stringPGM: str) -> list:
    vector = []
    string = ""
    isSkipLine = False
    for char in stringPGM:
        if char == "\n": isSkipLine = False
        if isSkipLine: continue
        if char == "#":
            isSkipLine = True
            continue

        if char in " \n":
            if string != "":
                if isInteger(string): vector.append(int(string))
                else: vector.append(string)
            string = ""
            continue

        string += char

    return vector

def isInteger(string: str) -> bool: 
    if string == "": return False
    for char in string:
        if char not in "1234567890":
            return False

    return True

def vectorToMatrix(vectorPixel: list, metaData: list) -> list:
    width = metaData[1] # Colum
    lengh = metaData[2] # Row

    matrix = [[0 for _ in range(width)] for _ in range(lengh)]
    index = 0
    for row in range(lengh):
        for col in range(width):
            matrix[row][col] = vectorPixel[index]
            index += 1

    return matrix

def filterMetadata(vectorPGM: list) -> tuple:
    metaData = []
    for _ in range(4):
        metaData.append(vectorPGM.pop(0))

    filteredVector = vectorPGM

    return filteredVector, metaData

def averagePooling(matrixPixel: list) -> list: 
    width = len(matrixPixel[0])
    lengh = len(matrixPixel)
    newWidth = width // 2 if width % 2 == 0 else (width + 1) // 2
    newLengh = lengh // 2 if lengh % 2 == 0 else (lengh + 1) // 2
    matrixAverage = [[0 for _ in range(newWidth)] for _ in range(newLengh)]

    r_start = 0
    c_start = 0
    for r_new in range(newLengh):
        for c_new in range(newWidth):
            total = 0
            count = 0
            r_start = r_new * 2
            c_start = c_new * 2

            for row in range(r_start, min(r_start + 2, lengh)):
                for col in range(c_start, min(c_start + 2, width)):
                    total += matrixPixel[row][col]
                    count += 1

            matrixAverage[r_new][c_new] = total // count

    return matrixAverage

def visualizationMatrix(matrix: list):
    width = len(matrix[0])
    lengh = len(matrix)
    bufferASCII = " .:-=+*#%@"

    maxBright = 0
    for row in matrix:
        for num in row:
            if num > maxBright: maxBright = num

    if maxBright == 0: maxBright = 255

    for row in range(lengh):
        for col in range(width):
            p = matrix[row][col]
            indexBuffer = (p * 9) // maxBright
            print(bufferASCII[indexBuffer], end="")
        print()