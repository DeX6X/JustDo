"""
-------------------------------------------------------------------------------------
String to vector with filter the comment and change it to Integer
-------------------------------------------------------------------------------------
"""
# *Main function in this domain* Transform the string PGM into Vector Pixel
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

# *Helper function* for Checking the string is Integer
def isInteger(string: str) -> bool: 
    if string == "": return False
    for char in string:
        if char not in "1234567890":
            return False

    return True

"""
-------------------------------------------------------------------------------------
Vector to matrix
-------------------------------------------------------------------------------------
"""
# *Main function in this domain* Transform Vector Pixel into Matrix Pixel
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

# *Helper function* for Filter Metadata out of Pixel data
def filterMetadata(vectorPGM: list) -> tuple:
    metaData = []
    for _ in range(4):
        metaData.append(vectorPGM.pop(0))

    filteredVector = vectorPGM

    return filteredVector, metaData

"""
-------------------------------------------------------------------------------------
Downsampling matrix
-------------------------------------------------------------------------------------
"""

# Reduce file size a half of it by using average pooling method
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

"""
-------------------------------------------------------------------------------------
ASCII Art Generator
-------------------------------------------------------------------------------------
"""
# Visualized the matrix into Terminal
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

"""
-------------------------------------------------------------------------------------
Main
-------------------------------------------------------------------------------------
"""
def main():
    # Open file
    numberFilePGM_P2 = open("dataset\\handwrite2_fiveP2.pgm")
    numberStringPGM_P2 = numberFilePGM_P2.read()

    # String to Vector
    vectorPGM_P2 = stringToVector(numberStringPGM_P2)

    # Seperate Metadata
    fileredVector, metaData = filterMetadata(vectorPGM_P2)

    # Vector to Matrix
    matrixPixel = vectorToMatrix(fileredVector, metaData)

    # Down Sampling use Average Pooling Method
    downsamplingMatrix = averagePooling(matrixPixel)

    visualizationMatrix(downsamplingMatrix)

# -----------------------------   Run main code  --------------------------------------

if "__main__" == __name__:
    main()

# -------------------------------------------------------------------------------------