import comvisionPackage as scv
from math import exp, log
import json

# Pull the Weight from json file in dataset
def load_weight(filepath: str) -> list:
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data["W"]

# linear function for CNN classify number
def linearCombination(X: list, W: list, B: list) -> list:
    matrixC = dotProduct(X, W)
    width = len(matrixC[0])
    lengh = len(matrixC)
    matrixZ = [[0 for _ in range(width)] for _ in range(lengh)]

    for row in range(lengh):
        for col in range(width):
            matrixZ[row][col] = matrixC[row][col] + B[col]

    return matrixZ

# Help function for linear in CNN
def dotProduct(matrixA: list, matrixB: list) -> list:
    k = len(matrixB)
    lengh = len(matrixA)
    width = len(matrixB[0])
    matrixC = [[0 for _ in range(width)] for _ in range(lengh)]

    for row in range(lengh):
        for col in range(width):
            for p in range(k):
                matrixC[row][col] += matrixA[row][p] * matrixB[p][col]

    return matrixC

# only for Weight in this case
def transposeMatrix(matrix: list) -> list:
    newWidth = len(matrix[0])
    newLengh = len(matrix)
    matrixTranspose = [[0 for _ in range(newLengh)] for _ in range(newWidth)]

    for row in range(newLengh):
        for col in range(newWidth):
            matrixTranspose[col][row] = matrix[row][col]

    return matrixTranspose

# From .pgm into matrix(Allow only png/jpg at 100 * 100 dimension)
def getMatrixPixel(filepath: str) -> list:
    numberFilePGM_P2 = open(filepath)
    numberStringPGM_P2 = numberFilePGM_P2.read()

    # String to Vector
    vectorPGM_P2 = scv.stringToVector(numberStringPGM_P2)

    # Seperate Metadata
    fileredVector, metaData = scv.filterMetadata(vectorPGM_P2)

    # Vector to Matrix
    matrixPixel = scv.vectorToMatrix(fileredVector, metaData)

    return matrixPixel

"""
-------------------------------------------------------------------------------------
Normalization matrix
-------------------------------------------------------------------------------------
"""
# Normalize data from 0 - 255 in 0.0 - 1.0
def normalizeMatrix(matrixPixel: list) -> list:
    lengh = len(matrixPixel)
    width = len(matrixPixel[0])
    normalizedMatrix = [[0 for _ in range(width)] for _ in range(lengh)]

    maxBright = 0
    for row in matrixPixel:
        for num in row:
            if num > maxBright: maxBright = num

    if maxBright == 0: maxBright = 255

    for row in range(lengh):
        for col in range(width):
            normalizedMatrix[row][col] = matrixPixel[row][col] / maxBright

    return normalizedMatrix

"""
-------------------------------------------------------------------------------------
Convolutional Layer (Forward Pass)
-------------------------------------------------------------------------------------
"""
# *Main function in this domain* for isolate specific characteristics
def convolutionalLayer(matrix: list, kernel: list, bias: int) -> list:
    matrixSize = len(matrix)
    kernelSize = len(kernel)
    featureSize = matrixSize - kernelSize + 1
    featureMap = [[0 for _ in range(featureSize)] for _ in range(featureSize)]

    for row in range(featureSize):
        for col in range(featureSize):
            total = bias

            for r in range(kernelSize):
                for c in range(kernelSize):
                    total += matrix[row + r][col + c] * kernel[r][c]

            featureMap[row][col] = ReLU(total)

    return featureMap

# *Helper function* for making the feature map eazy to calculate derivative of feature map(dF use for adjust the value)
def ReLU(x: float) -> float:
    return max(x, 0.0)

"""
-------------------------------------------------------------------------------------
Max Pooling Layer
-------------------------------------------------------------------------------------
"""
# Reduce size of feature map  
def maxPooling(featureMap: list) -> list:
    featureSize = len(featureMap)
    newFeatureSize = ((featureSize - 2) // 2) + 1
    reduceFeature = [[0.0 for _ in range(newFeatureSize)] for _ in range(newFeatureSize)]

    r_start = 0
    c_start = 0
    for r_new in range(newFeatureSize):
        for c_new in range(newFeatureSize):
            maxValue = 0.0
            r_start = r_new * 2
            c_start = c_new * 2
            for row in range(r_start, r_start + 2):
                for col in range(c_start, c_start + 2):
                    maxValue = max(maxValue, featureMap[row][col])

            reduceFeature[r_new][c_new] = maxValue

    return reduceFeature

"""
-------------------------------------------------------------------------------------
Flattening & Dense Layer
-------------------------------------------------------------------------------------
"""
# Transform feature map matrix into vector
def flattenMatrix(featureMap: list) -> list:
    lengh = len(featureMap)
    width = len(featureMap[0])
    vector = [0.0 for _ in range(width * lengh)]

    index = 0
    for row in range(lengh):
        for col in range(width):
            vector[index] = featureMap[row][col]
            index += 1

    return vector

# Calculate probability of activation
def activation(Z) -> list:
    lengh = len(Z)
    activationVector = [0.0 for _ in range(lengh)]

    total = 0
    for index in range(lengh):
        calculate = exp(Z[index])
        activationVector[index] = calculate
        total += calculate

    P = [x / total for x in activationVector]

    return P

"""
-------------------------------------------------------------------------------------
Backpropagation in Dense Layer
-------------------------------------------------------------------------------------
"""

def GradientOfZ(P: list, Y:list) -> list:
    gradientSize = len(P)
    gradientVector = [0.0 for _ in range(gradientSize)]

    for index in range(gradientSize):
        gradientVector[index] = P[index] - Y[index]

    return gradientVector

def GradientOfW(dZ: list, X: list) -> list:
    pass

def crossEntropy(Y: list, P: list, c: int) -> list:
    total = 0
    for index in range(c):
        total += Y[index] * log(P[index])

    Loss = -1 * total

    return Loss

"""
-------------------------------------------------------------------------------------
function for predict one images (Allow only PNG/JPG)
-------------------------------------------------------------------------------------
"""
def predict_image(image_matrix: list, Kernel: list, W: list, B: list):
    # Preprocessing / Normalize
    normalized_img = normalizeMatrix(image_matrix)
    
    # Convolution & ReLU
    feature_map = convolutionalLayer(normalized_img, Kernel, bias=0.0)
    
    # Max Pooling
    max_pool_map = maxPooling(feature_map)
    
    # Flatten
    feature_vector = [flattenMatrix(max_pool_map)]
    
    # Linear Transformation (Dense Layer)
    Z = linearCombination(feature_vector, W, B)
    
    # Softmax Activation
    P = activation(Z[0])
    
    # predict the result
    predicted_digit = P.index(max(P))
    confidence = max(P) * 100
    
    return predicted_digit, confidence, P
"""
-------------------------------------------------------------------------------------
Main
-------------------------------------------------------------------------------------
"""
def main():
    # Preparing Data
    directory = "dataset\\"
    fileName = [ "digit_0.pgm", "digit_1.pgm", "digit_2.pgm", "digit_3.pgm", "digit_4.pgm", "digit_5.pgm", "digit_6.pgm", "digit_7.pgm", "digit_8.pgm", "digit_9.pgm" ]
    X = []
    for f in fileName:
        filePGM_P2 = open(directory + f)
        numberStringPGM_P2 = filePGM_P2.read()
        vectorPGM_P2 = scv.stringToVector(numberStringPGM_P2)
        filteredVector, metaData = scv.filterMetadata(vectorPGM_P2)
        matrixPixel = scv.vectorToMatrix(filteredVector, metaData)

        X.append(matrixPixel)

    Kernel = [
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ]
    b_conv = 0.0

    # Weight data size (10 * 2401)
    json_path = "dataset\\weights_W.json"
    W = load_weight(json_path)
    W = transposeMatrix(W)
    
    # Bias vector for 10 node
    B = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    Y_targets = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Target สำหรับภาพเลข 0
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Target สำหรับภาพเลข 1
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Target สำหรับภาพเลข 2
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # Target สำหรับภาพเลข 3
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # Target สำหรับภาพเลข 4
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # Target สำหรับภาพเลข 5
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # Target สำหรับภาพเลข 6
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # Target สำหรับภาพเลข 7
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # Target สำหรับภาพเลข 8
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]   # Target สำหรับภาพเลข 9
    ]

    print("=== Result of Trained images 10 number (0 - 9) ===")
    
    # Loop Training
    for i in range(len(X)):
        # Preprocessing / Normalize
        normalized_img = normalizeMatrix(X[i])
        
        # Convolution & ReLU
        feature_map = convolutionalLayer(normalized_img, Kernel, b_conv)
        
        # Max Pooling
        max_pool_map = maxPooling(feature_map)
        
        # Flatten
        feature_vector = [flattenMatrix(max_pool_map)]
        
        # Linear Transformation (Dense Layer)
        Z = linearCombination(feature_vector, W, B)
        
        # Softmax Activation
        P = activation(Z[0])
        
        # Find the number from model predicted(the highest one)
        predicted_digit = P.index(max(P))
        
        # Calculate loss derivative with Y(target)
        loss = crossEntropy(Y_targets[i], P, 10)
        
        print(f"Image: {i} | predict: {predicted_digit} | probability: {P[predicted_digit]*100:.2f}% | Loss: {loss:.4f}")

    # ---------------------------------------------------------------------------------
    # Input into Model
    # ---------------------------------------------------------------------------------
    print("\n=== Result of model from outer input (PNG/JPG) ===")
    
    # Path input
    custom_pgm_path = "dataset\\handwrite_pgm_p2\\handwrite_8.pgm"

    try:
        # Clean Data
        matrixPixel = getMatrixPixel(custom_pgm_path)

        # Predict
        digit, confidence, probs = predict_image(matrixPixel, Kernel, W, B)

        # Display Result
        print(f"File: {custom_pgm_path}")
        print(f"Predicted Digit : {digit}")
        print(f"Confidence      : {confidence:.2f}%")
        print("\nProbabilities of all classes (0-9):")
        for idx, prob in enumerate(probs):
            print(f"  Class {idx}: {prob*100:.2f}%")

    except FileNotFoundError:
        print(f"Error: File not found at '{custom_pgm_path}'")

# -------------------------------------------------------------------------------------

if "__main__" == __name__:
    main()

# -------------------------------------------------------------------------------------
