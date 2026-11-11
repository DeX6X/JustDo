import math

"""
Matrix Data Representation & Forward Pass

Dense Layer
formula : Z = XW + b
X, W = Matrix
b = Vector
using Dot product between X and W 

X = input
W = Weight
b = Bias
"""
def linearCombination(matrixX, matrixW, vecterB) -> list:
    matrixC = dotProduct(matrixX, matrixW)
    rowSize = len(matrixC)
    colSize = len(matrixC[0])
    matrixZ = [[0 for _ in range(colSize)] for _ in range(rowSize)] # result of matrixC + vectorB

    for row in range(rowSize):
        for col in range(colSize):
            matrixZ[row][col] = matrixC[row][col] + vecterB[col]

    return matrixZ

def dotProduct(matrixA, matrixB) -> list:
    # C[i][j] = SUM(p = 0 to k-1) A[i][p] * B[p][j]
    k = len(matrixB)
    rowSize = len(matrixA)
    colSize = len(matrixB[0])
    matrixC = [[0 for _ in range(colSize)] for _ in range(rowSize)] # result of matrix product

    for row in range(rowSize):
        for col in range(colSize):
            for p in range(k):
                matrixC[row][col] += matrixA[row][p] * matrixB[p][col]

    return matrixC


"""
Activation Functions
sigmoid function : f(z) = 1 / (1 + e**(-z))
"""

def sigmoid(z) -> float:
    return 1 / (1 + math.exp(-z))

def matrixSigmoid(matrixZ) -> list:
    rowSize = len(matrixZ)
    colSize = len(matrixZ[0])
    result = [[0 for _ in range(colSize)] for _ in range(rowSize)]

    for row in range(rowSize):
        for col in range(colSize):
            result[row][col] = sigmoid(matrixZ[row][col])

    return result

"""
Loss Function / Mean Squared Error (MSE)
Loss formula = 1/m * sum(i = 1 to m)[(model)_i - y_i]^2
"""

def CalculateError(matrixA, matrixY) -> list: # get E not E square
    rowSize = len(matrixA)
    colSize = len(matrixA[0])
    matrixE = [[0 for _ in range(colSize)] for _ in range(rowSize)]

    for row in range(rowSize):
        for col in range(colSize):
            matrixE[row][col] = matrixA[row][col] - matrixY[row][col]

    return matrixE

def ESquare(matrixE) -> list:
    rowSize = len(matrixE)
    colSize = len(matrixE[0])
    result = [[0 for _ in range(colSize)] for _ in range(rowSize)]

    for row in range(rowSize):
        for col in range(colSize):
            result[row][col] = matrixE[row][col] ** 2

    return result

def MeanSquaredError(E) -> float: # square the E before input
    total = 0
    n = len(E) * len(E[0])
    for row in E:
        for num in row:
            total += num
            
    return total / n

"""
Backpropagation & Chain Rule
"""

def DerivativeofLoss_dA(matrixError) -> list:
    n = len(matrixError) * len(matrixError[0])

    frontTerm = 2 / n
    rowSize = len(matrixError)
    colSize = len(matrixError[0])
    matrixBackpropagation = [[0 for _ in range(colSize)] for _ in range(rowSize)]


    for row in range(rowSize):
        for col in range(colSize):
            matrixBackpropagation[row][col] = frontTerm * matrixError[row][col]

    return matrixBackpropagation

def chainRule_dZ(matrixA, matrixDfLoss) -> list:
    rowSize = len(matrixA)
    colSize = len(matrixA[0])
    result = [[0 for _ in range(colSize)] for _ in range(rowSize)]

    for row in range(rowSize):
        for col in range(colSize):
            A = matrixA[row][col]
            result[row][col] = A * matrixDfLoss[row][col] * (1 - A) 

    return result

def transpose(matrix) -> list:
    rowSize = len(matrix)
    colSize = len(matrix[0])
    result = [[0 for _ in range(rowSize)] for _ in range(colSize)]

    for row in range(rowSize):
        for col in range(colSize):
            result[col][row] = matrix[row][col]

    return result

def shapeMatching_dW(transposeMatrix, dfLoss) -> list:
    return dotProduct(transposeMatrix, dfLoss)

def GradientOfBias_db(matrixDz):
    rowSize = len(matrixDz)
    colSize = len(matrixDz[0])
    result = [0 for _ in range(colSize)]

    for col in range(colSize):
        for row in range(rowSize):
            result[col] += matrixDz[row][col]

    return result

"""
Gradient Descent & Training Loop
"""

def GradientDescent(W, b, dW, db, learningRate):
    rowSize = len(W)
    colSize = len(W[0])

    for row in range(rowSize):
        for col in range(colSize):
            W[row][col] = W[row][col] - (learningRate * dW[row][col])

    for col in range(len(db)):
        b[col] = b[col] - (learningRate * db[col])

    return W, b, dW, db, learningRate
             
def display2dArr(arr) -> None:
    for row in arr:
        print(row)
    print()

def main():
    # 1. กำหนดมิติข้อมูล
    #m = 2  # จำนวน Sample (Batch Size)
    #n_in = 3
    n_out = 4
    W = [
        [ 0.15, -0.32,  0.41, -0.08],  # Weight Input 1 -> 4 Output
        [-0.24,  0.05, -0.19,  0.37],  # Weight Input 2 -> 4 Output
        [ 0.48, -0.11,  0.02, -0.45]   # Weight Input 3 -> 4 Output
    ]

    # 2. Input X(2 x 3) 
    # Trainning input represent the scores of ability [Programming, Math, English]
    X = [
        [0.5, 0.8, 0.2],
        # Person 1
        # Programming : Moderate
        # Math : Good
        # English : Bad

        [0.1, 0.9, 0.6]
        # Person 2
        # Programming : Bad
        # Math : Good
        # English : Moderate
    ]

    # 3. Target/Answer Y(2 x 4) represent Major and Brach [Data Science, Pure Math, Business, English]
    Y = [
        [1.0, 0.0, 1.0, 0.0],
        # Person 1 
        # Proper in : Data Science, Business

        [0.0, 1.0, 0.0, 1.0]
        # Person 2
        # Proper in : Pure Math, English
    ]

    # 4. Bias B(1 x 4)
    b = [0.0 for _ in range(n_out)]

    # 5. Hyperparameter
    learningRate = 0.5
    epochs = 1000

    # Display Target
    print("-"*4 + " Answer " + "-"*4)
    display2dArr(Y)

    # Trainning
    for epoch in range(epochs):
        # Forward Pass
        Z = linearCombination(X, W, b)
        A = matrixSigmoid(Z)

        # Display Model Before Trainning
        if (epoch == 0):
            print("-"*4 + " init Model " + "-"*4)
            display2dArr(A)

        # Loss Check
        E = CalculateError(A, Y)
        E_square = ESquare(E)
        Loss = MeanSquaredError(E_square)

        # Backpropagation
        dLoss_dA = DerivativeofLoss_dA(E)
        dZ = chainRule_dZ(A, dLoss_dA)
        XTranspose = transpose(X)
        dW = shapeMatching_dW(XTranspose, dZ)
        db = GradientOfBias_db(dZ)

        # Update Parameters
        W, b, dW, db, learningRate = GradientDescent(W, b, dW, db, learningRate)

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: Loss = {Loss:.6f}")
            
            # Display Model After Trainning
            if(epoch == epochs - 1):
                print("\n" + "-"*4 + " Trained Model " + "-"*4)
                display2dArr(A)

    # Output after trained
    # [0.7869077959764706, 0.20850225442726866, 0.793263119772576, 0.2133907687283896]
    # Person 1
    # Model Scored : Data Science 78%, Business 79%
    # Model Reject : Pure Math 20%, English 21%

    # [0.20536371364858752, 0.7990263670184011, 0.19928455924527766, 0.79435079823512]
    # Person 2
    # Model Scored : Pure Math 79%, English 79%
    # Model Reject : Data Science 20%, Business 19%


    # Use new Input to test Model
    X_test = [
        [0.9, 0.1, 0.1], # Person 1
        [0.4, 0.4, 0.2] # Person 2
    ]

    Z_test = linearCombination(X_test, W, b)
    A_test = matrixSigmoid(Z_test)

    print("\n" + "-"*4 + " New Test Prediction " + "-"*4)
    display2dArr(A_test)
    # Output of new input by using above model
    # Person 1
    # Model Scored : Data Science 97%, Business 97%
    # Model Reject : Pure Math 3%, English 2%

    # Person 2
    # Model Scored : Data Science 76%, Business 76%
    # Model Reject : Pure Math 23%, English 22%
# -------------------------------------------------------------------------------------

if "__main__" == __name__:
    main()

# -------------------------------------------------------------------------------------