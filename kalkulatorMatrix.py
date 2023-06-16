import numpy as np

def input_matrix():
    return np.array([input().split() for _ in range(n)], dtype=float)

def input_equation(n=0):
    matrix = []
    if n == 0:
        n = int(input("Masukkan jumlah persamaan: "))

    print("Masukkan Persamaan")
    lhs_matrix = []  # Matrix for the left-hand side of equations
    rhs_matrix = []  # Matrix for the right-hand side of equations
    for _ in range(n):
        equation = input()
        equation = equation.replace(" ", "")
        equation = equation.split("=")

        # Process the left-hand side of the equation
        lhs = equation[0]
        lhs_terms = re.split(r"([+-])", lhs)  # Split on + or -
        lhs_coefficients = []
        for term in lhs_terms:
            if term != "+" and term != "-":
                coefficient = term[:-1]  # Extract the coefficient (excluding the variable)
                if coefficient == "":
                    coefficient = "1"  # If no coefficient is specified, assume it's 1
                lhs_coefficients.append(float(coefficient))

        # Process the right-hand side of the equation
        rhs = float(equation[1])

        lhs_matrix.append(lhs_coefficients)
        rhs_matrix.append(rhs)

    lhs_matrix = np.array(lhs_matrix)
    rhs_matrix = np.array(rhs_matrix)
    return lhs_matrix, rhs_matrix

def solve_matrix():
    print("Masukkan koefisien matriks A (baris x kolom):")
    a = input_matrix()
    print("Masukkan matriks B:")
    b = input_matrix()
    x = np.linalg.solve(a, b)
    print("Hasilnya adalah:")
    print(x)

def solve_equation():
    print("Masukkan jumlah persamaan")
    a, b = input_equation()
    x = np.linalg.solve(a, b)
    print("Hasilnya adalah:")
    print(x)

def diagonalize():
    print("Masukkan matriks:")
    a = input_matrix()
    diagonal = np.diag(np.diag(a))
    print("Diagonalisasi dari A adalah :")
    print(diagonal)
    
def svd():
    print("Masukkan matriks:")
    a = input_matrix()
    U, S, V = np.linalg.svd(a)
    print("Matriks U:")
    print(U)
    print("Matriks singular values:")
    print(S)
    print("Matriks V:")
    print(V)
    
def spl_complex_svd():
    print("Masukkan koefisien matriks A (baris x kolom):")
    a = input_matrix()
    print("Masukkan matriks B:")
    b = input_matrix()
    U,s,Vh = np.linalg.svd(a)
    c = np.dot(U.T.conj(), b)
    w = np.divide(c[:len(s)], s)
    x = np.dot(Vh.T.conj(), w)

print("Kalkulator Matriks")
n = int(input("Masukkan jumlah baris/kolom: "))

while True:
    print("\nPilih operasi:")
    print("1. Mencari solusi persamaan linier")
    print("2. Mendiagonalisasi Matriks")
    print("3. Mencari SVD")
    print("4. Keluar")
    choice = int(input("Masukkan pilihan: "))

    if choice == 1:
        solve_linear_equation()
    elif choice == 2:
        diagonalize()
    elif choice == 3:
        svd()
    elif choice == 4:
        spl_complex_svd()
    elif choice == 4:
        break
    else:
        print("Pilihan tidak valid. Silakan coba lagi.")
        
def solve_linear_equation():
    print("Masukkan koefisien matriks A (baris x kolom):")
    a = np.array([input().split() for _ in range(n)], dtype=float)
    print("Masukkan matriks B:")
    b = np.array(input().split(), dtype=float)
    x = np.linalg.solve(a, b)
    print("Hasilnya adalah:")
    print(x)

def diagonalize():
    print("Masukkan matriks:")
    a = np.array([input().split() for _ in range(n)], dtype=float)
    diagonal = np.diag(np.diag(a))
    print("Diagonalisasi dari A adalah :")
    print(diagonal)