import numpy as np
import re

def input_matrix(n, data_type):
    return np.array([input().split() for _ in range(n)], dtype=data_type)

def input_equation(n):
    print("Masukkan Persamaan")
    kiri_matrix = []  
    kanan_matrix = []  
    variables = set()
    
    for _ in range(n):
        equation = input()
        equation = equation.replace(" ", "").split("=")

        kiri = equation[0]
        kiri_terms = re.split(r"([+-])", kiri)  
        kiri_coefficients = []
        for term in kiri_terms:
            if term != "+" and term != "-":
                coefficient = term[:-1]  
                if coefficient == "":
                    coefficient = "1"  
                kiri_coefficients.append(float(coefficient))
                variable = term[-1]
                variables.add(variable)

        kiri_matrix.append(kiri_coefficients)
        kanan = float(equation[1])
        kanan_matrix.append(kanan)

    kiri_matrix = np.array(kiri_matrix, dtype=float)
    kanan_matrix = np.array(kanan_matrix, dtype=float)
    variables = np.sort(np.array(list(variables)))
    return kiri_matrix, kanan_matrix, variables

def input_complex(num_equations, num_variables):
    matrix_a = np.zeros((num_equations, num_variables), dtype=complex)
    matrix_b = np.zeros(num_equations, dtype=complex)
    equation_variables = set()

    print("Enter the equations:")
    for i in range(num_equations):
        equation = input(f"Equation {i+1}: ")
        equation = re.sub(r'([a-z])', r'1j*\1', equation)  
        coefficients = re.findall(r'[+-]?\d+\.?\d*|\d*\.?\d+[j]', equation)
        for j in range(num_variables):
            matrix_a[i, j] = complex(coefficients[j])
        matrix_b[i] = complex(coefficients[-1])
        variables = re.findall(r'[a-e]', equation)
        equation_variables.update(variables)

    equation_variables = sorted(list(equation_variables))

    return matrix_a, matrix_b, equation_variables

def determine_solution(matrix_a, matrix_b):
    rows, cols = matrix_a.shape
    if rows != cols:
        return "No unique solution"

    det_a = np.linalg.det(matrix_a)
    if det_a != 0:
        return "Unique solution"

    augmented_matrix = np.hstack((matrix_a, matrix_b.reshape(-1, 1)))
    rref_matrix = np.linalg.matrix_rank(augmented_matrix)

    if rref_matrix < np.linalg.matrix_rank(matrix_a):
        return "No solution"
    elif rref_matrix < cols:
        return "Infinite solutions"
    else:
        return "Unique solution"

def solve_matrix():
    n = int(input("Masukkan jumlah baris/kolom: "))
    print("Masukkan matriks A:")
    matrix_a = input_matrix(n, float)
    print("Masukkan matriks B:")
    matrix_b = input_matrix(n, float)
    
    y = determine_solution(matrix_a, matrix_b)
    print(y)
    if y == "Unique solution":
        x = np.linalg.solve(matrix_a, matrix_b)
        print("Hasilnya adalah:")
        print(x)
    
def solve_equation():
    n = int(input("Masukkan jumlah persamaan: "))
    matrix_a, matrix_b, c = input_equation(n)

    y = determine_solution(matrix_a, matrix_b)
    print(y)
    if y == "Unique solution":
        x = np.linalg.solve(matrix_a, matrix_b)
        print("Hasilnya adalah:")
        index = 0
        while index < len(c):
            result = str(c[index]) + " = " + str(x[index])
            print(result)
            index+=1            

def characteristicPolynomial_eigenvalue_eigenvector():
    n = int(input("Masukkan jumlah baris/kolom: "))
    print("Masukkan matriks:")
    matrix_input = input_matrix(n, float)
    
    characteristic_polynomial = np.poly(matrix_input)
    print("Karakteristik Polinomial: ", characteristic_polynomial)
    
    eigenvalue, eigenvector = np.linalg.eig(matrix_input)
    print("Eigenvalue: ", eigenvalue)
    print("Eigenvector:\n", eigenvector)
    
    A = matrix_input
    if len(eigenvalue) == A.shape[0]:
        print("Matrix A dapat didiagonalisasi")
        print("Sehingga")
        P = eigenvector
        P_inv = np.linalg.inv(P)
        print("Matrix P:\n", P)
        print("Matrix P inverse:\n", P_inv)
    else:
        print("Matrix A tidak dapat didiagonalisasi, sehingga matrix P dan inversenya tidak dapat dicari")
    
def svd():
    n = int(input("Masukkan jumlah baris/kolom: "))
    print("Masukkan matriks:")
    matrix_a = input_matrix(n, float)
    U, S, V = np.linalg.svd(matrix_a)
    print("Matriks U:")
    print(U)
    print("Matriks singular values:")
    print(S)
    print("Matriks V:")
    print(V)
    
def spl_complex_svd():
    n = int(input("Masukkan jumlah persamaan: "))
    m = int(input("Masukkan jumlah variabel: "))
    
    matrix_a, matrix_b, c = input_complex(n, m)
    U, s, Vh = np.linalg.svd(matrix_a)
    s_inv = np.zeros_like(matrix_a.T, dtype=complex)
    s_inv[:s.size, :s.size] = np.diag(1 / s)
    x = Vh.T @ s_inv @ U.T @ matrix_b

    print("Hasilnya adalah:")
    index = 0
    while index < len(c):
        result = str(c[index]) + " = " + str(x[index])
        print(result)
        index+=1  

print("Kalkulator Matriks")

while True:
    print("\nPilih operasi:")
    print("1. Mencari solusi persamaan linier(input matrix)")
    print("2. Mencari solusi persamaan linier(input persamaan)")
    print("3. Mencari karakteristik polinomial, eigenvalue, eigenvector")
    print("4. Mencari SVD")
    print("5. SPL Complek dengan SVD")
    print("0. Keluar")
    choice = int(input("Masukkan pilihan: "))

    if choice == 1:
        solve_matrix()
    elif choice == 2:
        solve_equation()
    elif choice == 3:
        characteristicPolynomial_eigenvalue_eigenvector()
    elif choice == 4:
        svd()
    elif choice == 5:
        spl_complex_svd()
    elif choice == 0:
        break
    else:
        print("Pilihan tidak valid. Silakan coba lagi.")
