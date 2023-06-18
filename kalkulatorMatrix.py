import numpy as np
import re

def input_matrix(n, type):
    return np.array([input().split() for _ in range(n)], dtype=type)

def input_equation(n):
    print("Masukkan Persamaan")
    kiri_matrix = []  
    kanan_matrix = []  
    variables = set()
    
    f = open('readme.txt', 'w')
    f.write('Persamaan :\n')
    
    for _ in range(n):
        equation = input()
        
        f.write(equation)
        f.write('\n')
    
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

    kiri_matrix = np.array(kiri_matrix)
    kanan_matrix = np.array(kanan_matrix)
    variables = np.sort(np.array(list(variables)))
    f.close()
    return kiri_matrix, kanan_matrix, variables

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
    if matrix_a.shape[0] != matrix_a.shape[1]:
        print("ERROR : Matriks tidak persegi")
        return
    
    f = open('readme.txt', 'w')
    f.write('Matriks A :\n')
    for row in matrix_a:
        f.write('[')
        f.write(' '.join(str(element) for element in row))
        f.write(']')
        f.write('\n')
    
    print("Masukkan matriks B:")
    matrix_b = input_matrix(n, float)
    if matrix_b.shape[1] > 1 or matrix_b.shape[0] !=  matrix_a.shape[0]:
        print("ERROR : Ukuran vektor tidak sesuai")
        return

    f.write('\nMatriks B :\n')
    for row in matrix_b:
        f.write('[')
        f.write(' '.join(str(element) for element in row))
        f.write(']')
        f.write('\n')
    
    y = determine_solution(matrix_a, matrix_b)
    print(y)
    
    f.write('\n')
    f.write(y)
        
    if y == "Unique solution":
        x = np.linalg.solve(matrix_a, matrix_b)
        print("Hasilnya adalah:")
        print(x)
        
        f.write('\nHasil SPL adalah :\n')
        for row in x:
            f.write('[')
            f.write(' '.join(str(element) for element in row))
            f.write(']')
            f.write('\n')
    f.close()
            
def solve_equation():
    n = int(input("Masukkan jumlah persamaan: "))
    print('(Ketik persamaan dalam bentuk (xa + yb + zc + kd + .... = i) dimana x,y,z,k dan i adalah koefisien)', '\n')
    matrix_a, matrix_b, c = input_equation(n)

    y = determine_solution(matrix_a, matrix_b)
    print(y)
    
    f = open('Readme.txt','a')
    f.write('\n')
    f.write(y)
    f.write('Hasilnya adalah:\n')
    
    if y == "Unique solution":
        x = np.linalg.solve(matrix_a, matrix_b)
        print("Hasilnya adalah:")
        index = 0
        while index < len(c):
            result = str(c[index]) + " = " + str(x[index])
            print(result)
            
            f.write(result)
            f.write('\n')
            
            index+=1
                        
    f.close()
    
def characteristicPolynomial_eigenvalue_eigenvector():
    n = int(input("Masukkan jumlah baris/kolom: "))
    print("Masukkan matriks:")
    matrix_input = input_matrix(n, float)
    if matrix_input.shape[0] != matrix_input.shape[1]:
        print("ERROR : Matriks tidak persegi")
        return
    
    f = open('Readme.txt','w')
    f.write('Matriks :\n')
    for row in matrix_input:
        f.write('[')
        f.write(' '.join(str(element) for element in row))
        f.write(']')
        f.write('\n')
    
    
    characteristic_polynomial = np.poly(matrix_input)
    print("\nKarakteristik Polinomial: ", characteristic_polynomial)
    
    f.write('\nKarakteristik Polinomial: \n')
    for row in characteristic_polynomial:
        f.write('[')
        f.write(str(row))
        f.write(']')
        f.write('\n')
    
    eigenvalue, eigenvector = np.linalg.eig(matrix_input)
    print("Eigenvalue: ", eigenvalue)
    print("Eigenvector:\n", eigenvector)
    
    f.write('\nEigenvalue : ')
    for row in eigenvalue:
        f.write('[')
        f.write(str(row))
        f.write(']')
        f.write('\n')
    
    f.write('\nEigenvector : \n')
    for row in eigenvector:
        f.write('[')
        f.write(' '.join(str(element) for element in row))
        f.write(']')
        f.write('\n')
    
    A = matrix_input
    if len(eigenvalue) == A.shape[0]:
        print("\nMatrix A dapat didiagonalisasi")
        print("Sehingga")
        
        f.write('\nMatrix A dapat didiagonalisasi')
        f.write(' sehingga')
        
        P = eigenvector
        P_inv = np.linalg.inv(P)
        print("Matrix P:\n", P)
        print("\nMatrix P inverse:\n", P_inv)
        
        f.write('\nMatrix P:')
        for row in P:
            f.write('[')
            f.write(str(row))
            f.write(']')
            f.write('\n')
        f.write('\n')
        f.write('\nMatrix P inverse:\n')
        for row in P_inv:
            f.write('[')
            f.write(' '.join(str(element) for element in row))
            f.write(']')
            f.write('\n')
        f.close()
    else:
        print("Matrix A tidak dapat didiagonalisasi, sehingga matrix P dan inversenya tidak dapat dicari")
        f.write('\n')
        f.write('Matrix A tidak dapat didiagonalisasi, sehingga matrix P dan inversenya tidak dapat dicari')
        f.close()
    
def svd():
    n = int(input("Masukkan jumlah baris/kolom: "))
    print("Masukkan matriks:")
    matrix_a = input_matrix(n, float)
    if matrix_a.shape[0] != matrix_a.shape[1]:
        print("ERROR : Matriks tidak persegi")
        return
    
    f = open('Readme.txt','w')
    f.write('Matriks :\n')
    for row in matrix_a:
            f.write('[')
            f.write(' '.join(str(element) for element in row))
            f.write(']')
            f.write('\n')
            
    U, S, V = np.linalg.svd(matrix_a)
    print("Matriks U:")
    print(U)
    
    f.write('Matriks U :\n')
    for row in U:
            f.write('[')
            f.write(' '.join(str(element) for element in row))
            f.write(']')
            f.write('\n')
            
    print("Matriks singular values:")
    print(S)
    
    f.write('Matriks singular values:\n')
    for row in S:
            f.write('[')
            f.write(str(row))
            f.write(']')
            f.write('\n')
            
    print("Matriks V:")
    print(V)
    
    f.write('Matriks V:\n')
    for row in V:
            f.write('[')
            f.write(' '.join(str(element) for element in row))
            f.write(']')
            f.write('\n')
    
    f.close()
    
def spl_complex_svd():
    print("Matriks A:")
    n = int(input("Masukkan jumlah baris untuk Matriks A: "))   
    print("Masukkan koefisien matriks A (baris x kolom):")
    matrix_a = input_matrix(n, complex)
    
    f = open('Readme.txt','w')
    f.write('Matriks A :\n')
    for row in matrix_a:
            f.write('[')
            f.write(' '.join(str(element) for element in row))
            f.write(']')
            f.write('\n')
            
    print("Matriks B:")
    n = int(input("Masukkan jumlah baris untuk Matriks B: "))
    print("Masukkan koefisien matriks B (baris x kolom):")
    matrix_b = input_matrix(n, complex)
    
    f.write('\nMatriks B :\n')
    for row in matrix_b:
            f.write('[')
            f.write(' '.join(str(element) for element in row))
            f.write(']')
            f.write('\n')
            
    U,s,Vh = np.linalg.svd(matrix_a)
    c = np.dot(U.T.conj(), matrix_b)
    w = np.divide(c[:len(s)], s)
    x = np.dot(Vh.T.conj(), w)
    print("Hasil SPL :")
    print(x)

    f.write('\nHasil SPL :\n')
    for row in x:
            f.write('[')
            f.write(' '.join(str(element) for element in row))
            f.write(']')
            f.write('\n')
    
    f.close()

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
        print("ERROR : Pilihan tidak valid. Silakan coba lagi.")
