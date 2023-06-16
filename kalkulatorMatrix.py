import numpy as np
import re

def input_matrix(n):
    return np.array([input().split() for _ in range(n)], dtype=float)

def input_equation(n):
    print("Masukkan Persamaan")
    kiri_matrix = []  
    kanan_matrix = []  
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

        kanan = float(equation[1])

        kiri_matrix.append(kiri_coefficients)
        kanan_matrix.append(kanan)

    kiri_matrix = np.array(kiri_matrix)
    kanan_matrix = np.array(kanan_matrix)
    return kiri_matrix, kanan_matrix

def solve_matrix():
    n = int(input("Masukkan jumlah baris/kolom: "))
    print("Masukkan koefisien matriks A (baris x kolom):")
    a = input_matrix(n)
    print("Masukkan matriks B:")
    b = input_matrix(n)
    x = np.linalg.solve(a, b)
    print("Hasilnya adalah:")
    print(x)

def solve_equation():
    n = int(input("Masukkan jumlah persamaan: "))
    a, b = input_equation(n)
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
    n = int(input("Masukkan jumlah baris/kolom: "))
    print("Masukkan matriks:")
    a = input_matrix(n)
    U, S, V = np.linalg.svd(a)
    print("Matriks U:")
    print(U)
    print("Matriks singular values:")
    print(S)
    print("Matriks V:")
    print(V)
    
def spl_complex_svd():
    n = int(input("Masukkan jumlah baris/kolom: "))
    print("Masukkan koefisien matriks A (baris x kolom):")
    a = input_matrix(n)
    print("Masukkan matriks B:")
    b = input_matrix(n)
    U,s,Vh = np.linalg.svd(a)
    c = np.dot(U.T.conj(), b)
    w = np.divide(c[:len(s)], s)
    x = np.dot(Vh.T.conj(), w)

print("Kalkulator Matriks")

while True:
    print("\nPilih operasi:")
    print("1. Mencari solusi persamaan linier(input matrix)")
    print("2. Mencari solusi persamaan linier(input persamaan)")
    print("3. Mendiagonalisasi Matriks")
    print("4. Mencari SVD")
    print("5. SPL Complek dengan SVD")
    print("0. Keluar")
    choice = int(input("Masukkan pilihan: "))

    if choice == 1:
        solve_matrix()
    elif choice == 2:
        solve_equation()
    elif choice == 3:
        diagonalize()
    elif choice == 4:
        svd()
    elif choice == 5:
        spl_complex_svd()
    elif choice == 0:
        break
    else:
        print("Pilihan tidak valid. Silakan coba lagi.")
