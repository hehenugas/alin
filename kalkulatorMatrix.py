import numpy as np
import re
import math
import os

def input_matrix(n, data_type):
    return np.array([input().split() for _ in range(n)], dtype=data_type)

def clear_txt():
    with open("readme.txt",'w') as file:
        pass

def file_string(message):
    with open("readme.txt", 'a') as file:
        print(message, file=file)
        
def file_matrix(matrix):
    with open("readme.txt", 'a') as file:
        print(matrix, file=file)

def fileProcessing(message, matrix):
    with open("readme.txt", 'a') as file:
        print(message, file=file)
        print(matrix, file=file)
        
def input_equation(n):
    string_matrix_a = []
    string_matrix_b = []
    for _ in range(n):
        input_str = input(); file_string(input_str)
        input_str = input_str.replace(" ", "").split("=");
        buffer_kanan = input_str[1]
        string_matrix_b.append(buffer_kanan)
        buffer_kiri = ''
        for char in input_str[0]:
            if char == "-":
                buffer_kiri += "+-"  
            else:
                buffer_kiri += char
        input_str = buffer_kiri.split("+")
        string_matrix_a.append(input_str)
    
    a = []
    for row in string_matrix_a:
        extracted_sublist = []
        for i in range(len(row)):
            for j in range(len(row[i])):
                temp = re.sub(r'(?<![0-9])[a-zA-Z]', r'1\g<0>', row[i]) 
                row[i] = temp
            matches = re.findall(r'-?\d+', row[i])
            extracted_element = int(matches[0])
            extracted_sublist.append(extracted_element)
        a.append(extracted_sublist)
        
    b = []
    for element in string_matrix_b:
        matches = re.findall(r'-?\d+', element)
        extracted_element = int(matches[0])
        b.append(extracted_element)
        
    matrix_a = np.array(a)
    matrix_b = np.array(b)
    return matrix_a, matrix_b

def input_complex(num_equations):
    matrix_a = []
    matrix_b = []

    string_matrix_a = []
    string_matrix_b = []
    for _ in range(num_equations):
        input_str = input(); file_string(input_str)
        input_str = input_str.replace(" ", "").split("=")
        buffer_kanan = input_str[1]
        string_matrix_b.append(buffer_kanan)
        buffer_kiri = ''
        for char in input_str[0]:
            if char == "-":
                buffer_kiri += "+-"  
            else:
                buffer_kiri += char
        input_str = buffer_kiri.split("+")
        string_matrix_a.append(input_str)

    for element in string_matrix_b:
        matches = re.findall(r'(-?\d+)i', element)
        if matches:
            real_part = 0
            imaginary_part = int(matches[0])
            matrix_b.append(complex(real_part, imaginary_part))
        else:
            matrix_b.append(element)
    
    angka = []
    for element in string_matrix_b:
        if 'i' in element:
            extracted_element = 0
        else:
            matches = re.findall(r'-?\d+', element)
            extracted_element = int(matches[0])
        angka.append(extracted_element)
        
    ima = []
    for element in string_matrix_b:
        if "-i" in element :
            element = element.replace("-i", "-1i")
        matches = re.findall(r'(?<!\d)i', element)
        matches1 = re.findall(r'(-?\d+)i', element)
        if matches:
            real_part = 0
            imaginary_part = 1
        elif matches1:
            real_part = 0
            imaginary_part = int(matches1[0])
        else:
            real_part = 0
            imaginary_part = 0
        extracted_element = complex(real_part, imaginary_part)
        ima.append(extracted_element)
        
    angka = np.array(angka)
    ima = np.array(ima)
    for i in range(len(angka)):
        ima[i] = angka[i] + ima[i]
    matrix_b = ima
    
    numbers = []
    for row in string_matrix_a:
        extracted_sublist = []
        for i in range(len(row)):
            if 'i' in row[i]:
                extracted_element = 0
            else:
                for _ in range(len(row[i])):
                    temp = re.sub(r'(?<![0-9])[a-zA-Z]', r'1\g<0>', row[i])
                    row[i] =  temp
                matches = re.findall(r'-?\d+', row[i])
                extracted_element = int(matches[0])
            extracted_sublist.append(extracted_element)
        numbers.append(extracted_sublist)

    imaginary = []
    for row in string_matrix_a:
        extracted_sublist = []
        for element in row:
            if "-i" in element :
                element = element.replace("-i", "-1i")
            matches = re.findall(r'(?<!\d)i', element)
            matches1 = re.findall(r'(-?\d+)i', element)
            if matches:
                real_part = 0
                imaginary_part = 1
            elif matches1:
                real_part = 0
                imaginary_part = int(matches1[0])
            else:
                real_part = 0
                imaginary_part = 0
            extracted_element = complex(real_part, imaginary_part)
            extracted_sublist.append(extracted_element)
        imaginary.append(extracted_sublist)

    numbers = np.array(numbers)
    imaginary = np.array(imaginary)

    for i in range(len(numbers)):
        imaginary[i] = numbers[i] + imaginary[i]

    matrix_a = imaginary
    return matrix_a, matrix_b

def round_complex(complex_number, digit):
    real = math.ceil(complex_number.real * (10 ** digit)) / (10 ** digit)
    imag = math.ceil(complex_number.imag * (10 ** digit)) / (10 ** digit)
    return complex(real, imag)
        
def determine_solution(matrix_a, matrix_b):
    row, cols = matrix_a.shape
    if row != cols:
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
    
    try:
        print("Masukkan matriks A:")
        matrix_a = input_matrix(n, float); clear_txt(); fileProcessing('Matriks A :', matrix_a); file_string("")
        print("Masukkan matriks B:")
        matrix_b = input_matrix(n, float); fileProcessing('Matriks B :', matrix_b); 
    except Exception:
        print("Terdapat kesalahan format yang Anda masukkan"); file_string('Terdapat kesalahan format yang Anda masukkan')
        return
    
    try:
        y = determine_solution(matrix_a, matrix_b)
        print(f"\nJenis Solusi: \n{y}\n"); file_string("\nJenis solusi :"); file_string(y); file_string("")
        if y == "Unique solution":
            x = np.linalg.solve(matrix_a, matrix_b)
            round_x = np.round(x, decimals=3)
            print("Hasilnya adalah:"); file_string('Hasilnya adalah :')
            print(round_x); file_matrix(round_x)
    except Exception:
        print("Tidak dapat melakukan operasi"); file_string('Tidak dapat melakukan operasi')
        return
    
def solve_equation():
    n = int(input("Masukkan jumlah persamaan: "))
    print("Masukkan Persamaan"); clear_txt(); file_string('Persamaan :')
    try:
        matrix_a, matrix_b = input_equation(n)
    except Exception:
        print("Terdapat kesalahan format yang Anda masukkan"); file_string('Terdapat kesalahan format yang Anda masukkan')
        return

    try:
        print("\nMatrix A: \n", matrix_a); fileProcessing("\nMatrix A:", matrix_a)
        print("\nMatrix B: \n", matrix_b); fileProcessing("\nMatrix B:", matrix_b)
        y = determine_solution(matrix_a, matrix_b)
        print(f"\nJenis Solusi: \n{y}\n"); file_string("\nJenis solusi :"); file_string(y)
        if y == "Unique solution":
            x = np.linalg.solve(matrix_a, matrix_b)
            round_x = np.round(x, decimals=3)
            print("Hasilnya adalah:"); file_string('\nHasilnya adalah:')
            print(round_x); file_string(round_x)
    except Exception:
        print("Tidak dapat melakukan operasi"); file_string('Tidak dapat melakukan operasi')

def characteristicPolynomial_eigenvalue_eigenvector():
    n = int(input("Masukkan jumlah baris: "))
    print("Masukkan matriks:"); clear_txt()
    try:
        matrix_input = input_matrix(n, float); fileProcessing('Matriks', matrix_input)
    except Exception:
        print("Terdapat kesalahan format yang Anda masukkan"); file_string('Terdapat kesalahan format yang Anda masukkan')
        return
    
    characteristic_polynomial = np.poly(matrix_input)
    print("\nKarakteristik Polinomial: \n", characteristic_polynomial); fileProcessing('Karakteristik Polinomial:', characteristic_polynomial)
    
    eigenvalue, eigenvector = np.linalg.eig(matrix_input)
    print("\nEigenvalue: \n", eigenvalue); fileProcessing('Eigenvalue : ',eigenvalue)
    print("\nEigenvector: \n", eigenvector); fileProcessing('Eigenvector', eigenvector)
    
    A = matrix_input
    if len(eigenvalue) == A.shape[0]:
        print("\nMatrix A dapat didiagonalisasi"); file_string('Matrix A dapat didiagonalisasi')
        print("Sehingga"); file_string('Sehingga')
        P = eigenvector
        P_inv = np.linalg.inv(P)
        print("\nMatrix P:\n", P); fileProcessing('Matriks P :', P)
        print("\nMatrix P inverse:\n", P_inv); fileProcessing('Matriks P inverse :', P_inv)
    else:
        print("Matrix A tidak dapat didiagonalisasi, sehingga matrix P dan inversenya tidak dapat dicari"); file_string('Matrix A tidak dapat didiagonalisasi, sehingga matrix P dan inversenya tidak dapat dicari')
    
def svd():
    n = int(input("Masukkan jumlah baris: "))
    print("Masukkan matriks:"); clear_txt()
    try:
        matrix_a = input_matrix(n, float); fileProcessing('Matriks :', matrix_a); file_string("")
    except Exception:
        print("Terdapat kesalahan format yang Anda masukkan"); file_string('Terdapat kesalahan format yang Anda masukkan')
        return
    
    try:
        U, S, V = np.linalg.svd(matrix_a)
        round_U = np.round(U, decimals=3)
        round_S = np.round(S, decimals=3)
        round_V = np.round(V, decimals=3)
        
        print("\nMatriks U: \n", round_U); fileProcessing('Matriks U :', round_U); file_string("")
        print("\nMatriks singular values: \n", round_S); fileProcessing('Matriks singular values:', round_S); file_string("")
        print("\nMatriks V: \n", round_V); fileProcessing('Matriks V :', round_V); file_string("")
    except Exception:
        print("Tidak dapat melakukan operasi"); file_string('Tidak dapat melakukan operasi')
        return
    
def spl_complex_svd():
    n = int(input("Masukkan jumlah persamaan: "))
    m = int(input("Masukkan jumlah variabel: "))
    print("Masukkan persamaan:"); clear_txt(); file_string('Persamaan :')
    try:
        matrix_a, matrix_b = input_complex(n)
    except Exception:
        print("Terdapat kesalahan format yang Anda masukkan"); file_string('Terdapat kesalahan format yang Anda masukkan')
        return
    
    try:
        U, s, Vh = np.linalg.svd(matrix_a)
        s_inv = np.zeros_like(matrix_a.T, dtype=complex)
        s_inv[:len(s), :len(s)] = np.diag(1 / s)
        x = Vh.T.conj() @ s_inv @ U.T.conj() @ matrix_b

        print("\nMatrix A: \n", matrix_a); fileProcessing('\nMatriks A: ', matrix_a)
        print("\nMatrix B: \n", matrix_b); fileProcessing('\nMatriks B: ', matrix_b)
        print("\nHasilnya adalah:"); file_string('\nHasilnya adalah:')
        for i in range(m):
            round = round_complex(x[i], 3)
            print(f"x{i+1} = {round}"); 
    except Exception:
        print("Tidak dapat melakukan operasi"); file_string('Tidak dapat melakukan operasi')
        return
    
print("Kalkulator Matriks")

while True:
    os.system('cls' if os.name == 'nt' else 'clear')
    print("\nPilih operasi:")
    print("1. Mencari solusi persamaan linier(input matrix)")
    print("2. Mencari solusi persamaan linier(input persamaan)")
    print("3. Mencari karakteristik polinomial, eigenvalue, eigenvector")
    print("4. Mencari SVD")
    print("5. SPL Complex dengan SVD")
    print("0. Keluar")
    choice = int(input("Masukkan pilihan: "))

    if choice == 1:
        solve_matrix()
        input("\nPress Enter to continue...")
    elif choice == 2:
        solve_equation()
        input("\nPress Enter to continue...")
    elif choice == 3:
        characteristicPolynomial_eigenvalue_eigenvector()
        input("\nPress Enter to continue...")
    elif choice == 4:
        svd()
        input("\nPress Enter to continue...")
    elif choice == 5:
        spl_complex_svd()
        input("\nPress Enter to continue...")
    elif choice == 0:
        break
    else:
        print("Pilihan tidak valid. Silakan coba lagi.")
