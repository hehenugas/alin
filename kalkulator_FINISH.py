import numpy as np
import re
import math
import os

def input_matrix(n, data_type):
    return np.array([input().split() for _ in range(n)], dtype=data_type)

def clear_txt():
    with open("hasil.txt",'w') as file:
        pass

def fileProcessing(message):
    with open("hasil.txt", 'a') as file:
        print(message, file=file)
        
def input_equation(n):
    string_matrix_a = []
    string_matrix_b = []
    for _ in range(n):
        input_str = input(); fileProcessing(input_str)
        input_str = input_str.replace(" ", "").split("=")
        buffer_kanan = input_str[1]
        string_matrix_b.append(buffer_kanan)
        buffer_kiri = ''
        for char in input_str[0]:
            if char == "-":
                if char == input_str[0][0]:
                    buffer_kiri += "-"
                else:
                    buffer_kiri += "+-"  
            else:
                buffer_kiri += char
        input_str = buffer_kiri.split("+")
        string_matrix_a.append(input_str)
    
    matrix = []
    for row in string_matrix_a:
        extracted_sublist = []
        for i in range(len(row)):
            for j in range(len(row[i])):
                temp = re.sub(r'(?<![0-9])[matrix-zA-Z]', r'1\g<0>', row[i]) 
                row[i] = temp
            matches = re.findall(r'-?\d+', row[i])
            extracted_element = int(matches[0])
            extracted_sublist.append(extracted_element)
        matrix.append(extracted_sublist)
        
    b = []
    for element in string_matrix_b:
        matches = re.findall(r'-?\d+', element)
        extracted_element = int(matches[0])
        b.append(extracted_element)
        
    matrix_a = np.array(matrix)
    matrix_b = np.array(b)
    return matrix_a, matrix_b

def input_complex(num_equations):
    matrix_a = []
    matrix_b = []

    string_matrix_a = []
    string_matrix_b = []
    for _ in range(num_equations):
        input_str = input(); fileProcessing(input_str)
        input_str = input_str.replace(" ", "").split("=")
        buffer_kanan = input_str[1]
        string_matrix_b.append(buffer_kanan)
        buffer_kiri = ''
        for char in input_str[0]:
            if char == "-":
                if char == input_str[0][0]:
                    buffer_kiri += "-"
                else:
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
                    temp = re.sub(r'(?<![0-9])[matrix-zA-Z]', r'1\g<0>', row[i])
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
        
def determine_solution(a, b):
    augmented_matrix = np.concatenate((a, b), axis=1)
    rank_a = np.linalg.matrix_rank(a)
    rank_augmented = np.linalg.matrix_rank(augmented_matrix)

    if rank_a == rank_augmented:
        if rank_a == a.shape[1]:
            return "Unique solution"
        else:
            return "Infinite solutions"
    else:
        return "No solution"

def solve_matrix():
    n = int(input("Masukkan jumlah baris: "))
    
    try:
        
        print(  "+----------------------------------------------------------------+\n"
                " - Jika terdapat matrix 1 2 3 \n"
                "                        4 5 6 \n"
                "                        7 8 9 \n"
                " - Inputkan :  \n"
                "       1(spasi)2(spasi)3 \n"
                "       4(spasi)5(spasi)6 \n"
                "       7(spasi)8(spasi)9 \n"
                " - Tidak boleh ada nilai yang kosong pada tiap baris \n"
                "+----------------------------------------------------------------+ \n"  
            )
        
        print("Masukkan matriks A:")
        matrix_a = input_matrix(n, float); clear_txt(); fileProcessing('\nMatriks A :'); fileProcessing(matrix_a)
        
        print(  "+----------------------------------------------------------------+\n"
                " - Untuk Matrix B, Jika terdapat matrix  \n"
                "       1 \n"
                "       2 \n"
                "       3 \n"
                " - Inputkan : \n"
                "       1(enter) \n"
                "       2(enter) \n"
                "       3(enter) \n"
                "+----------------------------------------------------------------+ \n"  
                )
        
        print("Masukkan matriks B:")
        matrix_b = input_matrix(n, float)
        
    except Exception:
        print("Terdapat kesalahan format yang Anda masukkan"); fileProcessing('\nTerdapat kesalahan format yang Anda masukkan')
        return
    
    try:
        y = determine_solution(matrix_a, matrix_b)
        print(f"\nJenis Solusi: \n{y}\n"); fileProcessing("\nJenis solusi :"); fileProcessing(y)
        if y == "Unique solution":
            x = np.linalg.solve(matrix_a, matrix_b)
            round_x = np.round(x, decimals=3)
            print("Hasilnya adalah:"); fileProcessing('\nHasilnya adalah :')
            print(round_x); fileProcessing(round_x)
        elif y == "Infinite solutions":
            x = np.linalg.lstsq(matrix_a, matrix_b, rcond=None)[0]
            print("Hasilnya dalam bentuk parameter: ")
            flat = np.squeeze(x).tolist()
            print(flat, "+ t *", np.linalg.pinv(matrix_a)[:, -1])
    except Exception:
        print("\nNo Solution"); fileProcessing('\nNo Solution')
        return
    
def solve_equation():
    
    print(  "+----------------------------------------------------------------------------+\n"
            " - Program ini sama seperti menu 1, namun menggunakan input persamaan langsung  \n"
            " - Variabel yang digunakan adalah x1, x2, x3 ... xn \n"
            " - contohnya jika terdapat persamaan \n"
            "       3x1 + 2x2 = 5 \n"
            " - maka inputkan saja persis seperti persamaan itu \n"
            "   yaitu: \n"
            "       3x1 + 2x2 = 5 : \n"
            " - jika salah satu koefisien tidak ada, maka tuliskan saja dengan angka 0 \n"
            "   misalnya terdapat 2 persamaan sebagai berikut \n"
            "       3x1 = 5 \n"
            "       4x1 + 2x2 = 20 \n"
            " - maka intputkan :"
            "       3x1 + 0x2 = 5 \n"
            "       4x1 + 2x2 = 20 \n"
            "+---------------------------------------------------------------------------+ \n"  
            )
    
    n = int(input("Masukkan jumlah persamaan: "))
    m = int(input("Masukkan Jumlah variabel: "))
    print("Masukkan Persamaan"); clear_txt(); fileProcessing('Persamaan :')
    try:
        matrix_a, matrix_b = input_equation(n)
    except Exception:
        print("Terdapat kesalahan format yang Anda masukkan"); fileProcessing('\nTerdapat kesalahan format yang Anda masukkan')
        return

    print(matrix_a); print(matrix_b)
    
    try:
        y = determine_solution(matrix_a, matrix_b)
        print(f"\nJenis Solusi: \n{y}\n"); fileProcessing("\nJenis solusi :"); fileProcessing(y)
        if y == "Unique solution":
            x = np.linalg.solve(matrix_a, matrix_b)
            round_x = np.round(x, decimals=3)
            print("Hasilnya adalah:"); fileProcessing('\nHasilnya adalah :')
            print(round_x); fileProcessing(round_x)
        elif y == "Infinite solutions":
            x = np.linalg.lstsq(matrix_a, matrix_b, rcond=None)[0]
            print("Hasilnya dalam bentuk parameter: ")
            flat = np.squeeze(x).tolist()
            print(flat, "+ t *", np.linalg.pinv(matrix_a)[:, -1])
    except Exception:
        print("\nNo Solution"); fileProcessing('\nNo Solution')
        return

def characteristicPolynomial_eigenvalue_eigenvector():
    n = int(input("Masukkan jumlah baris: "))
    
    print(  "+----------------------------------------------------------------+\n"
            " - Jika terdapat matrix 1 2 3 \n"
            "                        4 5 6 \n"
            "                        7 8 9 \n"
            " - Inputkan :  \n"
            "       1(spasi)2(spasi)3 \n"
            "       4(spasi)5(spasi)6 \n"
            "       7(spasi)8(spasi)9 \n"
            " - Tidak boleh ada nilai yang kosong pada tiap baris \n"
            "+----------------------------------------------------------------+ \n"  
          )
    
    print("Masukkan matriks:"); clear_txt()
    try:
        matrix_input = input_matrix(n, float); fileProcessing('\nMatriks :'); fileProcessing(matrix_input)
    except Exception:
        print("Terdapat kesalahan format yang Anda masukkan"); fileProcessing('\nTerdapat kesalahan format yang Anda masukkan')
        return
    
    characteristic_polynomial = np.poly(matrix_input)
    print("\nKarakteristik Polinomial: \n", characteristic_polynomial); fileProcessing('\nKarakteristik Polinomial:', ); fileProcessing(characteristic_polynomial)
    
    eigenvalue, eigenvector = np.linalg.eig(matrix_input)
    print("\nEigenvalue: \n", eigenvalue); fileProcessing('\nEigenvalue : '); fileProcessing(eigenvalue)
    print("\nEigenvector: \n", eigenvector); fileProcessing('\nEigenvector'); fileProcessing(eigenvector)
    
    A = matrix_input
    if len(eigenvalue) == A.shape[0]:
        print("\nMatrix A dapat didiagonalisasi"); fileProcessing('\nMatrix A dapat didiagonalisasi')
        print("Sehingga"); fileProcessing('Sehingga')
        P = eigenvector
        P_inv = np.linalg.inv(P)
        print("\nMatrix P:\n", P); fileProcessing('\nMatriks P :'); fileProcessing(P)
        print("\nMatrix P inverse:\n", P_inv); fileProcessing('Matriks P inverse :'); fileProcessing(P_inv)
    else:
        print("Matrix A tidak dapat didiagonalisasi, sehingga matrix P dan inversenya tidak dapat dicari"); fileProcessing('\nMatrix A tidak dapat didiagonalisasi, sehingga matrix P dan inversenya tidak dapat dicari')
    
def svd():
    n = int(input("Masukkan jumlah baris: "))
    
    print(  "+----------------------------------------------------------------+\n"
            " - Jika terdapat matrix 1 2 3 \n"
            "                        4 5 6 \n"
            "                        7 8 9 \n"
            " - Inputkan :  \n"
            "       1(spasi)2(spasi)3 \n"
            "       4(spasi)5(spasi)6 \n"
            "       7(spasi)8(spasi)9 \n"
            " - Tidak boleh ada nilai yang kosong pada tiap baris \n"
            "+----------------------------------------------------------------+ \n"  
          )
    
    
    print("Masukkan matriks:"); clear_txt()
    try:
        matrix_a = input_matrix(n, float); fileProcessing('\nMatriks:'); fileProcessing(matrix_a)
    except Exception:
        print("Terdapat kesalahan format yang Anda masukkan"); fileProcessing('\nTerdapat kesalahan format yang Anda masukkan')
        return
    
    try:
        U, S, V = np.linalg.svd(matrix_a)
        round_U = np.round(U, decimals=3)
        round_S = np.round(S, decimals=3)
        round_V = np.round(V, decimals=3)
        
        print("\nMatriks U: \n", round_U); fileProcessing('\nMatriks U:'); fileProcessing(round_S)
        print("\nMatriks singular values: \n", round_S); fileProcessing('\nMatriks singular values:'); fileProcessing(round_S)
        print("\nMatriks V: \n", round_V); fileProcessing('\nMatriks V:'); fileProcessing(round_V)
    except Exception:
        print("Tidak dapat melakukan operasi"); fileProcessing('\nTidak dapat melakukan operasi')
        return
    
def spl_complex_svd():
    
    print(  "+----------------------------------------------------------------------------+\n"
            " - Program ini menerima input persamaan langsung  \n"
            " - Variabel yang digunakan adalah x1, x2, x3 ... xn \n"
            " - contohnya jika terdapat persamaan complex \n"
            "       3x1 + 2ix2 = 5 \n"
            " - maka inputkan saja persis seperti persamaan itu \n"
            "   yaitu: \n"
            "       3x1 + 2ix2 = 5 : \n"
            " - jika salah satu koefisien tidak ada, maka tuliskan saja dengan angka 0 \n"
            "   misalnya terdapat 2 persamaan sebagai berikut \n"
            "       3x1 = 5 \n"
            "       4x1 + 2ix2 = 20 \n"
            " - maka intputkan :"
            "       3x1 + 0x2 = 5 \n"
            "       4x1 + 2ix2 = 20 \n"
            "+---------------------------------------------------------------------------+ \n"  
        )
    
    n = int(input("Masukkan jumlah persamaan: "))
    m = int(input("Masukkan jumlah variabel: "))
    print("Masukkan persamaan:"); clear_txt(); fileProcessing('Persamaan :')
    try:
        matrix_a, matrix_b = input_complex(n)
    except Exception:
        print("Terdapat kesalahan format yang Anda masukkan"); fileProcessing('\nTerdapat kesalahan format yang Anda masukkan')
        return
    
    try:
        U, s, Vh = np.linalg.svd(matrix_a)
        s_inv = np.zeros_like(matrix_a.T, dtype=complex)
        s_inv[:len(s), :len(s)] = np.diag(1 / s)
        x = Vh.T.conj() @ s_inv @ U.T.conj() @ matrix_b

        print("\nMatrix A: \n", matrix_a); fileProcessing('\nMatriks A: '); fileProcessing(matrix_a)
        print("\nMatrix B: \n", matrix_b); fileProcessing('\nMatriks B: '); fileProcessing(matrix_b)
        print("\nHasilnya adalah:"); fileProcessing('\nHasilnya adalah:')
        for i in range(m):
            round = round_complex(x[i], 3)
            output = f'x{i+1} = {round}'
            print(output); fileProcessing(output)
            
    except Exception:
        print("Tidak dapat melakukan operasi"); fileProcessing('\nTidak dapat melakukan operasi')
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
        os.system('cls' if os.name == 'nt' else 'clear')
        solve_matrix()
        input("\nPress Enter to continue...")
    elif choice == 2:
        os.system('cls' if os.name == 'nt' else 'clear')
        solve_equation()
        input("\nPress Enter to continue...")
    elif choice == 3:
        os.system('cls' if os.name == 'nt' else 'clear')
        characteristicPolynomial_eigenvalue_eigenvector()
        input("\nPress Enter to continue...")
    elif choice == 4:
        os.system('cls' if os.name == 'nt' else 'clear')
        svd()
        input("\nPress Enter to continue...")
    elif choice == 5:
        os.system('cls' if os.name == 'nt' else 'clear')
        spl_complex_svd()
        input("\nPress Enter to continue...")
    elif choice == 0:
        break
    else:
        print("Pilihan tidak valid. Silakan coba lagi.")
