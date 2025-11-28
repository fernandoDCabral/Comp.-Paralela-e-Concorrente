# run_client.py
import argparse
from typing import List, Tuple
import time
import os

import numpy as np
from distribuicao.cliente import DistributedMatrixClient


def parse_servers(servers_args: List[str]) -> List[Tuple[str, int]]:
    """
    Converte a lista de strings no formato 'host:port' para
    uma lista de tuplas (host, port).
    """
    servers = []
    for s in servers_args:
        host, port = s.split(":")
        servers.append((host, int(port)))
    return servers


def save_matrix_to_txt(matrix: np.ndarray, filename: str) -> None:
    """
    Salva uma matriz NumPy em um arquivo .txt com formatação simples.
    """
    np.savetxt(filename, matrix, fmt="%d")
    print(f"Matriz salva em: {os.path.abspath(filename)}")


def main():
    parser = argparse.ArgumentParser(
        description="Cliente de multiplicação de matrizes distribuída."
    )
    parser.add_argument(
        "--servers",
        nargs="+",
        required=True,
        help='Lista de servidores no formato "host:port", ex: 127.0.0.1:5000 127.0.0.1:5001',
    )

    # Dimensões livres para A e B
    parser.add_argument("--a-rows", type=int, required=True, help="Número de linhas da matriz A")
    parser.add_argument("--a-cols", type=int, required=True, help="Número de colunas da matriz A")
    parser.add_argument("--b-rows", type=int, required=True, help="Número de linhas da matriz B")
    parser.add_argument("--b-cols", type=int, required=True, help="Número de colunas da matriz B")

    # Flag para salvar matrizes em .txt
    parser.add_argument(
        "--save",
        action="store_true",
        help="Se definido, salva as matrizes A, B e C em arquivos .txt.",
    )

    args = parser.parse_args()

    servers = parse_servers(args.servers)
    client = DistributedMatrixClient(servers=servers)

    a_rows = args.a_rows
    a_cols = args.a_cols
    b_rows = args.b_rows
    b_cols = args.b_cols

    # Verifica compatibilidade: cols(A) == rows(B)?
    if a_cols != b_rows:
        print("ERRO: Matrizes incompatíveis para multiplicação.")
        print(f"cols(A) = {a_cols}, rows(B) = {b_rows}. É necessário cols(A) == rows(B).")
        return

    # Geração das matrizes A e B de forma automática (aleatória)
    A = np.random.randint(-5, 6, size=(a_rows, a_cols))
    B = np.random.randint(-5, 6, size=(b_rows, b_cols))

    print("Matriz A:")
    print(A)
    print(f"Dimensão de A: {A.shape}\n")

    print("Matriz B:")
    print(B)
    print(f"Dimensão de B: {B.shape}\n")

    # ---- CÁLCULO DISTRIBUÍDO COM TIMER ----
    start_dist = time.perf_counter()
    C_dist = client.multiply(A, B)
    end_dist = time.perf_counter()
    elapsed_dist = end_dist - start_dist

    print("Resultado C (distribuído):")
    print(C_dist)
    print(f"Dimensão de C (distribuído): {C_dist.shape}")
    print(f"Tempo do cálculo distribuído: {elapsed_dist:.6f} segundos\n")

    # ---- CÁLCULO LOCAL COM TIMER ----
    start_local = time.perf_counter()
    C_local = A.dot(B)
    end_local = time.perf_counter()
    elapsed_local = end_local - start_local

    print("Resultado C (local numpy.dot):")
    print(C_local)
    print(f"Dimensão de C (local): {C_local.shape}")
    print(f"Tempo do cálculo local: {elapsed_local:.6f} segundos\n")

    # Comparação
    iguais = np.allclose(C_dist, C_local)
    print("Os resultados são iguais?", iguais)

    # ---- SALVAR EM .TXT (opcional) ----
    if args.save:
        print("\nSalvando matrizes em arquivos .txt...")
        save_matrix_to_txt(A, "matriz_A.txt")
        save_matrix_to_txt(B, "matriz_B.txt")
        save_matrix_to_txt(C_dist, "matriz_C.txt")


if __name__ == "__main__":
    main()
