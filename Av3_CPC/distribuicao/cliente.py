# client.py
from typing import List, Tuple
import socket
import threading

import numpy as np
from .utilidades import send_obj, recv_obj


class DistributedMatrixClient:
    """
    Cliente que distribui a multiplicação de matrizes entre vários servidores.

    - servers: lista de tuplas (host, port) representando as máquinas workers.
    """

    def __init__(self, servers: List[Tuple[str, int]]):
        if not servers:
            raise ValueError("É necessário informar pelo menos um servidor.")
        self.servers = servers

    @staticmethod
    def _split_matrix_rows(A: np.ndarray, n_chunks: int) -> List[np.ndarray]:
        """
        Divide a matriz A em n_chunks blocos de linhas, de forma balanceada.
        Retorna lista de submatrizes.
        """
        rows = A.shape[0]
        base = rows // n_chunks
        remainder = rows % n_chunks

        parts = []
        start = 0
        for i in range(n_chunks):
            size = base + (1 if i < remainder else 0)
            sub = A[start:start + size, :]
            parts.append(sub)
            start += size
        return parts

    def _worker_task(self, server, subA, B, results, idx):
        """
        Tarefa de cada thread: conecta em um servidor, envia subA e B e armazena subC.
        """
        host, port = server
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((host, port))
            payload = {"subA": subA, "B": B}
            send_obj(sock, payload)
            resp = recv_obj(sock)

        if "error" in resp:
            raise RuntimeError(f"Erro no servidor {server}: {resp['error']}")

        results[idx] = np.array(resp["subC"])

    def multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Multiplica A x B distribuindo pelas máquinas (servidores) configuradas.

        Retorna a matriz C.
        """
        if A.shape[1] != B.shape[0]:
            raise ValueError(
                f"Dimensões incompatíveis: A é {A.shape}, B é {B.shape}."
            )

        k = len(self.servers)
        subAs = self._split_matrix_rows(A, k)

        threads = []
        results = [None] * k

        for idx, subA in enumerate(subAs):
            t = threading.Thread(
                target=self._worker_task,
                args=(self.servers[idx], subA, B, results, idx),
                daemon=True,
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Reconstroi C empilhando os sub-resultados em ordem de linhas
        C = np.vstack(results)
        return C
