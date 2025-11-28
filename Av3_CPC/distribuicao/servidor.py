# server.py
import socket
import threading
from typing import Tuple
import numpy as np

from .utilidades import send_obj, recv_obj


class MatrixWorkerServer:
    """
    Servidor de 'worker' que recebe submatriz A e matriz B, calcula subC = subA @ B
    e devolve o resultado para o cliente.

    Simula uma máquina de computação distribuída.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 5000, backlog: int = 5):
        self.host = host
        self.port = port
        self.backlog = backlog
        self._sock = None

    def _handle_client(self, conn: socket.socket, addr: Tuple[str, int]):
        """
        Atende uma conexão de cliente: recebe dados, multiplica e devolve resultado.
        """
        try:
            request = recv_obj(conn)
            subA = np.array(request["subA"])
            B = np.array(request["B"])

            # Aqui ocorre a "computação" do worker:
            subC = subA.dot(B)

            response = {"subC": subC}
            send_obj(conn, response)
        except Exception as e:
            # Em caso de erro, devolve uma mensagem para o cliente (opcional)
            try:
                send_obj(conn, {"error": str(e)})
            except Exception:
                pass
        finally:
            conn.close()

    def start(self):
        """
        Inicia o servidor e aceita conexões indefinidamente.
        """
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.bind((self.host, self.port))
        self._sock.listen(self.backlog)
        print(f"[SERVER] Escutando em {self.host}:{self.port}")

        try:
            while True:
                conn, addr = self._sock.accept()
                print(f"[SERVER] Conexão de {addr}")
                t = threading.Thread(
                    target=self._handle_client, args=(conn, addr), daemon=True
                )
                t.start()
        except KeyboardInterrupt:
            print("\n[SERVER] Encerrando servidor (Ctrl+C recebido).")
        finally:
            self._sock.close()
