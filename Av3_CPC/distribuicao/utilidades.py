#utilidades
import struct
import pickle
import socket


def send_obj(sock: socket.socket, obj: object) -> None:
    """
    Serializa um objeto Python e envia com cabeçalho de 4 bytes (tamanho).
    """
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    header = struct.pack('!I', len(data))  # 4 bytes, inteiro sem sinal, big-endian
    sock.sendall(header + data)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    """
    Lê exatamente n bytes do socket (ou dispara erro se a conexão fechar antes).
    """
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket fechado enquanto recebia dados")
        buf.extend(chunk)
    return bytes(buf)


def recv_obj(sock: socket.socket):
    """
    Lê 4 bytes de tamanho + payload e desserializa de volta para objeto Python.
    """
    header = _recv_exact(sock, 4)
    (length,) = struct.unpack('!I', header)
    data = _recv_exact(sock, length)
    return pickle.loads(data)