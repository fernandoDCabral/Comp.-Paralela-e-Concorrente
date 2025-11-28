# run_server.py
import argparse
from distribuicao.servidor import MatrixWorkerServer


def main():
    parser = argparse.ArgumentParser(description="Servidor worker de multiplicação de matrizes.")
    parser.add_argument("--host", default="0.0.0.0", help="Endereço para escutar (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, required=True, help="Porta para escutar")
    args = parser.parse_args()

    server = MatrixWorkerServer(host=args.host, port=args.port)
    server.start()


if __name__ == "__main__":
    main()
