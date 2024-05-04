import http.server
import socketserver

# Definindo o diretório onde o conteúdo está
diretorio = ""

# Definindo a porta em que o servidor será executado
porta = 80

# Configurando o manipulador do servidor
handler = http.server.SimpleHTTPRequestHandler

# Mudando para o diretório onde o conteúdo está
with socketserver.TCPServer(("", porta), handler) as httpd:
    print("Servidor rodando na porta:", porta)

    # Iniciando o servidor
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass

    httpd.server_close()
