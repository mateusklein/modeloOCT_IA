import os
from PIL import Image

def cortar_e_redimensionar_imagem(imagem, largura_desejada, altura_desejada):
    largura_original, altura_original = imagem.size

    # Calcula as coordenadas para cortar a imagem a partir do centro
    esquerda = (largura_original - largura_desejada) // 2
    superior = (altura_original - altura_desejada) // 2
    direita = esquerda + largura_desejada
    inferior = superior + altura_desejada

    # Corta a imagem
    imagem_cortada = imagem.crop((esquerda, superior, direita, inferior))

    # Redimensiona a imagem
    imagem_redimensionada = imagem_cortada.resize((largura_desejada, altura_desejada))

    return imagem_redimensionada

# Função para percorrer todas as pastas e redimensionar as imagens
def redimensionar_imagens_em_pastas(principal_dir):
    # Loop através das pastas test, train e val
    for pasta_principal in ['test', 'train', 'val']:
        pasta_principal_path = os.path.join(principal_dir, pasta_principal)
        pasta_destino_path = os.path.join(principal_dir, pasta_principal + '_new')
        os.makedirs(pasta_destino_path, exist_ok=True)  # Cria a pasta de destino se ela não existir
        
        # Loop através das pastas numeradas de 0 a 3
        for i in range(4):
            cont = 1
            pasta_numerada_path = os.path.join(pasta_principal_path, str(i))
            pasta_destino_numerada_path = os.path.join(pasta_destino_path, str(i))
            os.makedirs(pasta_destino_numerada_path, exist_ok=True)  # Cria a pasta de destino se ela não existir
            # Loop através de cada imagem na pasta
            listdiros = os.listdir(pasta_numerada_path)
            lenlistdiros = len(listdiros)
            for filename in listdiros:
                # Carrega a imagem
                imagem_path = os.path.join(pasta_numerada_path, filename)
                imagem_original = Image.open(imagem_path)

                # Dimensões desejadas
                largura_desejada = 512
                altura_desejada = 496

                # Corta e redimensiona a imagem
                imagem_cortada_redimensionada = cortar_e_redimensionar_imagem(imagem_original, largura_desejada, altura_desejada)

                # Salva a imagem resultante no diretório de destino
                caminho_destino = os.path.join(pasta_destino_numerada_path, filename.replace('.jpeg', '.jpeg'))
                imagem_cortada_redimensionada.save(caminho_destino)
                print("imagem", cont, "/",  lenlistdiros, "preprocessada")
                cont+=1

# Diretório principal onde estão as pastas test, train e val
principal_dir = 'imagens_OCT'

# Chama a função para redimensionar imagens em todas as pastas
redimensionar_imagens_em_pastas(principal_dir)