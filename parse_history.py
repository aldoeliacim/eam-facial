import json

# Imprime la precisión y el error cuadrático medio de la raíz del decodificador
# obtenidos durante el entrenamiento del modelo en los datos de prueba,
# para cada tamaño de dominio

domain_sizes = [128, 256, 512, 1024]
class_metric = 'accuracy'
autor_metric = 'decoder_root_mean_squared_error'

def print_keys(data):
    print('Keys: [ ', end='')
    for k in data.keys():
        print(f'{k}, ', end='')
    print(']')

if __name__ == "__main__":
    for domain in domain_sizes:
        class_values = []
        autor_values = []

        suffix = '/model-classifier.json'
        filename = f'runs-{domain}{suffix}'
        # Abriendo el archivo JSON
        with open(filename, 'r') as f:
            data = json.load(f)
            history = data['history']
            # En cada dos, el primer elemento es el rastro del entrenamiento,
            # y se ignora. El segundo elemento contiene la métrica y
            # pérdida del clasificador
            for i in range(0, len(history), 2):
                if i + 1 < len(history):
                    class_values.append(history[i+1][class_metric])

        suffix = '/model-autoencoder.json'
        filename = f'runs-{domain}{suffix}'
        # Abriendo el archivo JSON
        with open(filename, 'r') as f:
            data = json.load(f)
            history = data['history']
            # En cada dos, el primer elemento es el rastro del entrenamiento,
            # y se ignora. El segundo elemento contiene la métrica y
            # pérdida del clasificador
            for i in range(0, len(history), 2):
                if i + 1 < len(history):
                    autor_values.append(history[i+1][autor_metric])

        print(f'Tamaño de dominio: {domain}. Se presentan las salidas métricas.')
        print(f'Fold\tClasificación\tAutoencoder')
        for j in range(len(class_values)):
            print(f'{j}\t{class_values[j]:.3f}\t\t{autor_values[j]:.3f}')

        class_value_mean = sum(class_values) / len(class_values)
        autor_value_mean = sum(autor_values) / len(autor_values)
        print(f'\nValor medio de precisión: {class_value_mean:.4f}, valor medio de rmse: {autor_value_mean:.4f}')
        print('\n')
