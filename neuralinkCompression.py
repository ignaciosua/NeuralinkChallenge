import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Flatten, Dense, Reshape
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf  # Para manejar archivos WAV
import gzip

# ============================================
# 1. Configuración de Directorios y Parámetros
# ============================================

# Directorios
dataset_dir = 'data'        # Carpeta que contiene los archivos WAV de entrada
inference_dir = 'inference' # Carpeta donde se guardarán los resultados de la inferencia
os.makedirs(dataset_dir, exist_ok=True)
os.makedirs(inference_dir, exist_ok=True)

# Parámetros
sample_rate = 22050          # Frecuencia de muestreo para cargar y guardar WAV
fixed_length_seconds = 5     # Duración fija en segundos para todas las señales
fixed_length = fixed_length_seconds * sample_rate  # Número fijo de muestras
latent_dim = 128              # Dimensión del espacio latente
epochs = 100                  # Número de épocas para el entrenamiento
batch_size = 32              # Tamaño de batch para el entrenamiento

# Ajustar fixed_length para que sea divisible por 4
if fixed_length % 4 != 0:
    padding = 4 - (fixed_length % 4)
    fixed_length += padding
    print(f"Fixed_length ajustado a {fixed_length} para ser divisible por 4.")

# ============================================
# 2. Funciones Auxiliares
# ============================================

def load_wav_files(dataset_dir, sample_rate=22050, fixed_length=None):
    """
    Carga y preprocesa archivos WAV desde un directorio, asegurando una longitud fija.

    Parámetros:
    - dataset_dir: str, ruta al directorio que contiene los archivos WAV.
    - sample_rate: int, frecuencia de muestreo para cargar los archivos.
    - fixed_length: int o None, número fijo de muestras para todas las señales. Si es None, se usa la longitud máxima encontrada.

    Retorna:
    - data_padded: np.array, array de forma (num_samples, fixed_length, 1)
    """
    data = []
    lengths = []

    # Cargar todas las señales
    for file in os.listdir(dataset_dir):
        if file.lower().endswith('.wav'):
            file_path = os.path.join(dataset_dir, file)
            y, sr = librosa.load(file_path, sr=sample_rate)  # Cargar toda la señal
            data.append(y)
            lengths.append(len(y))

    if not data:
        raise ValueError(f"No se encontraron archivos WAV en el directorio: {dataset_dir}")

    if fixed_length is None:
        fixed_length = max(lengths)
        print(f"Longitud fija no especificada. Usando la longitud máxima detectada: {fixed_length} muestras ({fixed_length / sample_rate:.2f} segundos)")
    else:
        print(f"Usando longitud fija especificada: {fixed_length} muestras ({fixed_length / sample_rate:.2f} segundos)")

    # Padear o truncar todas las señales a la longitud fija
    data_padded = []
    for y in data:
        if len(y) < fixed_length:
            y = np.pad(y, (0, fixed_length - len(y)), 'constant')
        else:
            y = y[:fixed_length]
        # Normalizar entre -1 y 1
        if np.max(np.abs(y)) == 0:
            y = y.astype('float32')
        else:
            y = y.astype('float32') / np.max(np.abs(y))
        data_padded.append(y)

    data_padded = np.array(data_padded)
    data_padded = data_padded[..., np.newaxis]  # Añadir dimensión de canal
    return data_padded

def calculate_sample_difference(original, reconstructed):
    """
    Calcula la diferencia directa por muestra entre la señal original y la reconstruida.

    Parámetros:
    - original: np.array, señal original con forma (length, 1)
    - reconstructed: np.array, señal reconstruida con forma (length, 1)

    Retorna:
    - difference: np.array, diferencia por muestra
    """
    difference = original - reconstructed
    return difference

def save_difference_as_wav(difference, file_path, sample_rate=22050):
    """
    Guarda la diferencia por muestra como un archivo WAV.

    Parámetros:
    - difference: np.array, diferencia por muestra
    - file_path: str, ruta donde se guardará el archivo WAV de diferencias
    - sample_rate: int, frecuencia de muestreo para el archivo WAV
    """
    # Normalizar la señal de diferencia
    max_diff = np.max(np.abs(difference))
    if max_diff != 0:
        difference_normalized = difference / max_diff
    else:
        difference_normalized = difference

    # Guardar como WAV
    sf.write(file_path, difference_normalized.flatten(), samplerate=sample_rate)
    print(f"Archivo WAV de diferencias guardado en: {file_path}\n")

def save_compressed_latent_vector(latent_vector, file_path):
    """
    Reduce la precisión del vector latente a float16 y lo comprime usando Gzip.

    Parámetros:
    - latent_vector: np.array, el vector latente generado por el codificador.
    - file_path: str, ruta donde se guardará el vector latente comprimido.
    """
    # Reducir la precisión a float16
    latent_fp16 = latent_vector.astype(np.float16)

    # Comprimir usando Gzip
    with gzip.open(file_path, 'wb') as f:
        f.write(latent_fp16.tobytes())

    # Calcular el tamaño en bits
    file_size_bytes = os.path.getsize(file_path)
    file_size_bits = file_size_bytes * 8
    print(f"Vector latente comprimido y guardado en '{file_path}'.")
    print(f"Tamaño del vector latente comprimido: {file_size_bits} bits ({file_size_bytes} bytes).\n")

def load_compressed_latent_vector(file_path, dtype=np.float16, shape=(1, 64)):
    """
    Descomprime y carga el vector latente desde un archivo comprimido.

    Parámetros:
    - file_path: str, ruta del archivo comprimido.
    - dtype: tipo de dato de los elementos del vector latente.
    - shape: tuple, forma del vector latente.

    Retorna:
    - latent_decompressed: np.array, el vector latente descomprimido.
    """
    with gzip.open(file_path, 'rb') as f:
        latent_compressed = f.read()
    latent_decompressed = np.frombuffer(latent_compressed, dtype=dtype).reshape(shape)
    return latent_decompressed

def create_and_save_difference_plot(difference, num_plots=1, save_path='difference_plot.png'):
    """
    Crea y guarda un gráfico que muestra las diferencias por muestra.

    Parámetros:
    - difference: np.array, diferencia por muestra entre original y reconstruida.
    - num_plots: int, número de gráficos de diferencia a crear (se muestra solo uno).
    - save_path: str, ruta donde se guardará el gráfico de diferencias.
    """
    plt.figure(figsize=(15, 5 * num_plots))
    for i in range(num_plots):
        ax = plt.subplot(num_plots, 1, i + 1)
        plt.plot(difference.flatten())
        plt.title("Diferencias por Muestra entre Original y Reconstruida")
        plt.xlabel("Muestras")
        plt.ylabel("Amplitud")
    plt.tight_layout()
    # Mostrar el gráfico
    # plt.show()  # Puedes comentar esta línea si no deseas que se muestre el gráfico

    # Guardar el gráfico de diferencias
    plt.savefig(save_path)
    plt.close()
    print(f"Gráfico de diferencias guardado como '{save_path}'.\n")

def calculate_bit_match_percentage(original, reconstructed):
    """
    Calcula el porcentaje de coincidencia bit a bit entre la señal original y la reconstruida.

    Nota: Esta función es opcional y puede ser eliminada si no es necesaria.

    Parámetros:
    - original: np.array, señal original con forma (length,)
    - reconstructed: np.array, señal reconstruida con forma (length,)

    Retorna:
    - match_percentage: float, porcentaje de coincidencia bit a bit
    """
    # Convertir a 16-bit PCM
    original_int = np.int16(original.flatten() * 32767)
    reconstructed_int = np.int16(reconstructed.flatten() * 32767)

    # Convertir a bits
    original_bits = np.unpackbits(original_int.view(np.uint8))
    reconstructed_bits = np.unpackbits(reconstructed_int.view(np.uint8))

    # Asegurar que los arrays tengan la misma longitud
    min_length = min(len(original_bits), len(reconstructed_bits))
    original_bits = original_bits[:min_length]
    reconstructed_bits = reconstructed_bits[:min_length]

    matches = np.sum(original_bits == reconstructed_bits)
    total_bits = original_bits.size
    match_percentage = (matches / total_bits) * 100
    return match_percentage

# ============================================
# 3. Definición del Modelo Autoencoder Convolucional 1D
# ============================================

def build_autoencoder(input_length, latent_dim=64):
    """
    Construye y retorna un Autoencoder Convolucional 1D.

    Parámetros:
    - input_length: int, número total de muestras por audio (longitud fija).
    - latent_dim: int, dimensión del espacio latente.

    Retorna:
    - autoencoder: Model, modelo autoencoder completo.
    - encoder: Model, modelo codificador.
    - decoder: Model, modelo decodificador.
    """
    input_shape = (input_length, 1)
    input_layer = Input(shape=input_shape)

    # Codificación
    x = Conv1D(16, 9, activation='relu', padding='same')(input_layer)
    x = MaxPooling1D(2, padding='same')(x)  # Reducción de la longitud
    x = Conv1D(8, 9, activation='relu', padding='same')(x)
    x = MaxPooling1D(2, padding='same')(x)  # Reducción de la longitud
    x = Flatten()(x)
    latent = Dense(latent_dim, activation='relu')(x)  # Espacio latente reducido

    # Decodificación
    x = Dense((input_length // 4) * 8, activation='relu')(latent)
    x = Reshape((input_length // 4, 8))(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(8, 9, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(16, 9, activation='relu', padding='same')(x)
    output_layer = Conv1D(1, 9, activation='sigmoid', padding='same')(x)

    # Modelo Autoencoder
    autoencoder = Model(input_layer, output_layer)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # Modelo Codificador
    encoder = Model(inputs=input_layer, outputs=latent)

    # Modelo Decodificador
    decoder_input = Input(shape=(latent_dim,))
    x = autoencoder.layers[-7](decoder_input)  # Dense
    x = autoencoder.layers[-6](x)  # Reshape
    x = autoencoder.layers[-5](x)  # UpSampling1D
    x = autoencoder.layers[-4](x)  # Conv1D(8)
    x = autoencoder.layers[-3](x)  # UpSampling1D
    x = autoencoder.layers[-2](x)  # Conv1D(16)
    decoder_output = autoencoder.layers[-1](x)  # Conv1D(1)
    decoder = Model(decoder_input, decoder_output)

    return autoencoder, encoder, decoder

# ============================================
# 4. Entrenamiento del Modelo
# ============================================

print("Cargando y preprocesando los datos WAV...")
x_train = load_wav_files(dataset_dir, sample_rate=sample_rate, fixed_length=fixed_length)
x_test = load_wav_files(dataset_dir, sample_rate=sample_rate, fixed_length=fixed_length)
print(f"Datos cargados y preprocesados correctamente.\nNúmero de muestras de entrenamiento: {x_train.shape[0]}\nNúmero de muestras de prueba: {x_test.shape[0]}\n")

print("Definiendo la arquitectura del Autoencoder 1D...")
autoencoder, encoder, decoder = build_autoencoder(input_length=fixed_length, latent_dim=latent_dim)
autoencoder.summary()
print("\nArquitectura del Autoencoder definida correctamente.\n")

print("Iniciando el entrenamiento del Autoencoder...")
autoencoder.fit(
    x_train, x_train,
    epochs=epochs,
    batch_size=batch_size,  # Ajusta según tu hardware
    shuffle=True,
    validation_data=(x_test, x_test)
)
print("Entrenamiento completado.\n")

# Guardar los modelos entrenados
autoencoder.save('autoencoder_compressed_model.keras')
encoder.save('encoder_compressed_model.keras')
decoder.save('decoder_compressed_model.keras')
print("Modelos autoencoder, codificador y decodificador guardados correctamente.\n")

# ============================================
# 5. Proceso de Inferencia Optimizado
# ============================================

def inference_process_compressed_optimized(encoder_model, decoder_model, original_signal, inference_folder, sample_rate=22050):
    """
    Codifica una señal, guarda el vector latente comprimido, decodifica, guarda las señales,
    calcula y guarda las diferencias, y verifica la reconstrucción.

    Parámetros:
    - encoder_model: Model, el modelo codificador.
    - decoder_model: Model, el modelo decodificador.
    - original_signal: np.array, la señal original de entrada (length, 1).
    - inference_folder: str, la carpeta donde se guardarán los resultados de la inferencia.
    - sample_rate: int, frecuencia de muestreo para guardar los archivos WAV.

    Retorna:
    - reconstructed_signal: np.array, la señal reconstruida.
    - is_equal: bool, si la señal reconstruida es igual a la original dentro de una tolerancia.
    """
    print("Iniciando el proceso de inferencia optimizado...\n")
    os.makedirs(inference_folder, exist_ok=True)

    # Asegurarse de que la señal tenga la forma correcta
    if original_signal.ndim == 1:
        signal = original_signal.reshape(1, -1, 1)
    elif original_signal.ndim == 2:
        signal = original_signal.reshape(1, -1, 1)
    else:
        raise ValueError("La señal debe tener 1 o 2 dimensiones.")

    # Codificar la señal
    print("Codificando la señal...")
    latent = encoder_model.predict(signal)

    # Guardar el vector latente comprimido
    latent_compressed_path = os.path.join(inference_folder, 'latent_vector_compressed.gz')
    save_compressed_latent_vector(latent, latent_compressed_path)

    # Decodificar el vector latente comprimido
    print("Decodificando el vector latente comprimido...")
    latent_loaded = load_compressed_latent_vector(latent_compressed_path, dtype=np.float16, shape=(1, latent_dim))

    # Reconstruir la señal usando el vector latente descomprimido
    reconstructed = decoder_model.predict(latent_loaded)

    # Guardar la señal original como gráfico y WAV
    original_path_png = os.path.join(inference_folder, 'original_signal.png')
    original_path_wav = os.path.join(inference_folder, 'original_signal.wav')
    plt.figure(figsize=(15, 5))
    librosa.display.waveshow(original_signal.flatten(), sr=sample_rate)
    plt.title("Señal Original")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.savefig(original_path_png)
    plt.close()
    print(f"Señal original guardada en '{original_path_png}'.")

    # Guardar la señal original como WAV
    sf.write(original_path_wav, original_signal.flatten(), samplerate=sample_rate)
    print(f"Señal original guardada como WAV en '{original_path_wav}'.\n")

    # Guardar la señal reconstruida como gráfico y WAV
    reconstructed_signal = reconstructed.reshape(-1)
    reconstructed_path_png = os.path.join(inference_folder, 'reconstructed_signal.png')
    reconstructed_path_wav = os.path.join(inference_folder, 'reconstructed_signal.wav')
    plt.figure(figsize=(15, 5))
    librosa.display.waveshow(reconstructed_signal, sr=sample_rate)
    plt.title("Señal Reconstruida")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.savefig(reconstructed_path_png)
    plt.close()
    print(f"Señal reconstruida guardada en '{reconstructed_path_png}'.")

    # Guardar la señal reconstruida como WAV
    sf.write(reconstructed_path_wav, reconstructed_signal, samplerate=sample_rate)
    print(f"Señal reconstruida guardada como WAV en '{reconstructed_path_wav}'.\n")

    # Calcular el porcentaje de coincidencia bit a bit (opcional)
    print("Calculando el porcentaje de coincidencia bit a bit...")
    bit_match_percentage = calculate_bit_match_percentage(original_signal, reconstructed_signal)
    print(f"Porcentaje de coincidencia bit a bit: {bit_match_percentage:.2f}%\n")

    # Calcular las diferencias por muestra
    difference = calculate_sample_difference(original_signal.flatten(), reconstructed_signal)

    # Guardar las diferencias como WAV
    difference_wav_path = os.path.join(inference_folder, 'difference.wav')
    save_difference_as_wav(difference, difference_wav_path, sample_rate)

    # Crear y guardar el gráfico de diferencias
    difference_plot_path = os.path.join(inference_folder, 'difference_plot.png')
    create_and_save_difference_plot(difference, num_plots=1, save_path=difference_plot_path)

    # Verificar si la señal reconstruida es igual a la original dentro de una tolerancia
    tolerance = 1e-5
    is_equal = np.allclose(original_signal.flatten(), reconstructed_signal, atol=tolerance)
    print(f"¿La señal reconstruida es igual a la original dentro de la tolerancia? {'Sí' if is_equal else 'No'}\n")

    return reconstructed_signal, is_equal

# ============================================
# 6. Ejecución de Inferencia en Todas las Señales de Prueba
# ============================================

print("Iniciando el proceso de inferencia para todas las señales de prueba...\n")

for test_index in range(x_test.shape[0]):
    test_signal = x_test[test_index].reshape(-1, 1)
    print(f"Realizando inferencia para la señal de prueba con índice {test_index}.\n")

    # Crear una carpeta específica para cada señal de prueba
    specific_inference_dir = os.path.join(inference_dir, f'signal_{test_index}')
    os.makedirs(specific_inference_dir, exist_ok=True)

    reconstructed_signal, is_equal = inference_process_compressed_optimized(
        encoder_model=encoder,
        decoder_model=decoder,
        original_signal=test_signal,
        inference_folder=specific_inference_dir,
        sample_rate=sample_rate  # Asegúrate de que esto coincida con la frecuencia de muestreo usada al cargar
    )

    print(f"Inferencia completada para la señal de prueba con índice {test_index}. ¿Iguales? {'Sí' if is_equal else 'No'}\n")

# ============================================
# 7. Visualización de las Señales
# ============================================

# Visualizar las Señales para Cada Señal de Prueba
for test_index in range(x_test.shape[0]):
    specific_inference_dir = os.path.join(inference_dir, f'signal_{test_index}')

    original_path_png = os.path.join(specific_inference_dir, 'original_signal.png')
    reconstructed_path_png = os.path.join(specific_inference_dir, 'reconstructed_signal.png')
    difference_plot_path = os.path.join(specific_inference_dir, 'difference_plot.png')

    if os.path.exists(original_path_png) and os.path.exists(reconstructed_path_png) and os.path.exists(difference_plot_path):
        # Cargar las imágenes de las señales
        original_img = plt.imread(original_path_png)
        reconstructed_img = plt.imread(reconstructed_path_png)
        difference_img = plt.imread(difference_plot_path)

        # Mostrar las imágenes
        plt.figure(figsize=(18, 6))

        # Señal original
        plt.subplot(1, 3, 1)
        plt.imshow(original_img)
        plt.title("Señal Original")
        plt.axis("off")

        # Señal reconstruida
        plt.subplot(1, 3, 2)
        plt.imshow(reconstructed_img)
        plt.title("Señal Reconstruida")
        plt.axis("off")

        # Diferencias por muestra
        plt.subplot(1, 3, 3)
        plt.imshow(difference_img)
        plt.title("Diferencias por Muestra")
        plt.axis("off")

        plt.tight_layout()
        plt.show()
    else:
        print(f"Archivos de visualización no encontrados para la señal de prueba con índice {test_index}. Asegúrate de que la inferencia se realizó correctamente.\n")
