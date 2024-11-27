import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Flatten, Dense, Reshape
import matplotlib.pyplot as plt
import gzip
import librosa
import librosa.display
import soundfile as sf  # Para manejar archivos WAV

# ============================================
# 2. Configuración de Directorios y Parámetros
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
latent_dim = 64              # Dimensión del espacio latente
epochs = 50                  # Número de épocas para el entrenamiento
batch_size = 32              # Tamaño de batch para el entrenamiento

# Ajustar fixed_length para que sea divisible por 4
if fixed_length % 4 != 0:
    padding = 4 - (fixed_length % 4)
    fixed_length += padding
    print(f"Fixed_length ajustado a {fixed_length} para ser divisible por 4.")

# ============================================
# 3. Funciones Auxiliares
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

def calculate_bit_match_percentage(original, reconstructed):
    """
    Calcula el porcentaje de coincidencia bit a bit entre la señal original y la reconstruida.

    Parámetros:
    - original: np.array, señal original con forma (length, 1)
    - reconstructed: np.array, señal reconstruida con forma (length, 1)

    Retorna:
    - match_percentage: float, porcentaje de coincidencia bit a bit
    """
    # Convertir a 16-bit PCM
    original_int = np.int16(original.flatten() * 32767)
    reconstructed_int = np.int16(reconstructed.flatten() * 32767)
    
    # Convertir a bits
    original_bits = np.unpackbits(original_int.view(np.uint8))
    reconstructed_bits = np.unpackbits(reconstructed_int.view(np.uint8))
    
    matches = np.sum(original_bits == reconstructed_bits)
    total_bits = original_bits.size
    match_percentage = (matches / total_bits) * 100
    return match_percentage

def save_difference_bits_as_wav(differences, file_path, sample_rate=22050):
    """
    Convierte los bits de diferencia en una señal de audio y guarda como WAV.

    Parámetros:
    - differences: np.array, array de bits de diferencia (0 o 1).
    - file_path: str, ruta donde se guardará el archivo WAV.
    - sample_rate: int, frecuencia de muestreo para el archivo WAV.
    """
    # Mapear los bits a amplitudes: 0 -> -1.0, 1 -> 1.0
    audio_signal = differences.astype(np.float32) * 2 - 1  # 0 -> -1, 1 -> 1

    # Opcional: Suavizar la señal para evitar clicks abruptos
    window_size = 100  # Puedes ajustar este valor
    window = np.hanning(window_size)
    audio_signal = np.convolve(audio_signal, window, mode='same') / window.sum()

    # Normalizar la señal
    if np.max(np.abs(audio_signal)) != 0:
        audio_signal /= np.max(np.abs(audio_signal))

    # Guardar como WAV
    sf.write(file_path, audio_signal, samplerate=sample_rate)
    print(f"Archivo WAV de diferencias guardado en: {file_path}\n")

def save_remaining_bits_compressed(original, reconstructed, file_path, sample_rate=22050):
    """
    Guarda los bits de diferencia entre la señal original y reconstruida, comprimidos con gzip.
    Además, genera y guarda un archivo WAV que representa los bits de diferencia.

    Parámetros:
    - original: np.array, señal original con forma (length, 1)
    - reconstructed: np.array, señal reconstruida con forma (length, 1)
    - file_path: str, ruta donde se guardarán los bits comprimidos
    - sample_rate: int, frecuencia de muestreo para el archivo WAV de diferencias
    """
    original_int = np.int16(original.flatten() * 32767)
    reconstructed_int = np.int16(reconstructed.flatten() * 32767)

    original_bits = np.unpackbits(original_int.view(np.uint8))
    reconstructed_bits = np.unpackbits(reconstructed_int.view(np.uint8))

    differences = original_bits != reconstructed_bits
    remaining_bits_packed = np.packbits(differences)  # Empaquetar bits en bytes

    # Comprimir los bits empaquetados usando gzip
    with gzip.open(file_path, 'wb') as f:
        f.write(remaining_bits_packed.tobytes())
    print(f"Bits restantes comprimidos guardados en: {file_path}\n")

    # Generar y guardar el archivo WAV de diferencias
    difference_wav_path = file_path.replace('.bin.gz', '_difference.wav')
    save_difference_bits_as_wav(differences, difference_wav_path, sample_rate)

def create_and_save_difference_plot(original, reconstructed, num_plots=1, save_path='difference_plot.png'):
    """
    Crea y guarda gráficos que muestran las diferencias entre la señal original y la reconstruida.

    Parámetros:
    - original: np.array, señal original con forma (length, 1)
    - reconstructed: np.array, señal reconstruida con forma (length, 1)
    - num_plots: int, número de gráficos de diferencia a crear
    - save_path: str, ruta donde se guardará un gráfico de diferencias
    """
    difference = original - reconstructed
    plt.figure(figsize=(15, 5 * num_plots))
    for i in range(num_plots):
        ax = plt.subplot(num_plots, 1, i + 1)
        plt.plot(difference[i].flatten())
        plt.title("Diferencias entre Original y Reconstruida")
        plt.xlabel("Muestras")
        plt.ylabel("Amplitud")
    plt.tight_layout()
    plt.show()

    # Guardar un gráfico de ejemplo de diferencias
    plt.figure(figsize=(15, 5))
    plt.plot(difference[0].flatten())
    plt.title("Diferencias entre Original y Reconstruida")
    plt.xlabel("Muestras")
    plt.ylabel("Amplitud")
    plt.savefig(save_path)
    plt.close()
    print(f"Gráfico de diferencias guardado como '{save_path}'.\n")

def apply_bit_corrections(reconstructed, remaining_bits_packed, original_bits):
    """
    Aplica las correcciones de bits a la señal reconstruida utilizando los bits de diferencia.

    Parámetros:
    - reconstructed: np.array, señal reconstruida con forma (1, length, 1), valores [-1,1]
    - remaining_bits_packed: bytes, bits empaquetados de diferencias
    - original_bits: np.array, bits de la señal original

    Retorna:
    - corrected_signal: np.array, señal corregida con forma (1, length, 1), valores [-1,1]
    """
    # Desempaquetar los bits de diferencia
    remaining_bits = np.unpackbits(np.frombuffer(remaining_bits_packed, dtype=np.uint8))

    # Convertir la señal reconstruida a 16-bit PCM
    reconstructed_int = np.int16(reconstructed.flatten() * 32767)

    # Convertir a bits
    reconstructed_bits = np.unpackbits(reconstructed_int.view(np.uint8))

    # Aplicar las diferencias: donde remaining_bits es 1, reemplazar con original_bits
    corrected_bits = np.where(remaining_bits, original_bits, reconstructed_bits)

    # Asegurarse de que el número de bits coincide con el número de bits originales
    num_bits = original_bits.size
    corrected_bits = corrected_bits[:num_bits]

    # Reshape bits a muestras de 16 bits
    if num_bits % 16 != 0:
        # Rellenar con ceros si es necesario
        padding = 16 - (num_bits % 16)
        corrected_bits = np.pad(corrected_bits, (0, padding), 'constant')
    corrected_bits = corrected_bits.reshape(-1, 16)

    # Convertir cada grupo de 16 bits a int16
    corrected_bytes = np.packbits(corrected_bits, axis=1)
    corrected_int = corrected_bytes.view(np.int16).flatten()

    # Convertir a float32 y normalizar entre -1 y 1
    corrected_signal = corrected_int.astype(np.float32) / 32767.0

    # Asegurar la forma correcta
    corrected_signal = corrected_signal.reshape(1, -1, 1)

    return corrected_signal

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

# ============================================
# 4. Definición del Modelo Autoencoder Convolucional 1D
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
    decoder_layer = autoencoder.layers[-7](decoder_input)  # Dense
    decoder_layer = autoencoder.layers[-6](decoder_layer)  # Reshape
    decoder_layer = autoencoder.layers[-5](decoder_layer)  # UpSampling1D
    decoder_layer = autoencoder.layers[-4](decoder_layer)  # Conv1D(8)
    decoder_layer = autoencoder.layers[-3](decoder_layer)  # UpSampling1D
    decoder_layer = autoencoder.layers[-2](decoder_layer)  # Conv1D(16)
    decoder_output = autoencoder.layers[-1](decoder_layer)  # Conv1D(1)
    decoder = Model(decoder_input, decoder_output)

    return autoencoder, encoder, decoder

# ============================================
# 5. Entrenamiento del Modelo
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
# 7. Proceso de Inferencia Optimizado
# ============================================

def inference_process_compressed_optimized(encoder_model, decoder_model, original_signal, inference_folder, sample_rate=22050):
    """
    Codifica una señal, guarda el vector latente comprimido, decodifica, guarda las señales,
    calcula y guarda las diferencias, aplica correcciones y verifica la reconstrucción.

    Parámetros:
    - encoder_model: Model, el modelo codificador.
    - decoder_model: Model, el modelo decodificador.
    - original_signal: np.array, la señal original de entrada (length, 1).
    - inference_folder: str, la carpeta donde se guardarán los resultados de la inferencia.
    - sample_rate: int, frecuencia de muestreo para guardar los archivos WAV.

    Retorna:
    - reconstructed_signal: np.array, la señal reconstruida.
    - is_equal: bool, si la señal reconstruida es igual a la original.
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
    
    # Calcular el porcentaje de coincidencia bit a bit
    print("Calculando el porcentaje de coincidencia bit a bit...")
    bit_match_percentage = calculate_bit_match_percentage(original_signal, reconstructed_signal)
    print(f"Porcentaje de coincidencia bit a bit: {bit_match_percentage:.2f}%\n")
    
    # Calcular los bits de diferencia
    original_int = np.int16(original_signal.flatten() * 32767)
    reconstructed_int = np.int16(reconstructed_signal.flatten() * 32767)
    original_bits = np.unpackbits(original_int.view(np.uint8))
    reconstructed_bits = np.unpackbits(reconstructed_int.view(np.uint8))
    differences = original_bits != reconstructed_bits
    
    # Guardar los bits de diferencia comprimidos y como WAV
    remaining_bits_path = os.path.join(inference_folder, 'remaining_bits.bin.gz')
    print("Guardando los bits de diferencia comprimidos y como WAV...")
    save_remaining_bits_compressed(original_signal, reconstructed_signal, remaining_bits_path, sample_rate)
    
    # Leer los bits de diferencia comprimidos
    print("Cargando los bits de diferencia comprimidos...")
    with gzip.open(remaining_bits_path, 'rb') as f:
        remaining_bits_packed = f.read()
    
    # Aplicar las diferencias para corregir
    print("Aplicando correcciones a la señal reconstruida...")
    corrected_signal = apply_bit_corrections(reconstructed, remaining_bits_packed, original_bits)
    
    # Guardar la señal corregida como gráfico y WAV
    corrected_path_png = os.path.join(inference_folder, 'corrected_signal.png')
    corrected_path_wav = os.path.join(inference_folder, 'corrected_signal.wav')
    corrected_signal_flat = corrected_signal.reshape(-1)
    plt.figure(figsize=(15, 5))
    librosa.display.waveshow(corrected_signal_flat, sr=sample_rate)
    plt.title("Señal Corregida")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.savefig(corrected_path_png)
    plt.close()
    print(f"Señal corregida guardada en '{corrected_path_png}'.")
    
    # Guardar la señal corregida como WAV
    sf.write(corrected_path_wav, corrected_signal_flat, samplerate=sample_rate)
    print(f"Señal corregida guardada como WAV en '{corrected_path_wav}'.\n")
    
    # Comparar la señal corregida con la original
    original_int = np.int16(original_signal.flatten() * 32767)
    corrected_int = np.int16(corrected_signal.flatten() * 32767)
    is_equal = np.array_equal(original_int, corrected_int)
    print(f"¿La señal corregida es igual a la original? {'Sí' if is_equal else 'No'}\n")
    
    # Si no coincide, mostrar las diferencias adicionales
    if not is_equal:
        difference_signal = original_signal - corrected_signal
        difference_signal_scaled = difference_signal.flatten()
        additional_diff_path_png = os.path.join(inference_folder, 'additional_difference.png')
        additional_diff_path_wav = os.path.join(inference_folder, 'additional_difference.wav')
        plt.figure(figsize=(15, 5))
        librosa.display.waveshow(difference_signal_scaled, sr=sample_rate)
        plt.title("Diferencias Adicionales")
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Amplitud")
        plt.savefig(additional_diff_path_png)
        plt.close()
        print(f"Diferencias adicionales guardadas como '{additional_diff_path_png}'.")
        
        # Guardar las diferencias adicionales como WAV
        sf.write(additional_diff_path_wav, difference_signal_scaled, samplerate=sample_rate)
        print(f"Diferencias adicionales guardadas como WAV en '{additional_diff_path_wav}'.\n")
    
    return reconstructed_signal, is_equal

# ============================================
# 8. Ejecución de Inferencia en Todas las Señales de Prueba
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
# 9. Visualización de las Señales
# ============================================

# Visualizar las Señales para Cada Señal de Prueba
for test_index in range(x_test.shape[0]):
    specific_inference_dir = os.path.join(inference_dir, f'signal_{test_index}')
    
    original_path_png = os.path.join(specific_inference_dir, 'original_signal.png')
    reconstructed_path_png = os.path.join(specific_inference_dir, 'reconstructed_signal.png')
    corrected_path_png = os.path.join(specific_inference_dir, 'corrected_signal.png')
    
    if os.path.exists(original_path_png) and os.path.exists(reconstructed_path_png) and os.path.exists(corrected_path_png):
        # Cargar las imágenes de las señales
        original_img = plt.imread(original_path_png)
        reconstructed_img = plt.imread(reconstructed_path_png)
        corrected_img = plt.imread(corrected_path_png)
        
        # Mostrar las imágenes
        plt.figure(figsize=(18, 6))
        
        # Señal original
        plt.subplot(1, 3, 1)
        plt.imshow(original_img, aspect='auto')
        plt.title("Señal Original")
        plt.axis("off")
        
        # Señal reconstruida
        plt.subplot(1, 3, 2)
        plt.imshow(reconstructed_img, aspect='auto')
        plt.title("Señal Reconstruida")
        plt.axis("off")
        
        # Señal corregida
        plt.subplot(1, 3, 3)
        plt.imshow(corrected_img, aspect='auto')
        plt.title("Señal Corregida")
        plt.axis("off")
        
        plt.tight_layout()
        plt.show()
    else:
        print(f"Archivos de visualización no encontrados para la señal de prueba con índice {test_index}. Asegúrate de que la inferencia se realizó correctamente.\n")
