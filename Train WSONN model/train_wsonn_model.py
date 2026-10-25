"""
WSO-NN Model Training Script for LoRaWAN Parameter Optimization
SMALLER MODEL: Reduced architecture for ESP32-S3 Nano
Option 1: 12 → 10 → 8 layers (~40% smaller)
Corrected Version: Energy-Aware Constraints and Signal Logic Alignment
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

# ============================================
# STEP 1: Define the Optimization Task
# ============================================
def calculate_fitness(sf, tx_power, snr, rssi, battery):
    """
    Fitness function for the WSO algorithm. Lower values indicate better performance.
    """
    reliability_penalty = 0
    if snr < 5:
        reliability_penalty += 10
    if snr < 0:
        reliability_penalty += 10
    if rssi < -110:
        reliability_penalty += 10
    elif rssi < -100:
        reliability_penalty += 5

    energy_penalty = (sf / 12.0) + (tx_power / 20.0)
    
    if battery < 30:
        energy_penalty *= 1.5
    if battery < 15:
        energy_penalty *= 2.0

    return reliability_penalty + energy_penalty

# ============================================
# STEP 2: Simulated WSO Optimizer (CORRECTED)
# ============================================
def simulated_wso_optimizer(snr, rssi, battery, packet_loss):
    """
    Simulated WSO algorithm that determines optimal parameters based on real thesis bounds.
    """
    # 1. Align Channel Quality: Strong signal defaults to a high base attack ratio
    if snr > 10 and rssi > -80:
        base_attack = 0.75
    elif snr > 0 and rssi > -100:
        base_attack = 0.60
    else:
        base_attack = 0.45

    # 2. ENFORCE BATTERY CONSTRAINTS: Override attack ratio based on energy thresholds
    if battery < 10.0:    # Critical Battery Threshold
        attack_ratio = 0.31
    elif battery < 20.0:  # Low Battery Threshold
        attack_ratio = 0.36
    else:
        attack_ratio = base_attack

    # 3. Exploration Factor: High packet loss or weak battery drives scouting behavior
    exploration_factor = 0.22 + (packet_loss * 1.5)
    if battery < 20.0:
        exploration_factor += 0.35  # Force high defensive scouting when battery is dying
    exploration_factor = min(max(exploration_factor, 0.1), 0.9)

    # 4. Mutation Strength: Progressively grows with packet loss to crack system stagnation
    mutation_strength = 0.15 + (packet_loss * 2.0)
    mutation_strength = min(max(mutation_strength, 0.1), 0.7)

    # 5. Weight Decay: Heavy penalty applied under low-power modes
    if battery < 20.0:
        weight_decay = 0.20
    else:
        weight_decay = 0.05

    return attack_ratio, exploration_factor, mutation_strength, weight_decay

# ============================================
# STEP 3: Generate and Normalize Training Data
# ============================================
X_MIN = np.array([-10.0, -120.0,   5.0, 0.0, 0.1, 0.0, 0.1], dtype=np.float32)
X_MAX = np.array([ 20.0,  -60.0, 100.0, 0.25, 0.9, 0.5, 0.8], dtype=np.float32)

def generate_wsonn_training_data(num_samples=10000):
    X_raw = []
    Y_train = []
    print(f"Generating {num_samples} training samples...")
    
    for i in range(num_samples):
        snr = np.random.uniform(-10, 20)
        rssi = np.random.uniform(-120, -60)
        battery = np.random.uniform(5, 100)
        packet_loss = np.random.uniform(0, 0.25)
        
        diversity = np.random.uniform(0.1, 0.9)
        stagnation = np.random.uniform(0.0, 0.5)
        complexity = np.random.uniform(0.1, 0.8)
        
        features = [snr, rssi, battery, packet_loss, diversity, stagnation, complexity]
        
        attack_ratio, exploration_factor, mutation_strength, weight_decay = simulated_wso_optimizer(
            snr, rssi, battery, packet_loss
        )
        
        labels = [attack_ratio, exploration_factor, mutation_strength, weight_decay]
        X_raw.append(features)
        Y_train.append(labels)

    X_raw = np.array(X_raw, dtype=np.float32)
    Y_train = np.array(Y_train, dtype=np.float32)
    
    X_scaled = (X_raw - X_MIN) / (X_MAX - X_MIN)
    
    return X_scaled, Y_train

# ============================================
# STEP 4: Define the SMALLER Neural Network
# ============================================
def create_model():
    """
    SMALLER MODEL ARCHITECTURE - Option 1:
    Input (7) → Dense(12) → Dense(10) → Dense(8) → Output(4)
    """
    model = keras.Sequential([
        keras.layers.Input(shape=(7,)),
        keras.layers.Dense(12, activation='relu', name='dense_1'),   
        keras.layers.Dense(10, activation='relu', name='dense_2'),   
        keras.layers.Dense(8, activation='relu', name='dense_3'),    
        keras.layers.Dense(4, activation='sigmoid', name='output')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    print("\n📊 Model Architecture (Smaller Version):")
    model.summary()
    
    return model

# ============================================
# STEP 5: Convert and Fully Quantize the Model
# ============================================
def convert_to_tflite(model, X_train_sample, model_name="wsonn_model_quantized_small"):
    print("\n==================================================")
    print("Converting to TensorFlow Lite with Full INT8 Enforcement")
    print("==================================================")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.target_spec.supported_types = [tf.int8]
    
    def representative_dataset():
        sample_size = min(200, len(X_train_sample))
        indices = np.random.choice(len(X_train_sample), sample_size, replace=False)
        for idx in indices:
            yield [X_train_sample[idx].reshape(1, 7).astype(np.float32)]
            
    converter.representative_dataset = representative_dataset
    
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    
    tflite_model = converter.convert()
    
    tflite_filename = f"{model_name}.tflite"
    with open(tflite_filename, 'wb') as f:
        f.write(tflite_model)
    
    print(f" ✅ Quantized model saved as: {tflite_filename}")
    print(f" 📦 Model size: {len(tflite_model) / 1024:.2f} KB")
    
    original_size = 5.7  
    new_size = len(tflite_model) / 1024
    reduction = ((original_size - new_size) / original_size) * 100
    print(f" 📊 Size reduction: {reduction:.1f}% smaller than original")
    
    return tflite_model

def create_header_file(tflite_model, header_name="wsonn_model.h"):
    print("\n==================================================")
    print("Creating C++ Header File for Arduino")
    print("==================================================")
    
    model_bytes = ', '.join(f'0x{byte:02x}' for byte in tflite_model)
    header_content = f"""// Auto-generated WSO-NN Model Header (SMALLER VERSION - Option 1)
// Architecture: Input(7) → Dense(12) → Dense(10) → Dense(8) → Output(4)
// Model size: {len(tflite_model)} bytes (~40% smaller than original)
#ifndef WSONN_MODEL_H
#define WSONN_MODEL_H

const unsigned char wsonn_model_quantized_tflite[] = {{
{model_bytes}
}};
const unsigned int wsonn_model_quantized_tflite_len = {len(tflite_model)};
#endif // WSONN_MODEL_H
"""
    with open(header_name, 'w') as f:
        f.write(header_content)
    print(f" ✅ Header file saved as: {header_name}")
    print(f" 📁 File location: {os.path.abspath(header_name)}")

# ============================================
# STEP 6: Validate Model Performance
# ============================================
def validate_model(model, X_test, Y_test):
    print("\n==================================================")
    print("Validating Model Performance")
    print("==================================================")
    
    for i in range(5):
        idx = np.random.randint(0, len(X_test))
        sample = X_test[idx:idx+1]
        prediction = model.predict(sample, verbose=0)
        
        print(f"\nSample {i+1}:")
        print(f"  Input features: {X_test[idx]}")
        print(f"  Predicted (Atk, Exp, Mut, WD): {prediction[0]}")
        print(f"  Actual (Atk, Exp, Mut, WD):    {Y_test[idx]}")

# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == "__main__":
    print("=" * 50)
    print("WSO-NN Training - SMALLER MODEL (Option 1)")
    print("Architecture: 12 → 10 → 8 (Corrected Math constraints)")
    print("=" * 50)
    
    X_train, Y_train = generate_wsonn_training_data(num_samples=10000)
    model = create_model()
    
    print("\n🚀 Starting training...")
    history = model.fit(
        X_train, Y_train, 
        epochs=150, 
        validation_split=0.2, 
        verbose=1
    )
    
    model.save('wsonn_model_small.keras')
    print("\n✅ Keras model saved as: wsonn_model_small.keras")
    
    tflite_model = convert_to_tflite(model, X_train)
    create_header_file(tflite_model)
    validate_model(model, X_train[:100], Y_train[:100])
    
    print("\n" + "=" * 50)
    print("✅ ALL TASKS COMPLETED SUCCESSFULLY!")
    print("=" * 50)
