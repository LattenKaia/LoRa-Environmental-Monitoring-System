# LoRa-Environmental-Monitoring-System
LoRa wireless technology was developed by Semtech, it is a long-range, low-power wireless communication technology. The term "LoRa" derives from the first two letters of the English words "Long Range" ("Lo" and "Ra"), meaning long range. It is a physical-layer technology based on Linear Chirp Spread Spectrum (CSS) modulation, offering strong resistance to noise, multipath fading, and the Doppler effect. 
The project includes an end to end, low power IoT solution for remote environment monitoring. It features a custom-designed PCB (Printed Circuit Board), an ESP32-P4 micro-controller running a quantized War Strategy Optimization - Neural Network (WSO-NN) algorithm, and a full cloud data pipeline.
# Project Overview
This project was developed to solve the challenge of reliable, long-term environmental monitoring in remote outdoor locations. The system collects temperature, humidity, and barometric pressure, transmits the data via LoRaWAN (CN470), and visualizes it on a cloud platform. The core innovation is the WSO-NN (War Strategy Optimization - Neural Network) algorithm running directly on the microcontroller, which dynamically optimizes LoRa transmission parameters (Spreading Factor and TX Power) to maximize battery life and data reliability based on real-time conditions.
# Key Features
* Edge AI Optimization: Uses a quantized TensorFlow Lite Micro model model (4.6KB) to predict optimal LoRa parameters. Inference time is 2 - 5ms, trained offline using the War Strategy Optimization - Neural Network (WSO-NN) algorithm.
* Real Battery Monitoring: Implements a hardware voltage divider connected to an ADC pin for accurate real time battery voltage readings.
* Hardware design: Designed using a hybrid SMD/Through-Hole PCB assembly approach, housed in an IP66 enclosure with a breathable valve to prevent condensation.
* Solar Power Management: Integrate a CN3791 MPPT charger with a 9V solar panel and 3.7V Li-ion battery.
* Full Cloud Pipeline: Leverage ChirpStack, InfluxDB, ThingsBoard, and Grafana for data ingestion, storage, and visualization. 
# System Architecture
## Overall Architecture

The system is composed of five layers from sensor to visualization. The WSO-NN algorithm runs directly inside the LoRa sensor node in the perception layer for edge intelligence.
<img width="2889" height="2118" alt="LoRaWAN architecture" src="https://github.com/user-attachments/assets/8ec1a3c0-d279-43fa-aea0-d1a737e21d99" />
# Hardware Design
* Microcontroller: ESP32-P4-M (That specific Microcontroller is chosen for because of its hardware FPU and large PSRAM to run the Neural Network)
* LoRa Transceiver: Ai-Thinker RA-02 (SX1278)
* Sensors: Bosch BME280 sensor
* Power: CN3791 MPPT Solar Charger, AP2112K-33 LDO regulator.
* PCB: Custom 2-layer board designed in EasyEDA. Feature a solid ground pour on the bottom layer and a 3.3V copper pour on the top layer.
# Software & Firmware
* Language: C, C++ (Arduino Framework), Python (train WSONN model). Note: The firmware code, train wsonn model code is upload and attach above 
* Algorithm: The neural network acts as a real-time surrogate for the WSO (War Strategy Optimization) algorithm. It takes 7 inputs(SNR, RSSI, Battery %, Packet Loss, Diversity, Stagnation, Complexity) and output 4 control parameters(Attack Ratio, Exploration, Mutation, Weight Decay).
* Power Management: The LoRa Sensor node operates in a continuous loop with a 30-second uplink interval. (Deep sleep is currently disabled for testing purpose, during diployment, deepsleep will be activated to help save power consumption)
# Cloud Platform
The backend is fully containerized using Docker and Docker Compose: 
* ChirpStack: LoRaWAN Network Server
* InfluxDB: Time-series database for sensor data storage
* ThingsBoard: Device management and downlink command generation.
* Grafana: Advanced data visualization and dashboarding.


