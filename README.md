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
