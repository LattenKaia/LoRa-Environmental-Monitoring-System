# LoRa-Environmental-Monitoring-System
LoRa wireless technology was developed by Semtech, it is a long-range, low-power wireless communication technology. The term "LoRa" derives from the first two letters of the English words "Long Range" ("Lo" and "Ra"), meaning long range. It is a physical-layer technology based on Linear Chirp Spread Spectrum (CSS) modulation, offering strong resistance to noise, multipath fading, and the Doppler effect. 
The project includes an end to end, low power IoT solution for remote environment monitoring. It features a custom-designed PCB (Printed Circuit Board), an ESP32-P4 micro-controller running a quantized War Strategy Optimization - Neural Network (WSO-NN) algorithm, and a full cloud data pipeline.
## Project Overview
This project was developed to solve the challenge of reliable, long-term environmental monitoring in remote outdoor locations. The system collects temperature, humidity, and barometric pressure, transmits the data via LoRaWAN (CN470), and visualizes it on a cloud platform. The core innovation is the WSO-NN (War Strategy Optimization - Neural Network) algorithm running directly on the microcontroller, which dynamically optimizes LoRa transmission parameters (Spreading Factor and TX Power) to maximize battery life and data reliability based on real-time conditions.
## Key Features
* Edge AI Optimization: Uses a quantized TensorFlow Lite Micro model model (4.6KB) to predict optimal LoRa parameters. Inference time is 2 - 5ms, trained offline using the War Strategy Optimization - Neural Network (WSO-NN) algorithm.
* Real Battery Monitoring: Implements a hardware voltage divider connected to an ADC pin for accurate real time battery voltage readings.
* Hardware design: Designed using a hybrid SMD/Through-Hole PCB assembly approach, housed in an IP66 enclosure with a breathable valve to prevent condensation.
* Solar Power Management: Integrate a CN3791 MPPT charger with a 9V solar panel and 3.7V Li-ion battery.
* Full Cloud Pipeline: Leverage ChirpStack, InfluxDB, ThingsBoard, and Grafana for data ingestion, storage, and visualization. 
## System Architecture
### Overall Architecture

The system is composed of five layers from sensor to visualization. The WSO-NN algorithm runs directly inside the LoRa sensor node in the perception layer for edge intelligence.
<img width="2889" height="2118" alt="architecture" src="https://github.com/user-attachments/assets/163a553b-20b7-449d-8967-4b91c806c789" />
## Hardware Design
* Microcontroller: ESP32-P4-M (That specific Microcontroller is chosen for because of its hardware FPU and large PSRAM to run the Neural Network)
* LoRa Transceiver: Ai-Thinker RA-02 (SX1278)
* Sensors: Bosch BME280 sensor
* Power: CN3791 MPPT Solar Charger, AP2112K-33 LDO regulator.
* PCB: Custom 2-layer board designed in EasyEDA. Feature a solid ground pour on the bottom layer and a 3.3V copper pour on the top layer.<br><br>
*LoRa Gateway*<img width="1536" height="2048" alt="Gateway" src="https://github.com/user-attachments/assets/81ba3352-c8f2-468b-b0cf-1292be95e04e" /><br><br>
*LoRa Sensor node*<img width="1536" height="2048" alt="LoRa Sensor node" src="https://github.com/user-attachments/assets/61dae78f-96b3-4568-97f7-87d8f91aff03" />
## Software & Firmware
* Language: C, C++ (Arduino Framework), Python (train WSONN model). Note: The firmware code, train wsonn model code is upload and attach above 
* Algorithm: The neural network acts as a real-time surrogate for the WSO (War Strategy Optimization) algorithm. It takes 7 inputs(SNR, RSSI, Battery %, Packet Loss, Diversity, Stagnation, Complexity) and output 4 control parameters(Attack Ratio, Exploration, Mutation, Weight Decay). There are two files in the Firmware Source Code folder, the WSONN.ino file, which is the C++ Arduino code I uploaded to the LoRa sensor node I design, and the wsonn_model.h file, which is the header file I generated when training the wsonn model. The Train WSONN model folder contain the python code I use to train the WSONN (War Strategy Optimization - Neural Network) model. 
* Power Management: The LoRa Sensor node operates in a continuous loop with a 30-second uplink interval. (Deep sleep is currently disabled for testing purpose, during diployment, deepsleep will be activated to help save power consumption)
## Cloud Platform
The backend is fully containerized using Docker and Docker Compose: 
* ChirpStack: LoRaWAN Network Server
* InfluxDB: Time-series database for sensor data storage
* ThingsBoard: Device management and downlink command generation.
* Grafana: Advanced data visualization and dashboarding.
<br><br>
*docker*<img width="1647" height="957" alt="docker" src="https://github.com/user-attachments/assets/e56eb392-83e1-4bfc-b941-beb673b153cb" /><br><br>
*docker-compose up*<img width="1902" height="1055" alt="Powershell" src="https://github.com/user-attachments/assets/dcc07076-abee-4c80-aa75-82551f98d02c" /><br><br>
*ChirpStack*<img width="1899" height="916" alt="Chirpstack" src="https://github.com/user-attachments/assets/65ab25c8-922c-493a-a128-73a3be6277b5" /><br><br>
*ChirpStack Dashboard*<img width="1877" height="983" alt="chirpstack22" src="https://github.com/user-attachments/assets/d195f484-fa71-4ff3-8840-10e2378446bd" /><br><br>
*ThingsBoard*<img width="1912" height="984" alt="github thingsboard" src="https://github.com/user-attachments/assets/90274352-fd96-4462-aa15-81e67a2b1120" /><br><br>
*Grafana Visualization*<img width="1909" height="1012" alt="Grafana 2" src="https://github.com/user-attachments/assets/eb89834f-db83-49e5-900d-1332919b788a" />
## What I'll improve and currently working on it
1. Deep Sleep Integration: Currently, the ESP32 runs continuosly. Enabling deep sleep will drastically extend battery life for the sensor node.
2. Direct MPPT (Maximum Power Point Tracking) Integration: Instead of using the waveshare module, I will place the CN3791 IC (Integraded Circuit) directly on my custom PCB (Printed Circuit Board) to save space and reduce cost. I am currently designing the whole PCB, integrating all the component of the sensor node into one small PCB (Printed Circuit Board). The below image is my current project I am working on, redesigning the LoRa Sensor node with all the electrical component into one small Print Circuit Board (PCB).<br><br>
*LoRa Sensor node circuit diagram*<img width="777" height="551" alt="Sensor node PCB" src="https://github.com/user-attachments/assets/575c4ff0-20ca-4465-8b2d-c4a7891c93b7" />







