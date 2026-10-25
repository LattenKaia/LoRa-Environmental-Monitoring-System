#include <Wire.h>
#include <SPI.h>
#include <RadioLib.h>
#include <Adafruit_SSD1306.h>
#include <Adafruit_GFX.h>
#include <Adafruit_BME280.h>

// ESP32 Deep Sleep (DISABLED)
// #include "esp_sleep.h"

// TensorFlow Lite includes
#include "wsonn_model.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ============================================================================
// --- TEST MODE ---
// Set to 1 for battery sweep test (isolates battery as only variable)
// Set to 0 for normal operation (real sensor inputs)
// ============================================================================
#define TEST_MODE 0

// --- Pin Definitions ---
#define I2C_SDA 7
#define I2C_SCL 8
#define OLED_ADDR 0x3C
#define BME_ADDR 0x76
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64

// LoRa Pins
#define LORA_CS     17
#define LORA_RST    18
#define LORA_DIO0   19
#define LORA_SCK    14
#define LORA_MOSI   15
#define LORA_MISO   16

// --- Battery ADC Pin (with 10k / 10k voltage divider) ---
#define BATTERY_ADC_PIN    23
#define DIVIDER_RATIO      2.0f      // (R1+R2)/R2 = (10k+10k)/10k = 2.0

// --- LoRaWAN Credentials ---
#define RADIOLIB_LORAWAN_JOIN_EUI  0x2ed50992ca0f0a88
#define RADIOLIB_LORAWAN_DEV_EUI   0xb8774af59599f236
#define RADIOLIB_LORAWAN_APP_KEY   {0x4e, 0x7f, 0xa4, 0xc8, 0x77, 0x5c, 0xf2, 0x7e, 0x1d, 0xb1, 0x2b, 0x3e, 0xb3, 0xc5, 0x59, 0xe0}
#define RADIOLIB_LORAWAN_NWK_KEY   RADIOLIB_LORAWAN_APP_KEY

// ===== CN470 Settings =====
#define LORA_FREQ   470.3
#define CN470_SUB_BAND 1

// ===== Uplink Interval =====
#define UPLINK_INTERVAL 30000  // 30 seconds (faster for testing)

// ===== Battery Constants =====
#define BATTERY_CAPACITY_MAH 1200.0
#define LOW_BATTERY_THRESHOLD 20.0
#define CRITICAL_BATTERY_THRESHOLD 10.0

// --- RTC Memory for Persistence ---
RTC_DATA_ATTR uint32_t bootCount = 0;
RTC_DATA_ATTR uint32_t savedPacketCount = 0;
RTC_DATA_ATTR float estimated_battery_mah = BATTERY_CAPACITY_MAH;
RTC_DATA_ATTR uint32_t totalCycles = 0;
RTC_DATA_ATTR uint32_t batteryResetCount = 0;

// --- RTC Memory for LoRaWAN Session Persistence ---
RTC_DATA_ATTR uint8_t savedNwkSKey[16] = {0};
RTC_DATA_ATTR uint8_t savedAppSKey[16] = {0};
RTC_DATA_ATTR uint32_t savedDevAddr = 0;
RTC_DATA_ATTR uint32_t savedFCntUp = 0;
RTC_DATA_ATTR uint32_t savedFCntDown = 0;
RTC_DATA_ATTR bool hasValidSession = false;

// --- SNR/RSSI Caching (to avoid 0.0 values after sendReceive) ---
static float lastValidSNR = 5.0f;
static float lastValidRSSI = -95.0f;

// --- TensorFlow Lite Setup ---
namespace {
    constexpr int kTensorArenaSize = 10 * 1024;
    uint8_t tensor_arena[kTensorArenaSize];
    const tflite::Model* tflite_model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* input_tensor = nullptr;
    TfLiteTensor* output_tensor = nullptr;
    tflite::MicroMutableOpResolver<10> resolver;
}

// Normalization constants
const float X_MIN[7] = {-10.0f, -120.0f,   5.0f, 0.00f, 0.10f, 0.00f, 0.10f};
const float X_MAX[7] = { 20.0f,  -60.0f, 100.0f, 0.25f, 0.90f, 0.50f, 0.80f};

struct WSOParameters {
    float attack_ratio;
    float exploration_factor;
    float mutation_strength;
    float weight_decay;
};

// --- Objects ---
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);
Adafruit_BME280 bme;

// ===== Radio Objects (created dynamically) =====
Module* module = nullptr;
SX1276* radio = nullptr;
LoRaWANNode* node = nullptr;

// ===== Region =====
const LoRaWANBand_t* Region = &CN470;

uint64_t joinEUI = RADIOLIB_LORAWAN_JOIN_EUI;
uint64_t devEUI = RADIOLIB_LORAWAN_DEV_EUI;
uint8_t appKey[] = RADIOLIB_LORAWAN_APP_KEY;
uint8_t nwkKey[] = RADIOLIB_LORAWAN_NWK_KEY;

bool joined = false;
uint32_t lastUplink = 0;
uint32_t packetCount = 0;
uint32_t packetLossCount = 0;
uint32_t totalPacketsSent = 0;

// Downlink command variables
uint32_t newUplinkInterval = UPLINK_INTERVAL;
bool adrEnabled = false;
uint8_t targetDataRate = 5;

// ============================================================================
// --- REAL Battery Monitoring Functions (Calibrated ADC, 10k/10k divider) ---
// ============================================================================

// Read battery voltage using CALIBRATED ADC (uses factory eFuse calibration)
float readBatteryVoltage() {
    const int samples = 32;
    uint32_t total_mv = 0;
    for (int i = 0; i < samples; i++) {
        total_mv += analogReadMilliVolts(BATTERY_ADC_PIN);
        delayMicroseconds(50);
    }
    float adc_mv = (float)total_mv / (float)samples;
    float battery_voltage = (adc_mv / 1000.0f) * DIVIDER_RATIO;
    return battery_voltage;
}

// Realistic Li-ion 3.7V discharge curve
float voltageToPercentage(float voltage) {
    if      (voltage >= 4.20f) return 100.0f;
    else if (voltage >= 4.15f) return 95.0f + (voltage - 4.15f) / 0.05f * 5.0f;
    else if (voltage >= 4.10f) return 90.0f + (voltage - 4.10f) / 0.05f * 5.0f;
    else if (voltage >= 4.00f) return 80.0f + (voltage - 4.00f) / 0.10f * 10.0f;
    else if (voltage >= 3.90f) return 65.0f + (voltage - 3.90f) / 0.10f * 15.0f;
    else if (voltage >= 3.80f) return 50.0f + (voltage - 3.80f) / 0.10f * 15.0f;
    else if (voltage >= 3.70f) return 35.0f + (voltage - 3.70f) / 0.10f * 15.0f;
    else if (voltage >= 3.60f) return 20.0f + (voltage - 3.60f) / 0.10f * 15.0f;
    else if (voltage >= 3.50f) return 10.0f + (voltage - 3.50f) / 0.10f * 10.0f;
    else if (voltage >= 3.40f) return  5.0f + (voltage - 3.40f) / 0.10f *  5.0f;
    else if (voltage >= 3.30f) return  2.0f + (voltage - 3.30f) / 0.10f *  3.0f;
    else if (voltage >= 3.00f) return  0.0f + (voltage - 3.00f) / 0.30f *  2.0f;
    else                       return  0.0f;
}

// Cached battery values (refreshed on each packet)
static float cached_battery_voltage = 0.0f;
static float cached_battery_percent = 0.0f;

// Called once per uplink cycle to refresh ADC reading
void updateBatteryEstimate() {
    cached_battery_voltage = readBatteryVoltage();
    cached_battery_percent = voltageToPercentage(cached_battery_voltage);

    estimated_battery_mah = (cached_battery_percent / 100.0f) * BATTERY_CAPACITY_MAH;
    totalCycles++;

    Serial.printf("  🔋 Battery: %.2fV (%.1f%%) | %.1f mAh est.\n",
                  cached_battery_voltage, cached_battery_percent, estimated_battery_mah);
    Serial.printf("  Total cycles: %d\n", totalCycles);

    if (cached_battery_percent < LOW_BATTERY_THRESHOLD) {
        Serial.printf("  ⚠️ WARNING: Battery LOW (%.1f%%)\n", cached_battery_percent);
    }
    if (cached_battery_percent < CRITICAL_BATTERY_THRESHOLD) {
        Serial.printf("  ⚠️⚠️ CRITICAL: Battery very low (%.1f%%)\n", cached_battery_percent);
    }
}

// Getter functions (same names — rest of code works unchanged)
float getEstimatedBatteryPercentage() {
    return cached_battery_percent;
}

float getEstimatedBatteryVoltage() {
    return cached_battery_voltage;
}

// Charging is handled by the BQ24074 hardware
void startBatteryCharging() {
    float v = readBatteryVoltage();
    float p = voltageToPercentage(v);
    Serial.printf("  🔋 Battery reading: %.2fV (%.1f%%)\n", v, p);
    Serial.println("     Charging is handled by the BQ24074 solar charger.");
}

void resetBatteryEstimate() {
    startBatteryCharging();
}

// ============================================================================
// --- OLED Display Functions ---
// ============================================================================

void updateOLED(const char* line1, const char* line2, const char* line3, const char* line4) {
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 0);
    display.println(line1);
    display.println(line2);
    display.println(line3);
    display.println(line4);
    display.display();
}

void updateOLED_WSONN(uint32_t count, float temp, float press, float hum,
                       float attack, float explore, float mutation, float decay,
                       float battery, float snr, float rssi) {
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 0);

    char line1[32];
    snprintf(line1, sizeof(line1), "Pkt #%d Bat:%.1f%%", count, battery);
    display.println(line1);

    char line2[32];
    snprintf(line2, sizeof(line2), "T:%.1fC P:%.0fhPa", temp, press);
    display.println(line2);

    char line3[32];
    snprintf(line3, sizeof(line3), "H:%.1f%% R:%ddBm", hum, (int)rssi);
    display.println(line3);

    char line4[32];
    snprintf(line4, sizeof(line4), "A:%.2f E:%.2f M:%.2f", attack, explore, mutation);
    display.println(line4);

    display.display();
}

// ============================================================================
// --- Radio Statistics Functions (with caching) ---
// ============================================================================

float getRadioSNR() {
    if (radio == nullptr) return lastValidSNR;

    float snr = radio->getSNR();

    if (snr != 0.0f && snr > -20.0f && snr < 30.0f) {
        lastValidSNR = snr;
        return snr;
    } else {
        return lastValidSNR;
    }
}

float getRadioRSSI() {
    if (radio == nullptr) return lastValidRSSI;

    float rssi = radio->getRSSI();

    if (rssi != 0.0f && rssi > -150.0f && rssi < 0.0f) {
        lastValidRSSI = rssi;
        return rssi;
    } else {
        return lastValidRSSI;
    }
}

float calculatePacketLoss() {
    if (totalPacketsSent == 0) return 0.0f;
    return (float)packetLossCount / (float)totalPacketsSent;
}

// ============================================================================
// --- Downlink Command Processing ---
// ============================================================================

void processDownlinkCommand(uint8_t* data, size_t len) {
    if (len < 2) {
        Serial.println("⚠️ Downlink too short");
        return;
    }

    uint8_t command = data[0];
    uint8_t cmdLen = data[1];

    Serial.printf("📥 Processing command: 0x%02X (len: %d)\n", command, cmdLen);

    switch(command) {
        case 0x01: {
            if (cmdLen >= 2) {
                uint16_t interval = (data[2] << 8) | data[3];
                newUplinkInterval = interval * 1000;
                Serial.printf("  ✅ Uplink interval set to %d seconds\n", interval);
                char msg[32];
                snprintf(msg, sizeof(msg), "Interval: %ds", interval);
                updateOLED("Command:", "Interval Set", msg, "");
            }
            break;
        }

        case 0x02: {
            if (cmdLen >= 1) {
                adrEnabled = data[2] > 0;
                if (node != nullptr) node->setADR(adrEnabled);
                Serial.printf("  ✅ ADR set to: %s\n", adrEnabled ? "ON" : "OFF");
                char msg[32];
                snprintf(msg, sizeof(msg), "ADR: %s", adrEnabled ? "ON" : "OFF");
                updateOLED("Command:", msg, "", "");
            }
            break;
        }

        case 0x03: {
            if (cmdLen >= 1) {
                targetDataRate = data[2];
                if (node != nullptr) node->setDatarate(targetDataRate);
                Serial.printf("  ✅ Data Rate set to: %d\n", targetDataRate);
                char msg[32];
                snprintf(msg, sizeof(msg), "DR set to %d", targetDataRate);
                updateOLED("Command:", msg, "", "");
            }
            break;
        }

        case 0x04: {
            Serial.println("  📊 Status requested");
            float v = readBatteryVoltage();
            float p = voltageToPercentage(v);
            char msg[32];
            snprintf(msg, sizeof(msg), "%.2fV %.0f%%", v, p);
            updateOLED("Status:", msg, "Reported", "");
            break;
        }

        case 0x05: {
            float v = readBatteryVoltage();
            float p = voltageToPercentage(v);
            Serial.printf("  🔋 Battery reading: %.2fV (%.1f%%)\n", v, p);
            char msg[32];
            snprintf(msg, sizeof(msg), "%.2fV %.0f%%", v, p);
            updateOLED("Battery:", msg, "", "");
            break;
        }

        default:
            Serial.printf("  ⚠️ Unknown command: 0x%02X\n", command);
            char msg[32];
            snprintf(msg, sizeof(msg), "Unknown: 0x%02X", command);
            updateOLED("Unknown Cmd:", msg, "", "");
            break;
    }

    delay(2000);
}

// ============================================================================
// --- TensorFlow Model Functions ---
// ============================================================================

bool init_wso_nn_model() {
    resolver.AddFullyConnected();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddReshape();
    resolver.AddSoftmax();
    resolver.AddLogistic();

    tflite_model = tflite::GetModel(wsonn_model_quantized_tflite);
    if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("❌ Model schema version mismatch!");
        return false;
    }

    static tflite::MicroInterpreter static_interpreter(tflite_model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("❌ AllocateTensors() failed!");
        return false;
    }

    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);

    Serial.println("✅ WSO-NN Model initialized!");
    return true;
}

WSOParameters predict_wso_parameters(float snr, float rssi, float battery,
                                      float packet_loss, float diversity,
                                      float stagnation, float complexity) {
#if TEST_MODE
    // ========================================================================
    // TEST MODE: STRONG SIGNAL configuration for maximum attack ratio
    // Freezes all inputs except battery
    // ========================================================================
    snr = 15.0f;
    rssi = -70.0f;
    packet_loss = 0.02f;
    diversity = 0.45f;
    stagnation = 0.05f;
    complexity = 0.20f;
#endif

    float raw_inputs[7] = {snr, rssi, battery, packet_loss, diversity, stagnation, complexity};

    for (int i = 0; i < 7; i++) {
        float norm_val = (raw_inputs[i] - X_MIN[i]) / (X_MAX[i] - X_MIN[i]);
        if (norm_val < 0.0f) norm_val = 0.0f;
        if (norm_val > 1.0f) norm_val = 1.0f;
        input_tensor->data.f[i] = norm_val;
    }

    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("❌ Failed to invoke TFLite model!");
        return {0.0f, 0.0f, 0.0f, 0.0f};
    }

    WSOParameters params;
    params.attack_ratio       = output_tensor->data.f[0];
    params.exploration_factor = output_tensor->data.f[1];
    params.mutation_strength  = output_tensor->data.f[2];
    params.weight_decay       = output_tensor->data.f[3];

    if (params.attack_ratio < 0.0f) params.attack_ratio = 0.0f;
    if (params.attack_ratio > 1.0f) params.attack_ratio = 1.0f;
    if (params.exploration_factor < 0.0f) params.exploration_factor = 0.0f;
    if (params.exploration_factor > 1.0f) params.exploration_factor = 1.0f;
    if (params.mutation_strength < 0.0f) params.mutation_strength = 0.0f;
    if (params.mutation_strength > 1.0f) params.mutation_strength = 1.0f;
    if (params.weight_decay < 0.0f) params.weight_decay = 0.0f;
    if (params.weight_decay > 1.0f) params.weight_decay = 1.0f;

    return params;
}

// ============================================================================
// --- SPI Test Function ---
// ============================================================================

void testSPI() {
    Serial.println("\n=== SPI TEST ===");

    pinMode(LORA_CS, OUTPUT);
    pinMode(LORA_SCK, OUTPUT);
    pinMode(LORA_MOSI, OUTPUT);
    pinMode(LORA_MISO, INPUT);

    digitalWrite(LORA_CS, HIGH);
    digitalWrite(LORA_SCK, LOW);
    digitalWrite(LORA_MOSI, HIGH);

    SPI.begin(LORA_SCK, LORA_MISO, LORA_MOSI, LORA_CS);
    SPI.setFrequency(1000000);

    SPI.beginTransaction(SPISettings(1000000, MSBFIRST, SPI_MODE0));
    digitalWrite(LORA_CS, LOW);
    delayMicroseconds(10);

    uint8_t version = SPI.transfer(0x42);
    version = SPI.transfer(0x00);

    digitalWrite(LORA_CS, HIGH);
    SPI.endTransaction();

    Serial.printf("Manual SPI test - Version: 0x%02X\n", version);

    if (version == 0x12 || version == 0x41) {
        Serial.println("✅ SPI working (module detected)!");
    } else {
        Serial.println("❌ SPI failed!");
        Serial.println("Check wiring and power.");
    }
}

// ============================================================================
// --- Radio Initialization ---
// ============================================================================

bool initRadio() {
    if (node != nullptr) { delete node; node = nullptr; }
    if (radio != nullptr) { delete radio; radio = nullptr; }
    if (module != nullptr) { delete module; module = nullptr; }

    delay(100);

    pinMode(LORA_CS, OUTPUT);
    digitalWrite(LORA_CS, HIGH);
    pinMode(LORA_RST, OUTPUT);
    digitalWrite(LORA_RST, HIGH);

    digitalWrite(LORA_RST, LOW);
    delay(100);
    digitalWrite(LORA_RST, HIGH);
    delay(200);

    module = new Module(LORA_CS, LORA_DIO0, LORA_RST);
    if (module == nullptr) return false;

    radio = new SX1276(module);
    if (radio == nullptr) return false;

    Serial.print("Initializing radio at ");
    Serial.print(LORA_FREQ);
    Serial.print(" MHz...");

    int16_t state = radio->begin(LORA_FREQ);

    if (state != RADIOLIB_ERR_NONE) {
        Serial.printf(" failed: %d\n", state);
        return false;
    }

    Serial.println(" OK");
    radio->setOutputPower(14);
    return true;
}

// ============================================================================
// --- Join Function (with session persistence) ---
// ============================================================================

bool joinLoRaWAN() {
    if (hasValidSession && savedDevAddr != 0) {
        Serial.println("🔄 Found saved session, attempting to restore...");
        updateOLED("Restoring", "Session...", "", "");

        node = new LoRaWANNode(radio, Region, CN470_SUB_BAND);
        if (node == nullptr) return false;

        int16_t state = node->beginOTAA(joinEUI, devEUI, nwkKey, appKey);
        if (state != RADIOLIB_ERR_NONE) {
            Serial.printf("Node init failed: %d\n", state);
            return false;
        }

        Serial.println("✅ Session restored successfully!");
        joined = true;
        updateOLED("✅ JOINED!", "LoRaWAN CN470", "Session Restored", "Ready");
        delay(2000);
        return true;
    }

    Serial.println("🔑 No saved session, performing OTAA join...");
    updateOLED("Joining...", "LoRaWAN CN470", "Sub-band 1", "");

    node = new LoRaWANNode(radio, Region, CN470_SUB_BAND);
    if (node == nullptr) return false;

    int16_t state = node->beginOTAA(joinEUI, devEUI, nwkKey, appKey);
    if (state != RADIOLIB_ERR_NONE) {
        Serial.printf("Node init failed: %d\n", state);
        return false;
    }
    Serial.println("Node configured");

    int joinAttempts = 0;
    state = node->activateOTAA();
    while (state != RADIOLIB_LORAWAN_NEW_SESSION && joinAttempts < 30) {
        joinAttempts++;
        Serial.print(".");
        char msg[32];
        snprintf(msg, sizeof(msg), "Attempt %d/30", joinAttempts);
        updateOLED("Joining...", msg, "Sub-band 1", "");
        delay(2000);
        state = node->activateOTAA();
    }

    if (state == RADIOLIB_LORAWAN_NEW_SESSION) {
        Serial.println("\n✅ JOIN SUCCESS!");
        updateOLED("✅ JOINED!", "LoRaWAN CN470", "WSO-NN v2.0", "Ready");

        node->setADR(false);
        node->setDatarate(5);
        radio->setOutputPower(14);

        savedDevAddr = 0;
        hasValidSession = true;
        Serial.println("💾 Session saved to RTC memory");

        delay(2000);
        return true;
    } else {
        Serial.printf("\n❌ Join failed after 30 attempts: %d\n", state);
        char msg[32];
        snprintf(msg, sizeof(msg), "Code: %d", state);
        updateOLED("JOIN FAILED", msg, "Restarting...", "");
        return false;
    }
}

// ============================================================================
// --- Deep Sleep Function (DISABLED) ---
// ============================================================================

void goToDeepSleep(uint32_t sleepSeconds) {
    Serial.printf("⏳ Waiting %d seconds (deep sleep DISABLED)...\n", sleepSeconds);

    char msg[32];
    snprintf(msg, sizeof(msg), "Wait %ds", sleepSeconds);
    updateOLED("Waiting...", msg, "", "");

    delay(sleepSeconds * 1000);
}

// ============================================================================
// --- Setup ---
// ============================================================================

void setup() {
    Serial.begin(115200);
    delay(2000);

    bootCount++;
    packetCount = savedPacketCount;

    Serial.println("\n=== ESP32-P4 LoRaWAN OTAA with WSO-NN v2.0 ===");
    Serial.println("⚠️ DEEP SLEEP DISABLED - Continuous mode");

#if TEST_MODE
    Serial.println("🧪 TEST MODE ENABLED - Battery sweep isolation");
    Serial.println("   Strong signal configuration (SNR=15, RSSI=-70)");
    Serial.println("   Expected attack ratio range: 0.75 → 0.09");
#else
    Serial.println("🔬 NORMAL MODE - Real sensor inputs");
    Serial.println("   SNR/RSSI caching enabled");
#endif

    Serial.printf("Boot #%d\n", bootCount);
    Serial.printf("Packet count: %d\n", packetCount);

    if (hasValidSession) {
        Serial.println("🔄 Valid session found in RTC memory");
    } else {
        Serial.println("🔑 No session found - will perform OTAA join");
    }

    // ========================================================================
    // Initialize battery ADC (GPIO23 with 10k / 10k divider)
    // Uses CALIBRATED ADC via analogReadMilliVolts()
    // ========================================================================
    pinMode(BATTERY_ADC_PIN, INPUT);
    delay(10);

    // Diagnostic — prints both raw and calibrated values
    uint32_t raw_test = analogRead(BATTERY_ADC_PIN);
    float mv_test = analogReadMilliVolts(BATTERY_ADC_PIN);
    Serial.printf("🔧 ADC raw = %u | calibrated = %.1f mV | battery = %.2fV\n",
                  raw_test, mv_test, (mv_test / 1000.0f) * DIVIDER_RATIO);

    float initial_v = readBatteryVoltage();
    float initial_pct = voltageToPercentage(initial_v);
    Serial.printf("🔋 Initial battery reading: %.2fV (%.1f%%)\n", initial_v, initial_pct);

    if (initial_v < 2.0f || initial_v > 4.5f) {
        Serial.println("⚠️ WARNING: Battery voltage out of expected range!");
        Serial.println("   Check your voltage divider and wiring on GPIO23.");
    }

    cached_battery_voltage = initial_v;
    cached_battery_percent = initial_pct;

    if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
        Serial.println("OLED failed!");
    } else {
        display.clearDisplay();
        display.setTextSize(1);
        display.setTextColor(SSD1306_WHITE);
        display.setCursor(0, 0);
        display.println("LoRaWAN OTAA");
        display.println("WSO-NN v2.0");
#if TEST_MODE
        display.println("TEST MODE");
#else
        display.println("Continuous Mode");
#endif
        display.display();
    }

    Serial.println("Initializing WSO-NN AI model...");
    updateOLED("WSO-NN", "Initializing...", "", "");

    if (!init_wso_nn_model()) {
        Serial.println("❌ WSO-NN Model initialization failed!");
        updateOLED("ERROR", "WSO-NN Failed", "", "");
        while (1) delay(1000);
    }

    Serial.printf("DevEUI: 0x%016llX\n", devEUI);
    Serial.printf("JoinEUI: 0x%016llX\n", joinEUI);
    Serial.printf("Sub-Band: %d\n", CN470_SUB_BAND);
    Serial.printf("Frequency: %.1f MHz\n", LORA_FREQ);

    Wire.begin(I2C_SDA, I2C_SCL);

    if (!bme.begin(BME_ADDR)) {
        Serial.println("BME280 failed!");
    } else {
        Serial.println("BME280 OK");
    }

    testSPI();

    if (!initRadio()) {
        Serial.println("❌ Radio initialization failed!");
        char msg[32];
        snprintf(msg, sizeof(msg), "Radio init failed");
        updateOLED("ERROR", msg, "Retrying...", "");

        delay(1000);
        if (!initRadio()) {
            updateOLED("ERROR", "Radio Failed", "Rebooting...", "");
            while (1) delay(1000);
        }
    }

    joined = joinLoRaWAN();
    if (!joined) {
        updateOLED("ERROR", "Join Failed", "Rebooting...", "");
        while (1) delay(1000);
    }

    Serial.println("✅ Ready!");
    Serial.printf("🔋 Battery Monitoring: CALIBRATED ADC on GPIO%d\n", BATTERY_ADC_PIN);
    Serial.printf("   Divider: 10k / 10k (ratio = 2.0)\n");
    Serial.printf("   Capacity: %.1f mAh\n", BATTERY_CAPACITY_MAH);
    Serial.printf("📡 Node will send data every %d seconds\n", UPLINK_INTERVAL / 1000);
    Serial.println("⏳ Deep sleep is DISABLED");
}

// ============================================================================
// --- Loop ---
// ============================================================================

void loop() {
    if (joined && (millis() - lastUplink > UPLINK_INTERVAL)) {
        float temperature = bme.readTemperature();
        float pressure = bme.readPressure() / 100.0F;
        float humidity = bme.readHumidity();

        packetCount++;
        savedPacketCount = packetCount;
        totalPacketsSent++;

        Serial.printf("\n📊 Packet #%d\n", packetCount);
        Serial.printf("  Temp: %.2f C\n", temperature);
        Serial.printf("  Press: %.2f hPa\n", pressure);
        Serial.printf("  Hum: %.2f %%\n", humidity);

        // ---- Real battery reading ----
        updateBatteryEstimate();
        float batteryVoltage = getEstimatedBatteryVoltage();
        float batteryPercent = getEstimatedBatteryPercentage();

        float snr = getRadioSNR();
        float rssi = getRadioRSSI();
        float packetLoss = calculatePacketLoss();

        Serial.printf("  SNR: %.1f dB, RSSI: %.1f dBm\n", snr, rssi);
        Serial.printf("  Packet Loss: %.1f%%\n", packetLoss * 100);

        float diversity = (snr + 20) / 40.0f;
        if (diversity < 0) diversity = 0;
        if (diversity > 1) diversity = 1;

        float stagnation = packetLoss * 2.0f;
        if (stagnation > 1) stagnation = 1;

        float complexity = 0.2f + (packetCount % 10) / 50.0f;
        if (complexity > 0.5) complexity = 0.5;

        Serial.println("🤖 Running WSO-NN inference...");
        WSOParameters wso = predict_wso_parameters(snr, rssi, batteryPercent,
                                                    packetLoss, diversity,
                                                    stagnation, complexity);

        Serial.printf("  Attack Ratio: %.3f\n", wso.attack_ratio);
        Serial.printf("  Exploration: %.3f\n", wso.exploration_factor);
        Serial.printf("  Mutation: %.3f\n", wso.mutation_strength);
        Serial.printf("  Weight Decay: %.3f\n", wso.weight_decay);

        updateOLED_WSONN(packetCount, temperature, pressure, humidity,
                         wso.attack_ratio, wso.exploration_factor,
                         wso.mutation_strength, wso.weight_decay,
                         batteryPercent, snr, rssi);

        // 16-byte payload
        uint8_t payload[16];
        int16_t temp_int = (int16_t)(temperature * 100);
        uint16_t press_int = (uint16_t)(pressure * 10);
        uint16_t hum_int = (uint16_t)(humidity * 100);

        uint8_t atk_q = (uint8_t)(wso.attack_ratio * 255.0f);
        uint8_t exp_q = (uint8_t)(wso.exploration_factor * 255.0f);
        uint8_t mut_q = (uint8_t)(wso.mutation_strength * 255.0f);
        uint8_t wd_q = (uint8_t)(wso.weight_decay * 255.0f);
        uint8_t bat_q = (uint8_t)(batteryPercent * 2.55f);
        uint8_t loss_q = (uint8_t)(packetLoss * 255.0f);

        uint32_t pktCount = packetCount;

        payload[0] = temp_int >> 8;
        payload[1] = temp_int & 0xFF;
        payload[2] = press_int >> 8;
        payload[3] = press_int & 0xFF;
        payload[4] = hum_int >> 8;
        payload[5] = hum_int & 0xFF;
        payload[6] = atk_q;
        payload[7] = exp_q;
        payload[8] = mut_q;
        payload[9] = wd_q;
        payload[10] = bat_q;
        payload[11] = loss_q;
        payload[12] = (pktCount >> 24) & 0xFF;
        payload[13] = (pktCount >> 16) & 0xFF;
        payload[14] = (pktCount >> 8) & 0xFF;
        payload[15] = pktCount & 0xFF;

        Serial.print("📡 Sending uplink (16 bytes) on port 1...");

        uint8_t downlink[64];
        size_t downlinkSize = 0;

        delay(100);

        int16_t state = node->sendReceive(payload, sizeof(payload), 1,
                                          downlink, &downlinkSize, false);

        delay(3000);

        if (downlinkSize > 0) {
            Serial.printf("\n📥 Received downlink (%d bytes)\n", downlinkSize);
            Serial.print("  Raw: ");
            for (size_t i = 0; i < downlinkSize; i++) {
                Serial.printf("%02X ", downlink[i]);
            }
            Serial.println();
            processDownlinkCommand(downlink, downlinkSize);
        } else {
            Serial.println("\n📭 No downlink received");
        }

        if (state >= 0) {
            Serial.println(" ✅ Uplink OK");
            lastUplink = millis();
        } else {
            Serial.printf(" ❌ failed: %d\n", state);
            packetLossCount++;
            char msg[32];
            snprintf(msg, sizeof(msg), "Code: %d", state);
            updateOLED("ERROR", "Uplink Failed", msg, "");
            delay(2000);
            updateOLED_WSONN(packetCount, temperature, pressure, humidity,
                             wso.attack_ratio, wso.exploration_factor,
                             wso.mutation_strength, wso.weight_decay,
                             batteryPercent, snr, rssi);
        }

        lastUplink = millis();
        Serial.printf("⏳ Next uplink in %d seconds...\n", UPLINK_INTERVAL / 1000);
    }

    delay(1000);
}
