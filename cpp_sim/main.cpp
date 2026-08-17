#include <iostream>
#include <unistd.h>
#include <signal.h>
#include <fstream>
#include <sstream>
#include <string>

#include "Data.hpp"
#include "DataFilter.hpp"
#include "ConvertAndPrepareData.hpp"
#include "ErrorDetector.hpp"
#include "SimulatedFesto.hpp"

// Global simulation instance (accessible to LLM controller)
SimulatedFesto* g_simulatedFesto = nullptr;

// Flag for graceful shutdown
volatile bool g_running = true;

// Shared command file written by the Colab bridge
//static const std::string LLM_COMMANDS_PATH =
//    "/scratch/rayarvid/Experiments/exp08/festo_live/llm_control.json";

// static const std::string LLM_COMMANDS_PATH =
 //    "/scratch/rayarvid/Experiments/exp05/festo_live/llm_control.json";
//static const std::string LLM_COMMANDS_PATH =
//    "/scratch/rayarvid/Experiments/Experiment_2_equal_cycles/festo_live/llm_control.json";

static const std::string LLM_COMMANDS_PATH =
    "/scratch/rayarvid/Experiments/Experiment_3_equal_cycles/festo_live/llm_control.json";

void signalHandler(int signum) {
    std::cout << "Interrupt signal (" << signum << ") received. Shutting down..." << std::endl;
    g_running = false;
}

// Function to initialize simulation with specific conditions
void initializeSimulation(float upperLevel = 20.0f, float lowerLevel = 80.0f, float temp = 25.0f) {
    if (g_simulatedFesto == nullptr) {
        g_simulatedFesto = new SimulatedFesto();
    }
    g_simulatedFesto->initialize(upperLevel, lowerLevel, temp);
    std::cout << "Simulation initialized:" << std::endl;
    std::cout << "  Upper tank level: " << upperLevel << "%" << std::endl;
    std::cout << "  Lower tank level: " << lowerLevel << "%" << std::endl;
    std::cout << "  Temperature: " << temp << "°C" << std::endl;
    std::cout << "  Target level: " << SimulatedFesto::getTargetLevel() << "% ± "
              << SimulatedFesto::getLevelTolerance() << "%" << std::endl;
    std::cout << "  Target temp: " << SimulatedFesto::getTargetTemperature() << "°C ± "
              << SimulatedFesto::getTempTolerance() << "°C" << std::endl;
}

// Function for LLM controller to send commands to simulation
bool sendLLMCommand(bool pumpOn, float pumpPower, bool heaterOn, bool valveOpen) {
    if (g_simulatedFesto == nullptr) return false;

    uint32_t digital_out = 0;
    uint32_t analog_out[2] = {0};

    digital_out |= (heaterOn ? 0x01 : 0x00) << 1;
    digital_out |= (0x01 << 2);  // Pump mode: analog
    digital_out |= (pumpOn ? 0x01 : 0x00) << 3;
    digital_out |= (valveOpen ? 0x01 : 0x00) << 4;

    pumpPower = std::max(0.0f, std::min(100.0f, pumpPower));
    analog_out[0] = static_cast<uint32_t>(pumpPower * 100.0f);

    g_simulatedFesto->writeActuators(digital_out, analog_out);

    float upperLevel, lowerLevel, temp;
    bool actualPumpOn, actualHeaterOn, actualValveOpen;
    g_simulatedFesto->getState(upperLevel, lowerLevel, temp, actualPumpOn, actualHeaterOn, actualValveOpen);

    bool safetyOverride = (pumpOn != actualPumpOn) || (heaterOn != actualHeaterOn);
    return !safetyOverride;
}

// Read latest command from llm_commands.json and apply it
void applyLatestLLMCommand() {
    std::ifstream in(LLM_COMMANDS_PATH.c_str());
    if (!in.is_open()) {
        return;
    }

    std::stringstream buffer;
    buffer << in.rdbuf();
    const std::string json = buffer.str();

    auto extractBool = [&](const std::string& key, bool defaultValue) {
        const std::string pattern = "\"" + key + "\"";
        std::size_t pos = json.find(pattern);
        if (pos == std::string::npos) return defaultValue;
        pos = json.find(':', pos);
        if (pos == std::string::npos) return defaultValue;
        const std::string tail = json.substr(pos + 1);
        if (tail.find("true") != std::string::npos) return true;
        if (tail.find("false") != std::string::npos) return false;
        return defaultValue;
    };

    auto extractFloat = [&](const std::string& key, float defaultValue) {
        const std::string pattern = "\"" + key + "\"";
        std::size_t pos = json.find(pattern);
        if (pos == std::string::npos) return defaultValue;
        pos = json.find(':', pos);
        if (pos == std::string::npos) return defaultValue;
        std::size_t start = json.find_first_of("-0123456789.", pos + 1);
        if (start == std::string::npos) return defaultValue;
        std::size_t end = json.find_first_not_of("0123456789.-", start);
        return std::stof(json.substr(start, end - start));
    };

    const bool pumpOn = extractBool("pump_on", false);
    const float pumpPower = extractFloat("pump_power", 0.0f);
    const bool heaterOn = extractBool("heater_on", false);
    const bool valveOpen = extractBool("valve_open", false);

    sendLLMCommand(pumpOn, pumpPower, heaterOn, valveOpen);
}

int main(int argc, const char * argv[]) {
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    uint32_t minimal_leftTank_waterlevel = 0x0;
    uint32_t maximum_leftTank_waterlevel = 0x80e8;
    uint32_t minimal_pipeline_waterflow = 0x0;
    uint32_t maximum_pipeline_waterflow = 0x80e8;
    uint32_t minimal_pipeline_pressure = 0x0;
    uint32_t maximum_pipeline_pressure = 0x80e8;
    uint32_t minimal_rightTank_temperature = 0x0;
    uint32_t maximum_rightTank_temperature = 0x7918;
    uint32_t minimal_pipeline_saturation = 0x0;
    uint32_t maximum_pipeline_saturation = 0x80e8;
    uint32_t minimal_pump_power = 0x0;
    uint32_t maximum_pump_power = 0xd9cf;
    uint32_t minimal_lowerValve_flowrate = 0x0;
    uint32_t maximum_lowerValve_flowrate = 0xd9cf;

    ErrorDetector detective(15);
    uint32_t time = 100000;
    Data data;

    DataFilter filter(minimal_leftTank_waterlevel, maximum_leftTank_waterlevel, minimal_pipeline_waterflow, maximum_pipeline_waterflow, minimal_pipeline_pressure, maximum_pipeline_pressure, minimal_rightTank_temperature, maximum_rightTank_temperature, minimal_pipeline_saturation, maximum_pipeline_saturation, minimal_pump_power, maximum_pump_power, minimal_lowerValve_flowrate, maximum_lowerValve_flowrate, &data);

    ConvertAndPrepareData capd(&data, &detective);

    initializeSimulation(20.0f, 80.0f, 25.0f);

    bool dataOK;

    std::cout << "Simulation running. Press Ctrl+C to stop." << std::endl;
    std::cout << "Target: Maintain upper tank at " << SimulatedFesto::getTargetLevel()
              << "% level and " << SimulatedFesto::getTargetTemperature() << "°C" << std::endl;

    while(g_running){
        applyLatestLLMCommand();
        dataOK = filter.routine();
        capd.runRoutine(dataOK);

        static int iterCount = 0;
        if (++iterCount % 50 == 0) {
            float upperLevel, lowerLevel, temp;
            bool pumpOn, heaterOn, valveOpen;
            g_simulatedFesto->getState(upperLevel, lowerLevel, temp, pumpOn, heaterOn, valveOpen);

            std::cout << "[" << iterCount << "] Upper: " << upperLevel
                      << "%, Lower: " << lowerLevel
                      << "%, Temp: " << temp << "°C";
            std::cout << " | Pump: " << (pumpOn ? "ON" : "OFF")
                      << " (" << data.getPump_power() << "%)";
            std::cout << " | Heater: " << (heaterOn ? "ON" : "OFF");
            std::cout << " | Valve: " << (valveOpen ? "OPEN" : "CLOSED") << std::endl;
        }

        usleep(time);
    }

    std::cout << "Cleaning up..." << std::endl;
    if (g_simulatedFesto != nullptr) {
        delete g_simulatedFesto;
        g_simulatedFesto = nullptr;
    }

    std::cout << "Shutdown complete." << std::endl;
    return 0;
}
