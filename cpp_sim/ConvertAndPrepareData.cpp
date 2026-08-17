// Implementation of the ConvertAndPrepareData class (MODIFIED FOR SIMULATION)
#include "ConvertAndPrepareData.hpp"
#include "SimulatedFesto.hpp"
#include <iomanip>
#include <ctime>
#include <sstream>
#include <fstream>

extern SimulatedFesto* g_simulatedFesto;

// Constructor
ConvertAndPrepareData::ConvertAndPrepareData(Data *data, ErrorDetector *detective)
: data(data), detective(detective)
{
    sem_data_bin = sem_open(SEM_NAME.c_str(), O_CREAT, SEM_PERMS, SEM_INITIAL_VALUE);
    if (sem_data_bin == SEM_FAILED) {
        perror("sem_open(3) error in ConvertAndPrepareData-Constructor");
        exit(EXIT_FAILURE);
    }
    initializeLogFile();
}

// Destructor
ConvertAndPrepareData::~ConvertAndPrepareData()
{
    if (sem_close(sem_data_bin) < 0)
        perror("sem_close(3) failed in ConvertAndPrepareData-Destructor");
    closeLogFile();
}

// Initialize log file
void ConvertAndPrepareData::initializeLogFile()
{
    logFile.open(log_path, std::ios::out | std::ios::app);
    if (!logFile.is_open()) {
        std::cerr << "Failed to open sensor log file: " << log_path << std::endl;
        return;
    }
    logFileInitialized = true;
    logFile.seekp(0, std::ios::end);
    if (logFile.tellp() == 0) {
        logFile << "[" << std::endl;
    }
}

// Close log file
void ConvertAndPrepareData::closeLogFile()
{
    if (logFileInitialized && logFile.is_open()) {
        logFile << std::endl << "]" << std::endl;
        logFile.close();
    }
}

// Log sensor reading
void ConvertAndPrepareData::logSensorReading(uint32_t error_a, uint32_t warning_a)
{
    if (!logFileInitialized || !logFile.is_open()) return;

    if (logEntryCounter > 0) logFile << "," << std::endl;

    std::time_t now = std::time(nullptr);
    std::tm* localTime = std::localtime(&now);

    logFile << "  {" << std::endl;
    logFile << "    \"entry\": " << ++logEntryCounter << "," << std::endl;
    logFile << "    \"timestamp\": " << now << "," << std::endl;
    logFile << "    \"datetime\": \""
            << (localTime->tm_year + 1900) << "-"
            << std::setfill('0') << std::setw(2) << (localTime->tm_mon + 1) << "-"
            << std::setfill('0') << std::setw(2) << localTime->tm_mday << " "
            << std::setfill('0') << std::setw(2) << localTime->tm_hour << ":"
            << std::setfill('0') << std::setw(2) << localTime->tm_min << ":"
            << std::setfill('0') << std::setw(2) << localTime->tm_sec << "\"," << std::endl;

    logFile << "    \"sensors\": {" << std::endl;
    logFile << "      \"Overflow\": " << (data->getRightTank_overflow() ? "true" : "false") << "," << std::endl;
    logFile << "      \"FloatSwitch\": " << (data->getLeftTank_floatswitch() ? "true" : "false") << "," << std::endl;
    logFile << "      \"MinSwitch\": " << (data->getRightTank_minswitch() ? "true" : "false") << "," << std::endl;
    logFile << "      \"MaxSwitch\": " << (data->getRightTank_maxswitch() ? "true" : "false") << "," << std::endl;
    logFile << "      \"UpperValve\": " << (data->getUpperValve_open() ? "true" : "false") << std::endl;
    logFile << "    }," << std::endl;

    logFile << "    \"analog\": {" << std::endl;
    logFile << std::fixed << std::setprecision(4);
    logFile << "      \"LeftTankWaterlevel\": " << data->getLeftTank_waterlevel() << "," << std::endl;
    logFile << "      \"RightTankWaterlevel\": " << (1.0f - data->getLeftTank_waterlevel()) << "," << std::endl;
    logFile << "      \"Waterflow\": " << data->getPipeline_waterflow() << "," << std::endl;
    logFile << "      \"Pressure\": " << data->getPipeline_pressure() << "," << std::endl;
    logFile << "      \"Temperature\": " << data->getRightTank_temperature() << "," << std::endl;
    logFile << "      \"Saturation\": " << data->getPipeline_saturation() << std::endl;
    logFile << "    }," << std::endl;

    logFile << "    \"actuators\": {" << std::endl;
    logFile << "      \"Heater\": " << (data->getHeater_status() ? "true" : "false") << "," << std::endl;
    logFile << "      \"PumpMode\": " << (data->getPump_mode() ? "true" : "false") << "," << std::endl;
    logFile << "      \"PumpStatus\": " << (data->getPump_status() ? "true" : "false") << "," << std::endl;
    logFile << "      \"LowerValve\": " << (data->getLowerValve_status() ? "true" : "false") << "," << std::endl;
    logFile << "      \"Filter\": " << (data->getFilter_status() ? "true" : "false") << "," << std::endl;
    logFile << "      \"Inflow\": " << (data->getInflow_status() ? "true" : "false") << "," << std::endl;
    logFile << "      \"PumpPower\": " << data->getPump_power() << "," << std::endl;
    logFile << "      \"LowerValveFlowrate\": " << data->getLowerValve_flowrate() << std::endl;
    logFile << "    }," << std::endl;

    logFile << "    \"status\": {" << std::endl;
    logFile << "      \"ErrorCode\": " << error_a << "," << std::endl;
    logFile << "      \"WarningCode\": " << warning_a << std::endl;
    logFile << "    }" << std::endl;
    logFile << "  }";

    logFile.flush();
}

void ConvertAndPrepareData::writeData()
{
    std::ofstream out(dest_path.c_str(), std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        std::cerr << "Failed to open JSON output file: " << dest_path << std::endl;
        return;
    }
    out << JSONString;
}

void ConvertAndPrepareData::convertData()
{
    // Data is already in engineering units in the simulation build.
}

uint32_t ConvertAndPrepareData::monitoringUnit()
{
    uint32_t warning = 0;

    if (data->getLeftTank_waterlevel() > 65.0f) {
        warning = 1;
    } else if (data->getRightTank_temperature() > 35.0f) {
        warning = 2;
    }

    currentWarning = warning;
    return warning;
}

void ConvertAndPrepareData::createJSONString(uint32_t error_a, uint32_t warning_a)
{
    std::ostringstream json;
    json << std::fixed << std::setprecision(4);
    json << "{\n";
    json << "  \"Overflow\": " << (data->getRightTank_overflow() ? 1 : 0) << ",\n";
    json << "  \"FloatSwitch\": " << (data->getLeftTank_floatswitch() ? 1 : 0) << ",\n";
    json << "  \"MinSwitch\": " << (data->getRightTank_minswitch() ? 1 : 0) << ",\n";
    json << "  \"MaxSwitch\": " << (data->getRightTank_maxswitch() ? 1 : 0) << ",\n";
    json << "  \"UpperValve\": " << (data->getUpperValve_open() ? 1 : 0) << ",\n";
    json << "  \"LeftTankWaterlevel\": " << data->getLeftTank_waterlevel() << ",\n";
    json << "  \"RightTankWaterlevel\": " << (100.0f - data->getLeftTank_waterlevel()) << ",\n";
    json << "  \"Waterflow\": " << data->getPipeline_waterflow() << ",\n";
    json << "  \"Pressure\": " << data->getPipeline_pressure() << ",\n";
    json << "  \"Temperature\": " << data->getRightTank_temperature() << ",\n";
    json << "  \"Heater\": " << (data->getHeater_status() ? 1 : 0) << ",\n";
    json << "  \"PumpMode\": " << (data->getPump_mode() ? 1 : 0) << ",\n";
    json << "  \"PumpStatus\": " << (data->getPump_status() ? 1 : 0) << ",\n";
    json << "  \"LowerValve\": " << (data->getLowerValve_status() ? 1 : 0) << ",\n";
    json << "  \"PumpPower\": " << data->getPump_power() << ",\n";
    json << "  \"LowerValveFlowrate\": " << data->getLowerValve_flowrate() << ",\n";
    json << "  \"Saturation\": " << data->getPipeline_saturation() << ",\n";
    json << "  \"Filter\": " << (data->getFilter_status() ? 1 : 0) << ",\n";
    json << "  \"Inflow\": " << (data->getInflow_status() ? 1 : 0) << ",\n";
    json << "  \"Error\": " << error_a << ",\n";
    json << "  \"Warning\": " << warning_a << "\n";
    json << "}\n";
    JSONString = json.str();
}

template <typename T>
void ConvertAndPrepareData::writeToBinaryFile(T inData, std::ofstream &outStream)
{
    outStream.write(reinterpret_cast<const char*>(&inData), sizeof(T));
}

void ConvertAndPrepareData::writeBinaryData()
{
    if (sem_wait(sem_data_bin) < 0) {
        perror("sem_wait failed in writeBinaryData");
        return;
    }

    std::ofstream out(bin_path.c_str(), std::ios::binary | std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        std::cerr << "Failed to open binary output file: " << bin_path << std::endl;
        sem_post(sem_data_bin);
        return;
    }

    writeToBinaryFile(data->getLeftTank_waterlevel(), out);
    writeToBinaryFile(data->getPipeline_waterflow(), out);
    writeToBinaryFile(data->getPipeline_pressure(), out);
    writeToBinaryFile(data->getRightTank_temperature(), out);
    writeToBinaryFile(data->getPipeline_saturation(), out);
    writeToBinaryFile(data->getPump_power(), out);
    writeToBinaryFile(data->getLowerValve_flowrate(), out);

    out.close();

    if (sem_post(sem_data_bin) < 0) {
        perror("sem_post failed in writeBinaryData");
    }
}

int ConvertAndPrepareData::runRoutine(bool dataIsCorrect_a)
{
    if (true) {  // always emit so the LLM supervisor never reads stale state
        convertData();
        uint32_t warning = monitoringUnit();
        detective->update_log(data->getLeftTank_waterlevel(), data->getUpperValve_open());
        uint32_t error = detective->detect_errors(data->getLeftTank_waterlevel(), data->getPump_status());
        createJSONString(error, warning);
        writeData();
        writeBinaryData();
        logSensorReading(error, warning);
        return 0;
    }
    else
        return -1;
}
