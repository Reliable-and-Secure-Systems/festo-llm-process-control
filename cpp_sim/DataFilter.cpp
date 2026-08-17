// Implementation of the DataFilter class (MODIFIED FOR SIMULATION)
#include "DataFilter.hpp"
#include "SimulatedFesto.hpp"

// Global simulation instance (externally accessible)
//SimulatedFesto* g_simulatedFesto = nullptr;
extern SimulatedFesto* g_simulatedFesto;


// ---------------- Constructor of the DataFilter -----------------------------------
DataFilter
::DataFilter(uint32_t minimal_leftTank_waterlevel_a, uint32_t maximum_leftTank_waterlevel_a, uint32_t minimal_pipeline_waterflow_a, uint32_t maximum_pipeline_waterflow_a, uint32_t minimal_pipeline_pressure_a, uint32_t maximum_pipeline_pressure_a, uint32_t minimal_rightTank_temperature_a, uint32_t maximum_rightTank_temperature_a, uint32_t minimal_pipeline_saturation_a, uint32_t maximum_pipeline_saturation_a, uint32_t minimal_pump_power_a, uint32_t maximum_pump_power_a, uint32_t minimal_lowerValve_flowrate_a, uint32_t maximum_lowerValve_flowrate_a, Data* data_a)
:
minimal_leftTank_waterlevel(minimal_leftTank_waterlevel_a),
maximum_leftTank_waterlevel(maximum_leftTank_waterlevel_a),
minimal_pipeline_waterflow(minimal_pipeline_waterflow_a),
maximum_pipeline_waterflow(maximum_pipeline_waterflow_a),
minimal_pipeline_pressure(minimal_pipeline_pressure_a),
maximum_pipeline_pressure(maximum_pipeline_pressure_a),
minimal_rightTank_temperature(minimal_rightTank_temperature_a),
maximum_rightTank_temperature(maximum_rightTank_temperature_a),
minimal_pipeline_saturation(minimal_pipeline_saturation_a),
maximum_pipeline_saturation(maximum_pipeline_saturation_a),
minimal_pump_power(minimal_pump_power_a),
maximum_pump_power(maximum_pump_power_a),
minimal_lowerValve_flowrate(minimal_lowerValve_flowrate_a),
maximum_lowerValve_flowrate(maximum_lowerValve_flowrate_a),
data(data_a)
{
}

// ---------------- Destructor ------------------------------------------------------
DataFilter
::~DataFilter()
{
}

// ---------------- void readSharedMemory() -----------------------------------------
// MODIFIED: Reads values from the SimulatedFesto instead of real shared memory.
void
DataFilter
::readSharedMemory(){
    
    // Check if simulation is initialized
    if (g_simulatedFesto == nullptr) {
        // Create simulation instance if not exists
        g_simulatedFesto = new SimulatedFesto();
    }
    
    // Read sensor values from simulation
    uint32_t digital_in = 0;
    uint32_t analog_in[5] = {0};
    uint32_t digital_out = 0;
    uint32_t analog_out[2] = {0};
    
    g_simulatedFesto->readSensors(digital_in, analog_in, digital_out, analog_out);
    
    // Parse digital inputs (sensors)
    data->setRightTank_overflow(((digital_in >> 1) & 0x01));
    data->setLeftTank_floatswitch(((digital_in >> 2) & 0x01));
    data->setRightTank_minswitch(((digital_in >> 3) & 0x01));
    data->setRightTank_maxswitch(((digital_in >> 4) & 0x01));
    data->setUpperValve_closed(((digital_in >> 5) & 0x01));
    data->setUpperValve_open(((digital_in >> 6) & 0x01));
    
    // Parse analog inputs (sensors)
    // Convert from fixed-point back to float
    data->setLeftTank_waterlevel(analog_in[0] / 100.0f);      // Level: stored as % * 100
    data->setRightTank_temperature(analog_in[1] / 100.0f);    // Temp: stored as °C * 100
    data->setPipeline_waterflow(analog_in[2] / 100.0f);       // Flow: stored as L/min * 100
    data->setPipeline_pressure(analog_in[3] / 100.0f);        // Pressure: stored as mbar * 100
    data->setPipeline_saturation(analog_in[4] / 10000.0f);    // Saturation: stored as % * 10000
    
    // Parse digital outputs (actuator feedback from simulation)
    data->setHeater_status(((digital_out >> 1) & 0x01));
    data->setPump_mode(((digital_out >> 2) & 0x01));
    data->setPump_status(((digital_out >> 3) & 0x01));
    data->setLowerValve_status(((digital_out >> 4) & 0x01));
    data->setFilter_status(((digital_out >> 5) & 0x01));
    data->setInflow_status(((digital_out >> 6) & 0x01));
    
    // Parse analog outputs (actuator values from simulation)
    data->setPump_power(analog_out[0] / 100.0f);              // Pump power: 0-100%
    data->setLowerValve_flowrate(analog_out[1] / 100.0f);     // Valve flow: 0-100%
}

// ---------------- bool dataCheck() ------------------------------------------------
bool
DataFilter
::dataCheck(){
    
    if (data->getLeftTank_waterlevel() < minimal_leftTank_waterlevel || data->getLeftTank_waterlevel() > maximum_leftTank_waterlevel) {
	return false;
    }
    else if (data->getPipeline_waterflow() < minimal_pipeline_waterflow || data->getPipeline_waterflow() > maximum_pipeline_waterflow) {
	return false;
    }
    else if (data->getPipeline_pressure() < minimal_pipeline_pressure || data->getPipeline_pressure() > maximum_pipeline_pressure) {
	return false;
    }
    else if (data->getRightTank_temperature() < minimal_rightTank_temperature || data->getRightTank_temperature() > maximum_rightTank_temperature) {
	return false;
    }
    else if (data->getPump_power() < minimal_pump_power || data->getPump_power() > maximum_pump_power) {
	return false;
    }
    else if (data->getLowerValve_flowrate() < minimal_lowerValve_flowrate || data->getLowerValve_flowrate() > maximum_lowerValve_flowrate) {
	return false;
    }
    else if (data->getPipeline_saturation() < minimal_pipeline_saturation || data->getPipeline_saturation() > maximum_pipeline_saturation) {
    return false;
    }
    else return true;
}

// ---------------- bool routine() --------------------------------------------------
bool
DataFilter
::routine(){
    readSharedMemory();
    return dataCheck();
}