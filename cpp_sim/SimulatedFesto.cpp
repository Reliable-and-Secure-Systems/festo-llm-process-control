// Implementation of the SimulatedFesto class
#include "SimulatedFesto.hpp"
#include <cmath>
#include <algorithm>

constexpr float SimulatedFesto::FILL_RATE_PER_STEP;
constexpr float SimulatedFesto::DRAIN_RATE_PER_STEP;
constexpr float SimulatedFesto::HEATING_RATE_PER_STEP;
constexpr float SimulatedFesto::COOLING_RATE_PER_STEP;
constexpr float SimulatedFesto::UPPER_TANK_CAPACITY;
constexpr float SimulatedFesto::LOWER_TANK_CAPACITY;
constexpr float SimulatedFesto::TOTAL_SYSTEM_WATER;
constexpr float SimulatedFesto::FLOAT_SWITCH_CUTOFF_LEVEL;
constexpr float SimulatedFesto::MAX_SAFE_TEMPERATURE;
constexpr float SimulatedFesto::MAX_UPPER_TANK_LEVEL;
constexpr float SimulatedFesto::TARGET_LEVEL;
constexpr float SimulatedFesto::TARGET_TEMPERATURE;
constexpr float SimulatedFesto::LEVEL_TOLERANCE;
constexpr float SimulatedFesto::TEMP_TOLERANCE;
constexpr uint32_t SimulatedFesto::THERMAL_INERTIA_STEPS;

// ---------------- Constructor ---------------------------------------------------
// Initializes the simulation with default starting conditions.
// Default: Lower tank 80%, Upper tank 20%, Temperature 25°C
SimulatedFesto::SimulatedFesto()
    : rng(std::random_device{}()),
      noiseDistribution(0.0f, 1.0f)
{
    // Initialize thermal history to zero
    for (uint32_t i = 0; i < THERMAL_INERTIA_STEPS; ++i) {
        heaterHistory[i] = 0.0f;
    }
    historyIndex = 0;
    
    // Set default initial conditions
    initialize(20.0f, 80.0f, 25.0f);
}

// ---------------- Destructor ------------------------------------------------------
SimulatedFesto::~SimulatedFesto()
{
}

// ---------------- void initialize() ---------------------------------------------
// Sets specific starting conditions for the simulation.
//
// initialUpperLevel: Starting level of upper tank (0-100%)
// initialLowerLevel: Starting level of lower tank (0-100%)  
// initialTemp: Starting water temperature (°C)
void
SimulatedFesto::initialize(float initialUpperLevel, float initialLowerLevel, float initialTemp)
{
    upperTankLevel = std::max(0.0f, std::min(100.0f, initialUpperLevel));
    lowerTankLevel = std::max(0.0f, std::min(100.0f, initialLowerLevel));
    waterTemperature = std::max(15.0f, std::min(95.0f, initialTemp));
    
    // Initialize actuators to OFF
    pumpStatus = false;
    pumpMode = false;
    pumpPower = 0.0f;
    heaterStatus = false;
    upperValveStatus = false;
    lowerValveStatus = false;
    filterStatus = false;
    inflowStatus = false;
    
    // Reset thermal history
    for (uint32_t i = 0; i < THERMAL_INERTIA_STEPS; ++i) {
        heaterHistory[i] = 0.0f;
    }
    historyIndex = 0;
    
    // Update sensors to match initial state
    updateSensors();
}

// ---------------- void readSensors() --------------------------------------------
// Reads all sensor values and packs them into the format expected by DataFilter.
// This replaces the /dev/mem read operation from the real Festo.
//
// digital_in: Packed digital input values (sensors)
// analog_in: Array of 5 analog input values
// digital_out: Packed digital output values (actuator feedback)
// analog_out: Array of 2 analog output values
void
SimulatedFesto::readSensors(uint32_t &digital_in, uint32_t analog_in[5], 
                            uint32_t &digital_out, uint32_t analog_out[2])
{
    // Update physics one step before reading
    updatePhysics();
    
    // Pack digital inputs (sensors)
    digital_in = 0;
    digital_in |= (rightTankOverflow ? 0x01 : 0x00) << 1;      // Bit 1: Overflow
    digital_in |= (leftTankFloatSwitch ? 0x01 : 0x00) << 2;    // Bit 2: Float switch
    digital_in |= (rightTankMinSwitch ? 0x01 : 0x00) << 3;     // Bit 3: Min switch
    digital_in |= (rightTankMaxSwitch ? 0x01 : 0x00) << 4;     // Bit 4: Max switch
    digital_in |= (upperValveClosed ? 0x01 : 0x00) << 5;       // Bit 5: Valve closed
    digital_in |= (upperValveOpen ? 0x01 : 0x00) << 6;         // Bit 6: Valve open
    
    // Pack analog inputs (with noise)
    analog_in[0] = static_cast<uint32_t>(addNoise(leftTankWaterLevel, 2.0f) * 100);     // Level * 100 for precision
    analog_in[1] = static_cast<uint32_t>(addNoise(rightTankTemperature, 1.5f) * 100);  // Temp * 100
    analog_in[2] = static_cast<uint32_t>(addNoise(pipelineWaterflow, 3.0f) * 100);       // Flow * 100
    analog_in[3] = static_cast<uint32_t>(addNoise(pipelinePressure, 2.5f) * 100);        // Pressure * 100
    analog_in[4] = static_cast<uint32_t>(addNoise(pipelineSaturation, 1.0f) * 10000);   // Saturation * 10000
    
    // Pack digital outputs (actuator states for feedback)
    digital_out = 0;
    digital_out |= (heaterStatus ? 0x01 : 0x00) << 1;          // Bit 1: Heater
    digital_out |= (pumpMode ? 0x01 : 0x00) << 2;              // Bit 2: Pump mode
    digital_out |= (pumpStatus ? 0x01 : 0x00) << 3;            // Bit 3: Pump status
    digital_out |= (lowerValveStatus ? 0x01 : 0x00) << 4;      // Bit 4: Lower valve
    digital_out |= (filterStatus ? 0x01 : 0x00) << 5;          // Bit 5: Filter
    digital_out |= (inflowStatus ? 0x01 : 0x00) << 6;          // Bit 6: Inflow
    
    // Pack analog outputs
    analog_out[0] = static_cast<uint32_t>(pumpPower * 100);    // Pump power * 100
    analog_out[1] = 0;                                          // Lower valve flow rate (not used in this sim)
}

// ---------------- void writeActuators() -----------------------------------------
// Writes actuator commands from the controller to the simulation.
// This replaces the /dev/mem write operation from the real Festo.
//
// digital_out: Packed digital output commands
// analog_out: Array of 2 analog output values
void
SimulatedFesto::writeActuators(uint32_t digital_out, uint32_t analog_out[2])
{
    // Extract digital commands
    heaterStatus = (digital_out >> 1) & 0x01;
    pumpMode = (digital_out >> 2) & 0x01;
    pumpStatus = (digital_out >> 3) & 0x01;
    lowerValveStatus = (digital_out >> 4) & 0x01;
    filterStatus = (digital_out >> 5) & 0x01;
    inflowStatus = (digital_out >> 6) & 0x01;
    
    // Extract analog commands
    pumpPower = analog_out[0] / 100.0f;  // Convert from fixed-point
    
    // Auto-select upper valve based on conditions
    autoSelectValve();
    
    // Enforce hardware safety limits immediately
    enforceHardwareLimits();
}

// ---------------- void updatePhysics() ------------------------------------------
// Updates the simulation physics based on current actuator states.
// This is called automatically before each sensor read.
void
SimulatedFesto::updatePhysics()
{
    // Calculate flow based on pump power (proportional to pumpPower%)
    float actualPumpFlow = 0.0f;
    if (pumpStatus && pumpPower > 0.0f) {
        actualPumpFlow = (FILL_RATE_PER_STEP * pumpPower) / 100.0f;
    }
    
    // Calculate drain based on valve status
    float actualDrainFlow = 0.0f;
    if (upperValveStatus) {
        actualDrainFlow = DRAIN_RATE_PER_STEP;
    }
    
    // Update tank levels (conservation of mass)
    // Upper tank: +fill -drain
    // Lower tank: -fill +drain (conservation)
    float upperLevelChange = actualPumpFlow - actualDrainFlow;
    upperTankLevel += upperLevelChange;
    lowerTankLevel -= upperLevelChange;  // Water moves from lower to upper or back
    
    // Clamp levels to physical limits
    upperTankLevel = std::max(0.0f, std::min(UPPER_TANK_CAPACITY, upperTankLevel));
    lowerTankLevel = std::max(0.0f, std::min(LOWER_TANK_CAPACITY, lowerTankLevel));
    
    // Update temperature
    float tempChange = 0.0f;
    
    // Heating: only affects lower tank (where heater is)
    if (heaterStatus) {
        tempChange += HEATING_RATE_PER_STEP;
    }
    
    // Passive cooling: always occurs
    // Cooling rate proportional to temperature above ambient (simplified)
    float ambientTemp = 20.0f;
    if (waterTemperature > ambientTemp) {
        tempChange -= COOLING_RATE_PER_STEP;
    }
    
    // Apply thermal inertia (temperature coasting)
    tempChange = applyThermalInertia(tempChange);
    
    waterTemperature += tempChange;
    
    // Clamp temperature
    waterTemperature = std::max(15.0f, std::min(95.0f, waterTemperature));
    
    // Update sensor values based on new state
    updateSensors();
    
    // Enforce hardware limits
    enforceHardwareLimits();
}

// ---------------- void updateSensors() ------------------------------------------
// Updates all sensor values based on current simulation state.
void
SimulatedFesto::updateSensors()
{
    // Digital sensors
    rightTankOverflow = (upperTankLevel >= 95.0f);           // Overflow at 95%
    leftTankFloatSwitch = (upperTankLevel >= FLOAT_SWITCH_CUTOFF_LEVEL);  // Hardware cutoff ~47%
    rightTankMinSwitch = (lowerTankLevel >= 10.0f);          // Min level in lower tank
    rightTankMaxSwitch = (lowerTankLevel >= 90.0f);          // Max level in lower tank
    upperValveClosed = !upperValveStatus;
    upperValveOpen = upperValveStatus;
    
    // Analog sensors
    leftTankWaterLevel = upperTankLevel;                     // Upper tank level (0-100%)
    rightTankTemperature = waterTemperature;                  // Water temperature (°C)
    
    // Pipeline flow: positive when pumping, negative when draining
    if (pumpStatus && pumpPower > 0.0f) {
        pipelineWaterflow = (FILL_RATE_PER_STEP * pumpPower) / 100.0f;
    } else if (upperValveStatus) {
        pipelineWaterflow = -DRAIN_RATE_PER_STEP;
    } else {
        pipelineWaterflow = 0.0f;
    }
    
    // Pipeline pressure: proportional to pump power
    if (pumpStatus && pumpPower > 0.0f) {
        pipelinePressure = 200.0f + (pumpPower * 2.0f);  // Base 200 mbar + pump contribution
    } else {
        pipelinePressure = 100.0f;  // Static pressure when pump off
    }
    
    // Saturation: fixed or slightly variable
    pipelineSaturation = 85.0f;  // Fixed at 85% for simplicity
}

// ---------------- void autoSelectValve() ----------------------------------------
// Automatically selects which recirculation valve to use based on flow conditions.
// In this simplified model, we use a single upper valve that opens when needed.
void
SimulatedFesto::autoSelectValve()
{
    // Auto-open valve if upper tank is getting full (>60%) to prevent overflow
    // Or if explicitly commanded by lower valve status (for compatibility)
    if (upperTankLevel > 60.0f || lowerValveStatus) {
        upperValveStatus = true;
    } else if (upperTankLevel < 40.0f && !lowerValveStatus) {
        // Close valve if level is low and not explicitly commanded
        upperValveStatus = false;
    }
}

// ---------------- float applyThermalInertia() -----------------------------------
// Applies thermal inertia effect - temperature continues changing for a few steps
// after actuator state changes.
float
SimulatedFesto::applyThermalInertia(float rawTempChange)
{
    // Store current heater state in history
    heaterHistory[historyIndex] = heaterStatus ? HEATING_RATE_PER_STEP : 0.0f;
    historyIndex = (historyIndex + 1) % THERMAL_INERTIA_STEPS;
    
    // Calculate delayed heating contribution
    float delayedHeating = 0.0f;
    for (uint32_t i = 0; i < THERMAL_INERTIA_STEPS; ++i) {
        delayedHeating += heaterHistory[i];
    }
    delayedHeating /= THERMAL_INERTIA_STEPS;
    
    // Blend immediate and delayed effects
    float blendedChange = (rawTempChange * 0.6f) + (delayedHeating * 0.4f);
    
    return blendedChange;
}

// ---------------- float addNoise() ----------------------------------------------
// Adds realistic sensor noise to a value.
float
SimulatedFesto::addNoise(float value, float noisePercent)
{
    // Generate noise: mean 0, std dev = noisePercent% of value
    float noise = noiseDistribution(rng) * (value * noisePercent / 100.0f);
    return value + noise;
}

// ---------------- void enforceHardwareLimits() ----------------------------------
// Enforces hardware safety limits that override software commands.
void
SimulatedFesto::enforceHardwareLimits()
{
    // Hardware float switch cuts pump at ~47% (cannot be overridden)
    if (upperTankLevel >= FLOAT_SWITCH_CUTOFF_LEVEL && pumpStatus) {
        pumpStatus = false;
        pumpPower = 0.0f;
    }
    
    // Safety: cut heater if temperature exceeds 38°C
    if (waterTemperature >= MAX_SAFE_TEMPERATURE && heaterStatus) {
        heaterStatus = false;
    }
    
    // Safety: cut pump if upper tank exceeds 70% (software safety)
    if (upperTankLevel >= MAX_UPPER_TANK_LEVEL && pumpStatus) {
        pumpStatus = false;
        pumpPower = 0.0f;
    }
    
    // Safety: cut pump if lower tank is empty (<5%)
    if (lowerTankLevel < 5.0f && pumpStatus) {
        pumpStatus = false;
        pumpPower = 0.0f;
    }
}

// ---------------- void getState() -------------------------------------------------
// Returns current simulation state for debugging/monitoring.
void
SimulatedFesto::getState(float &upperLevel, float &lowerLevel, float &temp,
                         bool &pumpOn, bool &heaterOn, bool &valveOpen)
{
    upperLevel = upperTankLevel;
    lowerLevel = lowerTankLevel;
    temp = waterTemperature;
    pumpOn = pumpStatus;
    heaterOn = heaterStatus;
    valveOpen = upperValveStatus;
}

// ---------------- void reset() ----------------------------------------------------
// Resets simulation to initial conditions.
void
SimulatedFesto::reset()
{
    initialize(20.0f, 80.0f, 25.0f);
}