// Class: SimulatedFesto
// This class simulates the Festo two-tank water system physics.
// It replaces direct /dev/mem access with internal state simulation.
// The simulation models: lower tank (source+heater) -> pump -> upper tank -> valve -> lower tank

#ifndef SimulatedFesto_hpp
#define SimulatedFesto_hpp

#include <cstdint>
#include <random>

class SimulatedFesto
{
    // ---------------- Physics Constants (from real observations) ----------------
    
    private:
        // Flow rates (% per 100ms step)
        static constexpr float FILL_RATE_PER_STEP = 3.27f;           // Pump ON: +3.27% per step
        static constexpr float DRAIN_RATE_PER_STEP = 1.58f;          // Valve OPEN: -1.58% per step
        
        // Temperature rates (°C per 100ms step)
        static constexpr float HEATING_RATE_PER_STEP = 0.074f;        // Heater ON: +0.074°C per step
        static constexpr float COOLING_RATE_PER_STEP = 0.0173f;     // Passive cooling: -0.0173°C per step
        
        // Tank configuration
        static constexpr float UPPER_TANK_CAPACITY = 100.0f;          // 100% = full
        static constexpr float LOWER_TANK_CAPACITY = 100.0f;        // 100% = full
        static constexpr float TOTAL_SYSTEM_WATER = 120.0f;         // Total water in system (arbitrary units)
        
        // Hardware safety limits (from real Festo)
        static constexpr float FLOAT_SWITCH_CUTOFF_LEVEL = 47.0f;   // Hardware cuts pump at ~47%
        static constexpr float MAX_SAFE_TEMPERATURE = 38.0f;          // Safety cuts heater at 38°C
        static constexpr float MAX_UPPER_TANK_LEVEL = 70.0f;        // Safety limit from monitoringUnit()
        
        // Target setpoints (for reference, enforced by LLM controller)
        static constexpr float TARGET_LEVEL = 45.0f;                // Target upper tank level
        static constexpr float TARGET_TEMPERATURE = 35.0f;          // Target temperature
        static constexpr float LEVEL_TOLERANCE = 5.0f;              // ±5% tolerance
        static constexpr float TEMP_TOLERANCE = 3.0f;               // ±3°C tolerance
        
        // Thermal inertia (steps before full effect)
        static constexpr uint32_t THERMAL_INERTIA_STEPS = 3;
        
    // ---------------- Internal State Variables ----------------------------------
    
    private:
        // Tank levels (0-100%)
        float upperTankLevel;        // Left tank in original code (measurement tank)
        float lowerTankLevel;        // Right tank in original code (source + heater)
        
        // Temperature (lower tank has heater)
        float waterTemperature;      // °C
        
        // Actuator states (written by controller, read by simulation)
        bool pumpStatus;             // 0 = OFF, 1 = ON
        bool pumpMode;               // 0 = binary, 1 = analog (not used in sim, assume binary)
        float pumpPower;             // 0-100% (analog power level)
        bool heaterStatus;           // 0 = OFF, 1 = ON
        bool upperValveStatus;       // 0 = CLOSED, 1 = OPEN (auto-selected valve)
        bool lowerValveStatus;       // 0 = CLOSED, 1 = OPEN
        bool filterStatus;           // 0 = OFF, 1 = ON
        bool inflowStatus;           // 0 = CLOSED, 1 = OPEN
        
        // Sensor states (read by controller, written by simulation)
        bool rightTankOverflow;        // Upper tank overflow sensor
        bool leftTankFloatSwitch;    // Hardware float switch (~47% cutoff)
        bool rightTankMinSwitch;     // Lower tank minimum level
        bool rightTankMaxSwitch;     // Lower tank maximum level
        bool upperValveClosed;       // Valve position feedback
        bool upperValveOpen;         // Valve position feedback
        
        // Analog sensor values
        float leftTankWaterLevel;    // Same as upperTankLevel (for compatibility)
        float pipelineWaterflow;     // Derived from pump/valve state
        float pipelinePressure;      // Derived from pump state
        float rightTankTemperature;  // Same as waterTemperature
        float pipelineSaturation;    // Fixed or derived
        
        // Thermal inertia tracking
        float heaterHistory[THERMAL_INERTIA_STEPS];
        uint32_t historyIndex;
        
        // Random noise generator
        std::mt19937 rng;
        std::normal_distribution<float> noiseDistribution;
        
    // ---------------- Private Methods ------------------------------------------
    
    private:
        // Update physics based on current actuator states
        void updatePhysics();
        
        // Update sensor values based on current state
        void updateSensors();
        
        // Auto-select recirculation valve based on flow conditions
        void autoSelectValve();
        
        // Apply thermal inertia to heating/cooling
        float applyThermalInertia(float rawTempChange);
        
        // Add sensor noise
        float addNoise(float value, float noisePercent);
        
        // Enforce hardware safety limits
        void enforceHardwareLimits();
        
    // ---------------- Public Methods -------------------------------------------
    
    public:
        // Constructor - initialize simulation state
        SimulatedFesto();
        
        // Destructor
        ~SimulatedFesto();
        
        // Initialize with specific starting conditions
        void initialize(float initialUpperLevel, float initialLowerLevel, float initialTemp);
        
        // Read all sensor values (replaces /dev/mem read)
        // Returns digital_in, analog_in[5], digital_out, analog_out[2] packed format
        void readSensors(uint32_t &digital_in, uint32_t analog_in[5], uint32_t &digital_out, uint32_t analog_out[2]);
        
        // Write actuator commands (replaces /dev/mem write)
        // Takes digital output bits and analog output values
        void writeActuators(uint32_t digital_out, uint32_t analog_out[2]);
        
        // Get current state for debugging
        void getState(float &upperLevel, float &lowerLevel, float &temp, 
                      bool &pumpOn, bool &heaterOn, bool &valveOpen);
        
        // Reset simulation to initial conditions
        void reset();
        
        // Get target setpoints (for LLM controller reference)
        static float getTargetLevel() { return TARGET_LEVEL; }
        static float getTargetTemperature() { return TARGET_TEMPERATURE; }
        static float getLevelTolerance() { return LEVEL_TOLERANCE; }
        static float getTempTolerance() { return TEMP_TOLERANCE; }
};

#endif /* SimulatedFesto_hpp */