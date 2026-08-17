// Class: ConvertAndPrepareData
// MODIFIED: Uses Google Drive shared paths and writes flat JSON for Colab controller

#ifndef ConvertAndPrepareData_hpp
#define ConvertAndPrepareData_hpp

#include "Data.hpp"
#include "ErrorDetector.hpp"
#include <string>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <semaphore.h>

class ConvertAndPrepareData
{
    private:
        Data *data;
        ErrorDetector *detective;
        std::string JSONString = " ";

        // Google Drive shared folder paths
        //std::string dest_path = "/Users/vidyashreerayar/Library/CloudStorage/GoogleDrive-reisenundarbeit@gmail.com/My Drive/festo_exp3/json_data.txt";
        //std::string bin_path = "/Users/vidyashreerayar/Library/CloudStorage/GoogleDrive-reisenundarbeit@gmail.com/My Drive/festo_exp3/data.bin";
        //std::string log_path = "/Users/vidyashreerayar/Library/CloudStorage/GoogleDrive-reisenundarbeit@gmail.com/My Drive/festo_exp3/sensor_log.json";
        //std::string dest_path = "/scratch/rayarvid/Experiments/exp08/festo_live/json_data.txt";
        //std::string bin_path  = "/scratch/rayarvid/Experiments/exp08/festo_live/data.bin";
        //std::string log_path  = "/scratch/rayarvid/Experiments/exp08/festo_live/sensor_log.json";
        
        //std::string dest_path = "/scratch/rayarvid/Experiments/Experiment_2_equal_cycles/festo_live/json_data.txt";
        //std::string bin_path  = "/scratch/rayarvid/Experiments/Experiment_2_equal_cycles/festo_live/data.bin";
        //std::string log_path  = "/scratch/rayarvid/Experiments/Experiment_2_equal_cycles/festo_live/sensor_log_exp07.json";
        std::string dest_path = "/scratch/rayarvid/Experiments/Experiment_3_equal_cycles/festo_live/json_data.txt";
        std::string bin_path  = "/scratch/rayarvid/Experiments/Experiment_3_equal_cycles/festo_live/data.bin";
        std::string log_path  = "/scratch/rayarvid/Experiments/Experiment_3_equal_cycles/festo_live/sensor_log_exp08.json";

        uint32_t counter = 0;
        uint32_t counterLimit = 500;
        uint32_t currentWarning = 0;
        sem_t *sem_data_bin;
        const std::string SEM_NAME = "/sem_data_bin";
        const int SEM_PERMS = (S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP);
        const int SEM_INITIAL_VALUE = 1;

        std::ofstream logFile;
        bool logFileInitialized = false;
        uint32_t logEntryCounter = 0;

    private:
        void writeData();
        void convertData();
        uint32_t monitoringUnit();
        void createJSONString(uint32_t error_a, uint32_t warning_a);
        void writeBinaryData();
        template <typename T> void writeToBinaryFile(T inData, std::ofstream &outStream);
        void logSensorReading(uint32_t error_a, uint32_t warning_a);
        void initializeLogFile();
        void closeLogFile();

    public:
        ConvertAndPrepareData(Data *data, ErrorDetector *detective);
        ~ConvertAndPrepareData();
        int runRoutine(bool dataIsCorrect_a);
};

#endif /* ConvertAndPrepareData_hpp */
