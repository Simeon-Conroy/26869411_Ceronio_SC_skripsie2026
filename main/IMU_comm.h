#ifndef IMU_COMM_H
#define IMU_COMM_H

#include <stdbool.h>
#include <stdint.h>
#include "driver/gpio.h"
#include "driver/spi_master.h"
#include "esp_err.h"

typedef struct {
    spi_host_device_t host;                              
    gpio_num_t mosi;
    gpio_num_t miso;
    gpio_num_t clk;
    gpio_num_t cs;
    gpio_num_t select_pins[3];                                                
    bool direct_cs;                                                                           
    spi_device_handle_t device;                                        
} imu_bus_t;
                                                                            
typedef struct {
                                                                              
    bool acc_fresh;
    bool gyro_fresh;

                                                                                
    int16_t acc_raw[3];
    int16_t gyro_raw[3];

                                                                                 
    int64_t timestamp_us;
} imu_reading_t;

typedef struct {
    uint8_t who_am_i;
    uint8_t ctrl1_xl;
    uint8_t ctrl2_g;
    uint8_t status;
} imu_diagnostic_t;
                                                                  
esp_err_t IMU_BusInit(imu_bus_t *bus);
                                                                                           
esp_err_t IMU_DirectCsInit(imu_bus_t *bus);
                                                                              
esp_err_t IMU_DeviceAttach(imu_bus_t *bus);
                                                                   
esp_err_t IMU_Configure(imu_bus_t *bus, uint8_t address);
                                                                             
esp_err_t IMU_ReadSample(imu_bus_t *bus, uint8_t address, imu_reading_t *reading);
                                                                                 
esp_err_t IMU_ReadDiagnostics(imu_bus_t *bus, uint8_t address, imu_diagnostic_t *diagnostic);

#endif
