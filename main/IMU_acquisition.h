#ifndef IMU_ACQUISITION_H
#define IMU_ACQUISITION_H

#include "IMU_config.h"
#include "IMU_comm.h"
#include "freertos/FreeRTOS.h"

typedef enum {
    IMU_CHANNEL_MIDDLE3 = 0,
    IMU_CHANNEL_MIDDLE2 = 1,
    IMU_CHANNEL_RING3 = 2,
    IMU_CHANNEL_RING2 = 3,
    IMU_CHANNEL_RING1 = 4,
    IMU_CHANNEL_PINKY3 = 5,
    IMU_CHANNEL_PINKY2 = 6,
    IMU_CHANNEL_PINKY1 = 7,
    IMU_CHANNEL_THUMB3 = 8,
    IMU_CHANNEL_THUMB2 = 9,
    IMU_CHANNEL_INDEX3 = 10,
    IMU_CHANNEL_INDEX2 = 11,
    IMU_CHANNEL_INDEX1 = 12,
    IMU_CHANNEL_MIDDLE1 = 13,
    IMU_CHANNEL_PALM = 14,
    IMU_CHANNEL_WRIST = 15,
    IMU_CHANNEL_THUMB_CMC = 16
} imu_channel_id_t;

#define IMU_NAME_LENGTH 12

typedef struct {
    uint8_t channel_id;                                     
    uint8_t spi_bus;                                           
    uint8_t mux_address;                                                    
    char name[IMU_NAME_LENGTH];                                                                       
    bool available;
    uint8_t diagnostic_code;                                                                                
    uint8_t diagnostic_hold_scans;
    uint16_t stale_scans;
    uint16_t recovery_cooldown_scans;
    imu_diagnostic_t diagnostic;
    imu_reading_t reading;                                              
} imu_record_t;
                                                                   
typedef struct {
    uint8_t group;                                     
    uint32_t scan_number;                                                   
    int64_t timestamp_us;                                                      
    imu_record_t imu[IMU_GROUP_RECORD_CAPACITY];                                                                                       
} imu_group_frame_t;

bool IMU_WaitForGroup(uint8_t group, imu_group_frame_t *frame, TickType_t wait_ticks);

#endif
