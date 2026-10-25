#ifndef IMU_CONFIG_H
#define IMU_CONFIG_H
                                                                                    
#define IMU_GROUP_COUNT 2
#define IMU_PER_GROUP 8
#define IMU_EXTRA_COUNT 1
#define IMU_GROUP_RECORD_CAPACITY (IMU_PER_GROUP + IMU_EXTRA_COUNT)
#define IMU_TOTAL_RECORD_COUNT (IMU_GROUP_COUNT * IMU_PER_GROUP + IMU_EXTRA_COUNT)             
#define IMU_SPI_HZ 500000
#define IMU_CS_SETUP_US 5
#define IMU_CS_HOLD_US 5                                                                         
#define IMU_SCAN_PERIOD_MS 10 
#define IMU_CTRL1_XL_VALUE 0x50
#define IMU_CTRL2_G_VALUE 0x5C                                                                            
#define IMU_BOOT_WAIT_MS 40
#define IMU_RESET_TIMEOUT_MS 100
#define IMU_GYRO_STARTUP_MS 70
#define IMU_STARTUP_DISCARD_SCANS 4
#define IMU_STARTUP_DISCARD_INTERVAL_MS 5                                                        
#define IMU_TASK_STACK_BYTES 6144
#define IMU_TASK_PRIORITY 5               
#define IMU_MUX_SETTLE_US 5

#endif



