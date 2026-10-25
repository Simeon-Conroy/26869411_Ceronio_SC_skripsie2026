#ifndef IMU_SELECT_H
#define IMU_SELECT_H

#include <stdint.h>
#include "driver/gpio.h"
#include "esp_err.h"
                                                              
esp_err_t IMU_SelectInit(const gpio_num_t pins[3]);
esp_err_t IMU_Select(uint8_t address, const gpio_num_t pins[3]);

#endif
