#include "IMU_select.h"
#include "IMU_config.h"
#include "esp_rom_sys.h"


esp_err_t IMU_SelectInit(const gpio_num_t pins[3]) {
         
    gpio_config_t config = {.pin_bit_mask = (1ULL << pins[0]) | (1ULL << pins[1]) | (1ULL << pins[2]),.mode = GPIO_MODE_OUTPUT,.pull_up_en = GPIO_PULLUP_DISABLE,.pull_down_en = GPIO_PULLDOWN_DISABLE,.intr_type = GPIO_INTR_DISABLE};

    esp_err_t err = gpio_config(&config);
    if (err != ESP_OK) {return err;}
                                                               
    return IMU_Select(0, pins);
}

esp_err_t IMU_Select(uint8_t address, const gpio_num_t pins[3]) {
    if (address >= IMU_PER_GROUP) {return ESP_ERR_INVALID_ARG;}

    for (unsigned address_bit_index = 0; address_bit_index < 3; ++address_bit_index) {
                                                                          
        unsigned pin_level = (address >> address_bit_index) & 1U;
        esp_err_t err = gpio_set_level(pins[address_bit_index], pin_level);
        if (err != ESP_OK) {return err;}
    }
                                                                         
    esp_rom_delay_us(IMU_MUX_SETTLE_US);
    return ESP_OK;
}
