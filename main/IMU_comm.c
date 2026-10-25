#include "IMU_comm.h"
#include "IMU_config.h"
#include "IMU_select.h"
#include <string.h>
#include "esp_log.h"
#include "esp_rom_sys.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
                                                       
enum {
    REG_WHO_AM_I = 0x0F,
    REG_CTRL1_XL = 0x10,
    REG_CTRL2_G = 0x11,
    REG_CTRL3_C = 0x12,
    REG_CTRL4_C = 0x13,
    REG_CTRL9_XL = 0x18,
    REG_STATUS = 0x1E,
    REG_GYRO = 0x22,                                                     
    REG_ACC = 0x28,                                                               
};
                                                            
#define SPI_READ_BIT 0x80U
#define STATUS_ACCEL_READY 0x01U
#define STATUS_GYRO_READY 0x02U
#define STATUS_MOTION_READY (STATUS_ACCEL_READY | STATUS_GYRO_READY)

static const char *TAG = "IMU_COMM";

static esp_err_t transfer(imu_bus_t *bus, const uint8_t *tx, uint8_t *rx, size_t byte_count) {
    if (bus == NULL || bus->device == NULL || tx == NULL || byte_count == 0) {
        return ESP_ERR_INVALID_ARG;}
                                                                                                                                                                  
    spi_transaction_t transaction = {.length = byte_count * 8, .tx_buffer = tx, .rx_buffer = rx};
                                                                                                     
    esp_err_t err = spi_device_acquire_bus(bus->device, portMAX_DELAY);
    if (err != ESP_OK) {return err;}
    gpio_set_level(bus->cs, 0);
    esp_rom_delay_us(IMU_CS_SETUP_US);
    err = spi_device_transmit(bus->device, &transaction);
    esp_rom_delay_us(IMU_CS_HOLD_US);
    gpio_set_level(bus->cs, 1);
    spi_device_release_bus(bus->device);
    return err;}
                                                                         
static esp_err_t read_registers(imu_bus_t *bus, uint8_t reg, uint8_t *data, size_t size) {
    if (data == NULL || size == 0 || size > 12 || reg > 0x7F) {return ESP_ERR_INVALID_ARG;}

    uint8_t tx[13] = {0};
    uint8_t rx[13] = {0};
    tx[0] = reg | SPI_READ_BIT;

    esp_err_t err = transfer(bus, tx, rx, size + 1);
    if (err == ESP_OK) {memcpy(data, &rx[1], size);}
    return err;}
                                                                 
static esp_err_t write_register(imu_bus_t *bus, uint8_t reg, uint8_t value) {
    if (reg > 0x7F) {return ESP_ERR_INVALID_ARG;}
                                                                     
    uint8_t tx[2] = {reg & 0x7FU, value};
    uint8_t rx[2] = {0};
    return transfer(bus, tx, rx, sizeof(tx));}
    
static esp_err_t select_sensor_route(imu_bus_t *bus, uint8_t address) {
    if (bus == NULL || address >= IMU_PER_GROUP) {return ESP_ERR_INVALID_ARG;}
    if (bus->direct_cs) {return ESP_OK;}
    return IMU_Select(address, bus->select_pins);
}

static esp_err_t direct_cs_init(imu_bus_t *bus) {
    if (bus == NULL || bus->cs >= GPIO_NUM_MAX) {return ESP_ERR_INVALID_ARG;}
    gpio_config_t config = {.pin_bit_mask = 1ULL << bus->cs,.mode = GPIO_MODE_OUTPUT,.pull_up_en = GPIO_PULLUP_DISABLE,.pull_down_en = GPIO_PULLDOWN_DISABLE,.intr_type = GPIO_INTR_DISABLE};
    esp_err_t err = gpio_config(&config);
    if (err != ESP_OK) {return err;}
    return gpio_set_level(bus->cs, 1);
}

esp_err_t IMU_DirectCsInit(imu_bus_t *bus) {
    if (bus == NULL || !bus->direct_cs) {return ESP_ERR_INVALID_ARG;}
    return direct_cs_init(bus);
}

esp_err_t IMU_BusInit(imu_bus_t *bus) {
    if (bus == NULL) {return ESP_ERR_INVALID_ARG;}

    esp_err_t err = ESP_OK;
    if (bus->direct_cs) {
        err = direct_cs_init(bus);
    } else {
                                                                                   
        err = gpio_set_level(bus->cs, 1);
        if (err == ESP_OK) {err = gpio_set_direction(bus->cs, GPIO_MODE_OUTPUT);}
        if (err == ESP_OK) {err = IMU_SelectInit(bus->select_pins);}
    }

    if (err != ESP_OK) {return err;}
                                                        
    spi_bus_config_t bus_config = {.mosi_io_num = bus->mosi,.miso_io_num = bus->miso,.sclk_io_num = bus->clk,.quadwp_io_num = -1,.quadhd_io_num = -1,.max_transfer_sz = 32};
                                                                                                                               
    err = spi_bus_initialize(bus->host, &bus_config, SPI_DMA_DISABLED);
    if (err != ESP_OK) {return err;}

    return IMU_DeviceAttach(bus);}

esp_err_t IMU_DeviceAttach(imu_bus_t *bus) {
    if (bus == NULL) {return ESP_ERR_INVALID_ARG;}

    spi_device_interface_config_t device_config = {
        .clock_speed_hz = IMU_SPI_HZ,
        .mode = 0,                                                    
        .spics_io_num = -1,
        .queue_size = 1,                                                   
        .cs_ena_posttrans = 1};

    return spi_bus_add_device(bus->host, &device_config, &bus->device);}

esp_err_t IMU_Configure(imu_bus_t *bus, uint8_t address) {
    if (bus == NULL) {return ESP_ERR_INVALID_ARG;}
                                                                                                           
    esp_err_t err = select_sensor_route(bus, address);
    if (err != ESP_OK) {return err;}
                                              
    uint8_t value = 0;
    err = read_registers(bus, REG_WHO_AM_I, &value, 1);
    if (err != ESP_OK) {return err;}
    if (value != 0x6C) {ESP_LOGW(TAG, "SPI%d address %u WHO_AM_I expected 0x6C, received 0x%02X", (int)bus->host + 1, address, value); return ESP_ERR_NOT_FOUND;}
                                                                           
    err = write_register(bus, REG_CTRL3_C, 0x05);
    if (err != ESP_OK) {return err;}
                                                                   
    const int64_t reset_deadline_us = esp_timer_get_time() + (IMU_RESET_TIMEOUT_MS * 1000LL);
    do {
        err = read_registers(bus, REG_CTRL3_C, &value, 1);
        if (err != ESP_OK) {return err;}
        if ((value & 0x01U) == 0) {break;}
        if (esp_timer_get_time() >= reset_deadline_us) {return ESP_ERR_TIMEOUT;}
        vTaskDelay(1);
    } while (true);
                                                                     
    const uint8_t settings[][2] = {
        {REG_CTRL3_C, 0x44},                                                   
        {REG_CTRL9_XL, 0xE2},                                                 
        {REG_CTRL4_C, 0x04},                                                  
        {REG_CTRL1_XL, IMU_CTRL1_XL_VALUE},                      
        {REG_CTRL2_G, IMU_CTRL2_G_VALUE},                             
    };
                                                                                
    const size_t setting_count = sizeof(settings) / sizeof(settings[0]);
                                                           
    for (size_t i = 0; i < setting_count; ++i) {
        err = write_register(bus, settings[i][0], settings[i][1]);
        if (err != ESP_OK) {return err;}
    }
                                                                                                                                       
    for (size_t i = 0; i < setting_count; ++i) {
        err = read_registers(bus, settings[i][0], &value, 1);
        if (err != ESP_OK) {return err;}
        if (value != settings[i][1]) {
            ESP_LOGW(TAG, "SPI%d address %u register 0x%02X expected 0x%02X, received 0x%02X", (int)bus->host + 1, address, settings[i][0], settings[i][1], value);
            if (settings[i][0] == REG_CTRL9_XL) {ESP_LOGW(TAG, "Continuing because this CTRL9_XL mismatch is non-critical and motion data remains usable"); continue;}
            return ESP_ERR_INVALID_RESPONSE;}
    }

    return ESP_OK;
}

                                                                   
static void decode_axes(const uint8_t bytes[6], int16_t raw[3]) {
    for (unsigned axis = 0; axis < 3; ++axis) {
                                                                         
        uint16_t word = (uint16_t)bytes[2 * axis] | ((uint16_t)bytes[2 * axis + 1] << 8);
                                                                             
        int32_t signed_word = (word >= 0x8000U) ? (int32_t)word - 65536 : (int32_t)word;

        raw[axis] = (int16_t)signed_word;
    }
}

esp_err_t IMU_ReadSample(imu_bus_t *bus, uint8_t address, imu_reading_t *reading) {
    if (bus == NULL || reading == NULL) {return ESP_ERR_INVALID_ARG;}
                                                                           
    reading->acc_fresh = false;
    reading->gyro_fresh = false;

    esp_err_t err = select_sensor_route(bus, address);
    if (err != ESP_OK) {return err;}

    uint8_t status = 0;
    err = read_registers(bus, REG_STATUS, &status, 1);
    if (err != ESP_OK) {return err;}

                                                                        
    status &= STATUS_MOTION_READY;
    if (status == 0) {return ESP_OK;}
                                                                                                                             
    uint8_t data[12];
    const bool both_ready = status == STATUS_MOTION_READY;
    const uint8_t first_register = (status & STATUS_GYRO_READY) ? REG_GYRO : REG_ACC;
    const size_t read_size = both_ready ? 12 : 6;

    err = read_registers(bus, first_register, data, read_size);
    if (err != ESP_OK) {return err;}

    reading->timestamp_us = esp_timer_get_time();

    if (status & STATUS_GYRO_READY) {decode_axes(data, reading->gyro_raw); reading->gyro_fresh = true;}
    if (status & STATUS_ACCEL_READY) {const uint8_t *acc_bytes = data + (both_ready ? 6 : 0); decode_axes(acc_bytes, reading->acc_raw); reading->acc_fresh = true;}
    return ESP_OK;
}

esp_err_t IMU_ReadDiagnostics(imu_bus_t *bus, uint8_t address, imu_diagnostic_t *diagnostic) {
    if (bus == NULL || diagnostic == NULL) {return ESP_ERR_INVALID_ARG;}
    memset(diagnostic, 0, sizeof(*diagnostic));
    esp_err_t err = select_sensor_route(bus, address);
    if (err == ESP_OK) {err = read_registers(bus, REG_WHO_AM_I, &diagnostic->who_am_i, 1);}
    if (err == ESP_OK) {err = read_registers(bus, REG_CTRL1_XL, &diagnostic->ctrl1_xl, 1);}
    if (err == ESP_OK) {err = read_registers(bus, REG_CTRL2_G, &diagnostic->ctrl2_g, 1);}
    if (err == ESP_OK) {err = read_registers(bus, REG_STATUS, &diagnostic->status, 1);}
    return err;
}


