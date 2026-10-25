#include "transmit_to_PC.h"
#include "IMU_acquisition.h"
#include <string.h>
#include "driver/uart.h"
#include "esp_log.h"
#include "esp_rom_serial_output.h"
#include "freertos/task.h"

#define SERIAL_PACKET_SIZE 372
#define SERIAL_HEADER_SIZE 28
#define SERIAL_RECORD_SIZE 20
#define SERIAL_RECORD_COUNT IMU_TOTAL_RECORD_COUNT
#define SERIAL_BAUD_RATE 921600
#define SERIAL_TASK_STACK_BYTES 4096
#define SERIAL_TASK_PRIORITY 4

typedef struct {uint8_t bytes[SERIAL_PACKET_SIZE];} serial_packet_t;

static void put_u16(uint8_t *destination, uint16_t value) {destination[0] = (uint8_t)(value >> 8); destination[1] = (uint8_t)value;}
static void put_u32(uint8_t *destination, uint32_t value) {destination[0] = (uint8_t)(value >> 24); destination[1] = (uint8_t)(value >> 16); destination[2] = (uint8_t)(value >> 8); destination[3] = (uint8_t)value;}
static void put_u64(uint8_t *destination, uint64_t value) {for (unsigned i = 0; i < 8; ++i) {destination[i] = (uint8_t)(value >> (56U - 8U * i));}}

static uint32_t crc32(const uint8_t *data, size_t length) {
    uint32_t crc = UINT32_MAX;
    for (size_t i = 0; i < length; ++i) {crc ^= data[i]; for (unsigned bit = 0; bit < 8; ++bit) {crc = (crc >> 1) ^ (0xEDB88320U & (0U - (crc & 1U)));}}
    return ~crc;
}

static void serialize_record(uint8_t *destination, const imu_record_t *record) {
    destination[0] = record->channel_id;
    destination[1] = (record->available ? BIT0 : 0) | (record->reading.acc_fresh ? BIT1 : 0) | (record->reading.gyro_fresh ? BIT2 : 0) | ((record->diagnostic_code & 0x07U) << 3);
    for (unsigned axis = 0; axis < 3; ++axis) {put_u16(&destination[2 + axis * 2], (uint16_t)record->reading.acc_raw[axis]);}
    for (unsigned axis = 0; axis < 3; ++axis) {put_u16(&destination[8 + axis * 2], (uint16_t)record->reading.gyro_raw[axis]);}
    destination[14] = record->diagnostic.who_am_i;
    destination[15] = record->diagnostic.ctrl1_xl;
    destination[16] = record->diagnostic.ctrl2_g;
    destination[17] = record->diagnostic.status;
    put_u16(&destination[18], record->stale_scans);
}

static void build_packet(serial_packet_t *packet, const imu_group_frame_t *group_a, const imu_group_frame_t *group_b, uint32_t sequence) {
    memset(packet, 0, sizeof(*packet));
    memcpy(packet->bytes, "IMU1", 4);
    packet->bytes[4] = 3;
    packet->bytes[5] = SERIAL_RECORD_COUNT;
    put_u16(&packet->bytes[6], SERIAL_PACKET_SIZE);
    put_u32(&packet->bytes[8], sequence);
    put_u64(&packet->bytes[12], (uint64_t)(group_a->timestamp_us > group_b->timestamp_us ? group_a->timestamp_us : group_b->timestamp_us));
    put_u32(&packet->bytes[20], group_a->scan_number);
    put_u32(&packet->bytes[24], group_b->scan_number);

    for (unsigned address = 0; address < IMU_PER_GROUP; ++address) {serialize_record(&packet->bytes[SERIAL_HEADER_SIZE + address * SERIAL_RECORD_SIZE], &group_a->imu[address]);}
    for (unsigned address = 0; address < IMU_PER_GROUP; ++address) {serialize_record(&packet->bytes[SERIAL_HEADER_SIZE + (IMU_PER_GROUP + address) * SERIAL_RECORD_SIZE], &group_b->imu[address]);}
    serialize_record(&packet->bytes[SERIAL_HEADER_SIZE + (IMU_GROUP_COUNT * IMU_PER_GROUP) * SERIAL_RECORD_SIZE], &group_a->imu[IMU_PER_GROUP]);
    put_u32(&packet->bytes[SERIAL_PACKET_SIZE - 4], crc32(packet->bytes, SERIAL_PACKET_SIZE - 4));
}

static void serial_task(void *argument) {
    imu_group_frame_t group_a;
    imu_group_frame_t group_b;
    uint32_t sequence = 0;
    esp_rom_output_tx_wait_idle(UART_NUM_0);
    esp_err_t driver_error = uart_driver_install(UART_NUM_0, 256, 2048, 0, NULL, 0);
    if (driver_error != ESP_OK && driver_error != ESP_ERR_INVALID_STATE) {ESP_LOGE("IMU_SERIAL_TX", "UART driver installation failed: %s", esp_err_to_name(driver_error)); vTaskDelete(NULL);}
    uart_set_baudrate(UART_NUM_0, SERIAL_BAUD_RATE);
    esp_log_level_set("*", ESP_LOG_NONE);
    while (true) {
        if (!IMU_WaitForGroup(0, &group_a, portMAX_DELAY)) {continue;}
        if (!IMU_WaitForGroup(1, &group_b, portMAX_DELAY)) {continue;}
        serial_packet_t packet;
        build_packet(&packet, &group_a, &group_b, sequence++);
        uart_write_bytes(UART_NUM_0, packet.bytes, sizeof(packet.bytes));
    }
}

esp_err_t TransmitToPC_Start(void) {
    return xTaskCreate(serial_task, "IMU_serial_tx", SERIAL_TASK_STACK_BYTES, NULL, SERIAL_TASK_PRIORITY, NULL) == pdPASS ? ESP_OK : ESP_ERR_NO_MEM;
}

