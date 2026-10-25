#include "IMU_acquisition.h"
#include "IMU_config.h"
#include "pins.h"
#include "transmit_to_PC.h"
#include <string.h>
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/event_groups.h"
#include "freertos/queue.h"
#include "freertos/task.h"
                                                                    
#if CONFIG_FREERTOS_UNICORE
#error "This firmware requires both ESP32-S3 cores."
#endif

#define GROUP_A_READY BIT0
#define GROUP_B_READY BIT1
#define START_SCANNING BIT2
#define IMU_STALE_DIAGNOSTIC_SCANS 5U
#define IMU_RECOVERY_COOLDOWN_SCANS 100U
#define IMU_DIAGNOSTIC_HOLD_SCANS 200U
#define MIDDLE1_FRAME_INDEX IMU_PER_GROUP
#define MIDDLE1_DIRECT_ADDRESS 0U
typedef struct {
    uint8_t index;                                            
    const char *log_tag;                                                         
    imu_bus_t bus;                                                       
    imu_group_frame_t frame;                                              
} imu_group_t;

static imu_group_t groups[IMU_GROUP_COUNT] = {
    {.index = 0,.log_tag = "IMU_A",.bus ={.host = SPI2_HOST,.mosi = SPI2_MOSI,.miso = SPI2_MISO,.clk = SPI2_CLK,.cs = SPI2_CS,.select_pins = {A0_PIN, A1_PIN, A2_PIN}}},
    {.index = 1,.log_tag = "IMU_B",.bus ={.host = SPI3_HOST,.mosi = SPI3_MOSI,.miso = SPI3_MISO,.clk = SPI3_CLK,.cs = SPI3_CS,.select_pins = {B0_PIN, B1_PIN, B2_PIN}}}
};
                                                                               
static imu_bus_t middle1_bus = {.host = SPI2_HOST,.mosi = SPI2_MOSI,.miso = SPI2_MISO,.clk = SPI2_CLK,.cs = THUMB_CMC_CS,.direct_cs = true};
                                                                                               
static const struct {
    uint8_t id;
    char name[IMU_NAME_LENGTH];
} channel_map[IMU_GROUP_COUNT][IMU_PER_GROUP] = {
    [0] ={                                              
            [0] = {IMU_CHANNEL_MIDDLE3, "middle3"},
            [1] = {IMU_CHANNEL_MIDDLE2, "middle2"},
            [2] = {IMU_CHANNEL_RING3, "ring3"},
            [3] = {IMU_CHANNEL_RING2, "ring2"},
            [4] = {IMU_CHANNEL_RING1, "ring1"},
            [5] = {IMU_CHANNEL_PINKY3, "pinky3"},
            [6] = {IMU_CHANNEL_PINKY2, "pinky2"},
            [7] = {IMU_CHANNEL_PINKY1, "pinky1"}},
    [1] ={                                      
            [0] = {IMU_CHANNEL_THUMB3, "Thumb3"},
            [1] = {IMU_CHANNEL_THUMB2, "Thumb2"},
            [2] = {IMU_CHANNEL_INDEX3, "Index3"},
            [3] = {IMU_CHANNEL_INDEX2, "Index2"},
            [4] = {IMU_CHANNEL_INDEX1, "Index1"},
            [5] = {IMU_CHANNEL_THUMB_CMC, "thumb_cmc"},
            [6] = {IMU_CHANNEL_PALM, "Palm"},
            [7] = {IMU_CHANNEL_WRIST, "Wrist"}} };

                                                                       
static QueueHandle_t frame_queues[IMU_GROUP_COUNT];                                                                                   
static EventGroupHandle_t startup_events;
static uint8_t sensor_mux_address(const imu_group_t *group, unsigned address) {return address;}
static esp_err_t configure_sensor(imu_group_t *group, unsigned address) {
    imu_bus_t *bus = &group->bus;
    esp_err_t err = bus->device != NULL ? IMU_Configure(bus, sensor_mux_address(group, address)) : ESP_ERR_INVALID_STATE;
    return err;
}

static esp_err_t read_sensor(imu_group_t *group, unsigned address, imu_reading_t *reading) {
    esp_err_t err = IMU_ReadSample(&group->bus, sensor_mux_address(group, address), reading);
    return err;
}


static esp_err_t read_sensor_diagnostics(imu_group_t *group, unsigned address, imu_diagnostic_t *diagnostic) {
    esp_err_t err = IMU_ReadDiagnostics(&group->bus, sensor_mux_address(group, address), diagnostic);
    return err;
}

static bool has_middle1_sensor(const imu_group_t *group) {return group->index == 0;}

static imu_record_t *middle1_record(imu_group_t *group) {return &group->frame.imu[MIDDLE1_FRAME_INDEX];}

static esp_err_t configure_middle1_sensor(void) {return middle1_bus.device != NULL ? IMU_Configure(&middle1_bus, MIDDLE1_DIRECT_ADDRESS) : ESP_ERR_INVALID_STATE;}

static esp_err_t read_middle1_sensor(imu_reading_t *reading) {return IMU_ReadSample(&middle1_bus, MIDDLE1_DIRECT_ADDRESS, reading);}

static esp_err_t read_middle1_diagnostics(imu_diagnostic_t *diagnostic) {return IMU_ReadDiagnostics(&middle1_bus, MIDDLE1_DIRECT_ADDRESS, diagnostic);}

static uint8_t classify_diagnostic(esp_err_t error, const imu_diagnostic_t *diagnostic) {
    if (error != ESP_OK) {return 4;}
    if (diagnostic->who_am_i != 0x6C) {return 2;}
    if (diagnostic->ctrl1_xl != IMU_CTRL1_XL_VALUE || diagnostic->ctrl2_g != IMU_CTRL2_G_VALUE) {return 3;}
    return (diagnostic->status & 0x03U) == 0 ? 1 : 0;
}

static void diagnose_stale_sensor(imu_group_t *group, unsigned address) {
    imu_record_t *record = &group->frame.imu[address];
    esp_err_t diagnostic_error = read_sensor_diagnostics(group, address, &record->diagnostic);
    record->diagnostic_code = classify_diagnostic(diagnostic_error, &record->diagnostic);
    ESP_LOGW(group->log_tag, "%s stale: code=%u WHO=0x%02X CTRL1_XL=0x%02X CTRL2_G=0x%02X STATUS=0x%02X", record->name, record->diagnostic_code, record->diagnostic.who_am_i, record->diagnostic.ctrl1_xl, record->diagnostic.ctrl2_g, record->diagnostic.status);
    record->recovery_cooldown_scans = IMU_RECOVERY_COOLDOWN_SCANS;
    record->diagnostic_hold_scans = IMU_DIAGNOSTIC_HOLD_SCANS;
}

static void diagnose_stale_middle1_sensor(imu_group_t *group) {
    imu_record_t *record = middle1_record(group);
    esp_err_t diagnostic_error = read_middle1_diagnostics(&record->diagnostic);
    record->diagnostic_code = classify_diagnostic(diagnostic_error, &record->diagnostic);
    ESP_LOGW(group->log_tag, "%s stale: code=%u WHO=0x%02X CTRL1_XL=0x%02X CTRL2_G=0x%02X STATUS=0x%02X", record->name, record->diagnostic_code, record->diagnostic.who_am_i, record->diagnostic.ctrl1_xl, record->diagnostic.ctrl2_g, record->diagnostic.status);
    record->recovery_cooldown_scans = IMU_RECOVERY_COOLDOWN_SCANS;
    record->diagnostic_hold_scans = IMU_DIAGNOSTIC_HOLD_SCANS;
}

bool IMU_WaitForGroup(uint8_t group, imu_group_frame_t *frame, TickType_t wait_ticks) {
    if (group >= IMU_GROUP_COUNT || frame == NULL || frame_queues[group] == NULL) {return false;}
    return xQueueReceive(frame_queues[group], frame, wait_ticks) == pdTRUE;
}

                                                                               
static void mark_unavailable(imu_group_t *group, unsigned address, esp_err_t error) {
                                                                                                    
    imu_record_t *record = &group->frame.imu[address];
    record->available = false;
    record->reading.acc_fresh = false;
    record->reading.gyro_fresh = false;

    ESP_LOGW(group->log_tag, "%s (MUX address %u) unavailable: %s", record->name, record->mux_address, esp_err_to_name(error));
}
  
static void initialise_group(imu_group_t *group) {
                                                                                 
    esp_err_t err = IMU_BusInit(&group->bus);
    if (err != ESP_OK) {
        for (unsigned address = 0; address < IMU_PER_GROUP; ++address) {
            mark_unavailable(group, address, err);}
        return;
    }

    if (has_middle1_sensor(group)) {
        ESP_LOGI(group->log_tag, "STARTUP_MARKER attaching Middle1 direct-CS SPI2 route");
        err = IMU_DirectCsInit(&middle1_bus);
        if (err == ESP_OK) {err = IMU_DeviceAttach(&middle1_bus);}
        if (err != ESP_OK) {ESP_LOGE(group->log_tag, "Could not attach Middle1 direct-CS SPI2 route: %s", esp_err_to_name(err));}
    }
                                                                         
    vTaskDelay(pdMS_TO_TICKS(IMU_BOOT_WAIT_MS) + 1);

    for (unsigned address = 0; address < IMU_PER_GROUP; ++address) {
                                                                                           
                                                                       
        for (unsigned attempt = 1; attempt <= 3; ++attempt) {
            err = configure_sensor(group, address);
            if (err == ESP_OK) {break;}
            ESP_LOGW(group->log_tag, "%s (MUX address %u) configuration attempt %u/3 failed: %s", group->frame.imu[address].name, sensor_mux_address(group, address), attempt, esp_err_to_name(err));
            if (attempt < 3) {vTaskDelay(pdMS_TO_TICKS(10) + 1);}
        }
        if (err != ESP_OK) {mark_unavailable(group, address, err); continue;}

        group->frame.imu[address].available = true;
        ESP_LOGI(group->log_tag, "%s (MUX address %u) configured", group->frame.imu[address].name, sensor_mux_address(group, address));
    }

    if (has_middle1_sensor(group)) {
        imu_record_t *record = middle1_record(group);
        ESP_LOGI(group->log_tag, "STARTUP_MARKER configuring Middle1 direct-CS sensor");
        for (unsigned attempt = 1; attempt <= 3; ++attempt) {
            err = configure_middle1_sensor();
            if (err == ESP_OK) {break;}
            ESP_LOGW(group->log_tag, "%s direct CS GPIO%d configuration attempt %u/3 failed: %s", record->name, THUMB_CMC_CS, attempt, esp_err_to_name(err));
            if (attempt < 3) {vTaskDelay(pdMS_TO_TICKS(10) + 1);}
        }
        if (err == ESP_OK) {record->available = true; ESP_LOGI(group->log_tag, "Middle1 configured on direct CS GPIO%d", THUMB_CMC_CS);}
        else {mark_unavailable(group, MIDDLE1_FRAME_INDEX, err);}
    }

    ESP_LOGI(group->log_tag, "STARTUP_MARKER configured normal sensors; direct Middle1 stage complete if applicable");                                                                       
    ESP_LOGI(group->log_tag, "STARTUP_MARKER gyro startup delay begin");
    vTaskDelay(pdMS_TO_TICKS(IMU_GYRO_STARTUP_MS) + 1);
    ESP_LOGI(group->log_tag, "STARTUP_MARKER gyro startup delay complete; discard scans begin");

    for (unsigned discard_scan = 0; discard_scan < IMU_STARTUP_DISCARD_SCANS; ++discard_scan) {
        ESP_LOGI(group->log_tag, "STARTUP_MARKER discard scan %u begin", discard_scan);
                                                                           
        for (unsigned address = 0; address < IMU_PER_GROUP; ++address) {
            imu_record_t *record = &group->frame.imu[address];
            if (!record->available) {continue;}

            imu_reading_t discarded = {0};
            err = read_sensor(group, address, &discarded);
                                                                                                                                                                       
            if (err != ESP_OK) {ESP_LOGW(group->log_tag, "%s startup discard read failed: %s", record->name, esp_err_to_name(err));}
        }
        if (has_middle1_sensor(group)) {
            imu_record_t *record = middle1_record(group);
            if (record->available) {
                imu_reading_t discarded = {0};
                err = read_middle1_sensor(&discarded);
            if (err != ESP_OK) {ESP_LOGW(group->log_tag, "middle1 startup discard read failed: %s", esp_err_to_name(err));}
            }
        }
        ESP_LOGI(group->log_tag, "STARTUP_MARKER discard scan %u complete", discard_scan);
                                                                         
        vTaskDelay(pdMS_TO_TICKS(IMU_STARTUP_DISCARD_INTERVAL_MS) + 1);
    }
    ESP_LOGI(group->log_tag, "STARTUP_MARKER initialise_group complete");
}

                                                                              
static void scan_group(imu_group_t *group) {
                                                                                         
    for (unsigned address = 0; address < IMU_PER_GROUP; ++address) {
        imu_record_t *record = &group->frame.imu[address];
        if (!record->available) {
            record->reading.acc_fresh = false;
            record->reading.gyro_fresh = false;
                                                                                                                                                      
            if (group->frame.scan_number % 100U == address * 10U) {
                esp_err_t recovery = configure_sensor(group, address);
                if (recovery == ESP_OK) {record->available = true; record->stale_scans = 0; record->diagnostic_code = 0; memset(&record->diagnostic, 0, sizeof(record->diagnostic)); ESP_LOGI(group->log_tag, "%s recovered and is available", record->name);}
            }
            continue;
        }

        record->reading.acc_fresh = false;
        record->reading.gyro_fresh = false;
        esp_err_t err = read_sensor(group, address, &record->reading);
        
        if (err != ESP_OK) {ESP_LOGW(group->log_tag, "%s sample read failed: %s", record->name, esp_err_to_name(err));}
        if (record->recovery_cooldown_scans > 0) {--record->recovery_cooldown_scans;}
        if (err == ESP_OK && record->reading.acc_fresh && record->reading.gyro_fresh) {
            record->stale_scans = 0;
            if (record->diagnostic_hold_scans > 0) {--record->diagnostic_hold_scans;}
            else {record->diagnostic_code = 0; memset(&record->diagnostic, 0, sizeof(record->diagnostic));}
        } else {
            if (record->stale_scans < UINT16_MAX) {++record->stale_scans;}
            if (record->stale_scans >= IMU_STALE_DIAGNOSTIC_SCANS && record->recovery_cooldown_scans == 0) {diagnose_stale_sensor(group, address);}
        }
    }
}

static void scan_middle1_sensor(imu_group_t *group) {
    if (!has_middle1_sensor(group)) {return;}
    imu_record_t *record = middle1_record(group);
    if (!record->available) {
        record->reading.acc_fresh = false;
        record->reading.gyro_fresh = false;
        if (group->frame.scan_number % 100U == 80U) {
            esp_err_t recovery = configure_middle1_sensor();
            if (recovery == ESP_OK) {record->available = true; record->stale_scans = 0; record->diagnostic_code = 0; memset(&record->diagnostic, 0, sizeof(record->diagnostic)); ESP_LOGI(group->log_tag, "%s recovered and is available", record->name);}
        }
        return;
    }
    record->reading.acc_fresh = false;
    record->reading.gyro_fresh = false;
    esp_err_t err = read_middle1_sensor(&record->reading);
    if (err != ESP_OK) {ESP_LOGW(group->log_tag, "%s sample read failed: %s", record->name, esp_err_to_name(err));}
    if (record->recovery_cooldown_scans > 0) {--record->recovery_cooldown_scans;}
    if (err == ESP_OK && record->reading.acc_fresh && record->reading.gyro_fresh) {
        record->stale_scans = 0;
        if (record->diagnostic_hold_scans > 0) {--record->diagnostic_hold_scans;}
        else {record->diagnostic_code = 0; memset(&record->diagnostic, 0, sizeof(record->diagnostic));}
    } else {
        if (record->stale_scans < UINT16_MAX) {++record->stale_scans;}
        if (record->stale_scans >= IMU_STALE_DIAGNOSTIC_SCANS && record->recovery_cooldown_scans == 0) {diagnose_stale_middle1_sensor(group);}
    }
}

                                                                
static void group_task(void *argument) {
                                                                               
    imu_group_t *group = argument;
    group->frame.group = group->index;
 
    for (unsigned address = 0; address < IMU_PER_GROUP; ++address) {
        imu_record_t *record = &group->frame.imu[address];
        record->channel_id = channel_map[group->index][address].id;
        record->spi_bus = group->index + 2;
        record->mux_address = sensor_mux_address(group, address);
        memcpy(record->name, channel_map[group->index][address].name, sizeof(record->name));
    }
    if (has_middle1_sensor(group)) {
        imu_record_t *record = middle1_record(group);
        record->channel_id = IMU_CHANNEL_MIDDLE1;
        record->spi_bus = 2;
        record->mux_address = MIDDLE1_DIRECT_ADDRESS;
        memcpy(record->name, "middle1", sizeof(record->name));
    }

    ESP_LOGI(group->log_tag, "Starting on core %d using SPI%u", xPortGetCoreID(), (unsigned)(group->index + 2));

    initialise_group(group);

    ESP_LOGI(group->log_tag, "STARTUP_MARKER group_task returned from initialise_group");
                                                                       
    EventBits_t ready_bit = (group->index == 0) ? GROUP_A_READY : GROUP_B_READY;
    ESP_LOGI(group->log_tag, "STARTUP_MARKER setting ready bit 0x%lx", (unsigned long)ready_bit);
                                                                                 
    xEventGroupSetBits(startup_events, ready_bit);
    ESP_LOGI(group->log_tag, "STARTUP_MARKER ready bit set; waiting for START_SCANNING");
                                                                      
    xEventGroupWaitBits(startup_events, START_SCANNING, pdFALSE, pdTRUE, portMAX_DELAY);
    ESP_LOGI(group->log_tag, "STARTUP_MARKER START_SCANNING released; entering scan loop");

    const TickType_t scan_period = pdMS_TO_TICKS(IMU_SCAN_PERIOD_MS);
                                                                           
    TickType_t previous_wake_tick = xTaskGetTickCount();

    while (true) {                                                                    
        xTaskDelayUntil(&previous_wake_tick, scan_period);                                                                   
        group->frame.timestamp_us = esp_timer_get_time();
        scan_group(group);
        scan_middle1_sensor(group);
        ++group->frame.scan_number;                                                              
        xQueueOverwrite(frame_queues[group->index], &group->frame);
    }
}

void app_main(void) {
                                                                                
    const TickType_t scan_period = pdMS_TO_TICKS(IMU_SCAN_PERIOD_MS);
    configASSERT(scan_period > 0);                                                                          
    startup_events = xEventGroupCreate();
    ESP_ERROR_CHECK(startup_events != NULL ? ESP_OK : ESP_ERR_NO_MEM);
                                                                 
    for (unsigned group = 0; group < IMU_GROUP_COUNT; ++group) {
                                                              
        frame_queues[group] = xQueueCreate(1, sizeof(imu_group_frame_t));
        ESP_ERROR_CHECK(frame_queues[group] != NULL ? ESP_OK : ESP_ERR_NO_MEM);
    }
                                                            
    for (unsigned i = 0; i < IMU_GROUP_COUNT; ++i) {

        BaseType_t created = xTaskCreatePinnedToCore(group_task, groups[i].log_tag, IMU_TASK_STACK_BYTES, &groups[i], IMU_TASK_PRIORITY, NULL, i);
        ESP_ERROR_CHECK(created == pdPASS ? ESP_OK : ESP_ERR_NO_MEM);
    }
                                                                                    
    ESP_LOGI("MainBoard", "STARTUP_MARKER app_main waiting for both group ready bits");
    xEventGroupWaitBits(startup_events, GROUP_A_READY | GROUP_B_READY, pdFALSE, pdTRUE, portMAX_DELAY);
    ESP_LOGI("MainBoard", "STARTUP_MARKER both groups ready; setting START_SCANNING");                                                         
    xEventGroupSetBits(startup_events, START_SCANNING);
    ESP_LOGI("MainBoard", "Startup complete: both groups scan at 100 Hz; IMUs sample at 208 Hz.");                                                                           
    ESP_ERROR_CHECK(TransmitToPC_Start());

}



























