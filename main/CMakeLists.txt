
idf_component_register(
    SRCS
        "main.c"
        "IMU_select.c"
        "IMU_comm.c"
        "transmit_to_PC.c"
    INCLUDE_DIRS "."
    PRIV_REQUIRES
        esp_driver_gpio
        esp_driver_spi
        esp_timer
        esp_rom
        esp_driver_uart
)

