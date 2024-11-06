#include <stdint.h>
#include <stdbool.h>
#include "inc/hw_memmap.h"
#include "inc/hw_types.h"
#include "driverlib/sysctl.h"
#include "driverlib/gpio.h"
#include "driverlib/pin_map.h"
#include "driverlib/uart.h"

#define START_BYTE 0xAA
#define STOP_BYTE  0x55

void UART5_Init(uint32_t baud_rate);
void UART5_SendByte(uint8_t byte);
void UART5_SendDataPacket(float *array, uint16_t count);
