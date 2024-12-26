#ifndef _SPEC_RES_H_
#define _SPEC_RES_H_

#include <stdint.h>
#include <stdbool.h>

#include "inc/hw_memmap.h"
#include "inc/hw_types.h"
#include "inc/hw_uart.h"

#include "driverlib/sysctl.h"
#include "driverlib/gpio.h"
#include "driverlib/pin_map.h"
#include "driverlib/uart.h"

#define START_BYTE 0xAA
#define STOP_BYTE  0x55
#if defined(ewarm)
#pragma data_alignment=1024
uint8_t pui8ControlTableUART[1024];
#elif defined(ccs)
#pragma DATA_ALIGN(pui8ControlTable, 1024)
uint8_t pui8ControlTableUART[1024];
#else
uint8_t pui8ControlTableUART[1024] __attribute__ ((aligned(1024)));
#endif

#define UART_MONITOR_BASE GPIO_PORTC_BASE
#define UART_MONITOR_GPIO GPIO_PIN_4

void UART5_DbgSet();
void UART5_DbgRst();
void UART5_DbgInit();
void UART5IntHandler(void);
void UART5_Init(uint32_t baud_rate);
void UART5_SendByte(uint8_t byte);
void UART5_SendDataPacket(float *array, uint16_t count);

#endif
