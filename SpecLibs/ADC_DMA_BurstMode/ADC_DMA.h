#ifndef ADC_DMA_H
#define ADC_DMA_H

#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>
#include "inc/hw_memmap.h"
#include "inc/hw_adc.h"
#include "inc/hw_udma.h"
#include "driverlib/sysctl.h"
#include "driverlib/gpio.h"
#include "driverlib/adc.h"
#include "driverlib/udma.h"
#include "driverlib/interrupt.h"
#include "inc/hw_ints.h"

#include "FreeRTOS.h"
#include "task.h"
#include "event_groups.h"

#if defined(ewarm)
#pragma data_alignment=1024
uint8_t pui8ControlTable[1024];
#elif defined(ccs)
#pragma DATA_ALIGN(pui8ControlTable, 1024)
uint8_t pui8ControlTable[1024];
#else
uint8_t pui8ControlTable[1024] __attribute__ ((aligned(1024)));
#endif

#define ADC_MONITOR_BASE GPIO_PORTD_BASE
#define ADC_MONITOR_GPIO GPIO_PIN_3

EventGroupHandle_t BurstEventGroup;
#define BURST_FIFO_FULL (1 << 0)
#define ADC_BUFFER_SIZE 8

volatile uint16_t adcBuffer[ADC_BUFFER_SIZE];

bool BurstModeConfig();
void BurstDMA_Check();
void ADCTriggerDbgSet();
void ADCTriggerDbgRst();

#endif
