#ifndef ADC_DMA_H
#define ADC_DMA_H

#include <stdint.h>
#include <stdbool.h>
#include "inc/hw_memmap.h"
#include "inc/hw_adc.h"
#include "driverlib/sysctl.h"
#include "driverlib/gpio.h"
#include "driverlib/adc.h"
#include "driverlib/udma.h"
#include "driverlib/interrupt.h"
#include "inc/hw_ints.h"

#include "FreeRTOS.h"
#include "task.h"
#include "event_groups.h"

EventGroupHandle_t BurstEventGroup;
#define BURST_FIFO_FULL (1 << 0)

#define ADC_BUFFER_SIZE 8
static uint32_t adcBuffer[ADC_BUFFER_SIZE];

bool BurstModeConfig();

#endif
