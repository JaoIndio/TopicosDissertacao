#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

/* Kernel includes. */
#include "FreeRTOS.h"
#include "task.h"
#include "semphr.h"

/* Hardware includes. */
#include "inc/hw_ints.h"
#include "inc/hw_memmap.h"
#include "driverlib/adc.h"
#include "driverlib/gpio.h"
#include "driverlib/interrupt.h"
#include "driverlib/sysctl.h"
#include "driverlib/pwm.h"
#include "driverlib/pin_map.h"
#include "utils/uartstdio.h"
#include "utils/ustdlib.h"
#include "inc/hw_types.h"
#include "driverlib/timer.h"
#include "MonoLight/mono_light.h"

bool GreenLightConfig(){

  GPIOUnlockPin(GREEN_LIGHT_PORT, GREEN_LIGHT_PIN);
  GPIOPinTypeGPIOOutput(GREEN_LIGHT_PORT, GREEN_LIGHT_PIN);
  GPIOPadConfigSet(GREEN_LIGHT_PORT, GREEN_LIGHT_PIN, GPIO_STRENGTH_2MA, GPIO_PIN_TYPE_STD);
  GPIOPinWrite(GREEN_LIGHT_PORT, GREEN_LIGHT_PIN, 0);
  return true;
}

bool GreenLightTurnOn(){
  GPIOPinWrite(GREEN_LIGHT_PORT, GREEN_LIGHT_PIN, GREEN_LIGHT_PIN);
  return true;
}

bool GreenLightTurnOff(){
  GPIOPinWrite(GREEN_LIGHT_PORT, GREEN_LIGHT_PIN, 0);
  return true;
}

