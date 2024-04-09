/*
 * hello
 *
 * Copyright (C) 2022 Texas Instruments Incorporated
 * 
 * 
 *  Redistribution and use in source and binary forms, with or without 
 *  modification, are permitted provided that the following conditions 
 *  are met:
 *
 *    Redistributions of source code must retain the above copyright 
 *    notice, this list of conditions and the following disclaimer.
 *
 *    Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the 
 *    documentation and/or other materials provided with the   
 *    distribution.
 *
 *    Neither the name of Texas Instruments Incorporated nor the names of
 *    its contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT 
 *  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
*/

/******************************************************************************
 *
 * The hello example project illustrates FreeRTOS's vTaskCreate, vTaskDelay and
 * vTaskDelete APIs.  A task is created to print the "Hello World!" message
 * five times on a terminal window at an one second interval.  After five
 * seconds, the task is then deleted.
 *
 * main() creates one task.  It then starts the scheduler.
 *
 * The Hello task creates a simple task that will output a 'Hello world!'
 * message over UART five times before self-terminating.
 
 * This example uses UARTprintf for output of UART messages.  UARTprintf is not
 * a thread-safe API and is only being used for simplicity of the demonstration
 * and in a controlled manner.
 *
 * Open a terminal with 115,200 8-N-1 to see the output for this demo.
 *
 */

/* Standard includes. */
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>

/* Kernel includes. */
#include "FreeRTOS.h"
#include "task.h"
#include "semphr.h"

/* Hardware includes. */
#include "inc/hw_ints.h" //Mapeamento de interrupcoes
#include "inc/hw_memmap.h"
#include "inc/hw_sysctl.h"
#include "driverlib/gpio.h"
#include "driverlib/interrupt.h"
#include "driverlib/pin_map.h"
#include "driverlib/rom.h"
#include "driverlib/rom_map.h"
#include "driverlib/sysctl.h"
#include "driverlib/uart.h"
#include "drivers/rtos_hw_drivers.h"
#include "utils/uartstdio.h"
#include "myLib.h"
/*-----------------------------------------------------------*/
SemaphoreHandle_t ToogleLED1;
SemaphoreHandle_t ToogleLED2;
bool Led1Value = false;
bool Led2Value = false;

// Meus Handlers
void GPIOaIntHandler(void){
  GPIOIntClear(GPIO_PORTA_BASE , GPIO_INT_PIN_5);
  BaseType_t xHigherPriorityTaskWoken = pdFALSE;
  xSemaphoreGiveFromISR(ToogleLED1, &xHigherPriorityTaskWoken);
}
void GPIObIntHandler(void){
  GPIOIntClear(GPIO_PORTB_BASE , GPIO_INT_PIN_4);
  BaseType_t xHigherPriorityTaskWoken = pdFALSE;
  xSemaphoreGiveFromISR(ToogleLED2, &xHigherPriorityTaskWoken);
}

/* Set up the hardware ready to run this demo. */
static void prvSetupHardware( void );

/* This function sets up UART0 to be used for a console to display information
 * as the example is running. */
static void prvConfigureUART(void);

/* API to trigger the 'Hello world' task. */
extern void vHelloTask( void );

/*-----------------------------------------------------------*/
void TriggerLed1(void *pvParameters);
void TriggerLed2(void *pvParameters);

int main( void ){
  /* Prepare the hardware to run this demo. */
  ToogleLED1 = xSemaphoreCreateMutex();
  ToogleLED2 = xSemaphoreCreateMutex();

  prvSetupHardware();

  /* Create the Hello task to output a message over UART. */
  vHelloTask();

  //Cria minhas tasks
  int a = 0;
  int b = 0;
  if(xTaskCreate( TriggerLed1,
               "Led1",
               configMINIMAL_STACK_SIZE,
               NULL,
               tskIDLE_PRIORITY + 2,
               NULL )!=pdTRUE)
    a = 0xFF;

  if(xTaskCreate( TriggerLed2,
               "Led2",
               configMINIMAL_STACK_SIZE,
               NULL,
               tskIDLE_PRIORITY + 3,
               NULL )!=pdTRUE)
    b = 0xFF;
  
  VoidFunc();
  /* Start the tasks and timer running. */
  vTaskStartScheduler();

  /* If all is well, the scheduler will now be running, and the following
  line will never be reached.  If the following line does execute, then
  there was insufficient FreeRTOS heap memory available for the idle and/or
  timer tasks to be created.  See the memory management section on the
  FreeRTOS web site for more details. */
  for( ;; );
}
/*-----------------------------------------------------------*/

static void prvConfigureUART(void)
{
    /* Enable GPIO port A which is used for UART0 pins.
     * TODO: change this to whichever GPIO port you are using. */
    SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOA);

    /* Configure the pin muxing for UART0 functions on port A0 and A1.
     * This step is not necessary if your part does not support pin muxing.
     * TODO: change this to select the port/pin you are using. */
    GPIOPinConfigure(GPIO_PA0_U0RX);
    GPIOPinConfigure(GPIO_PA1_U0TX);

    /* Enable UART0 so that we can configure the clock. */
    SysCtlPeripheralEnable(SYSCTL_PERIPH_UART0);

    /* Use the internal 16MHz oscillator as the UART clock source. */
    UARTClockSourceSet(UART0_BASE, UART_CLOCK_PIOSC);

    /* Select the alternate (UART) function for these pins.
     * TODO: change this to select the port/pin you are using. */
    GPIOPinTypeUART(GPIO_PORTA_BASE, GPIO_PIN_0 | GPIO_PIN_1);

    /* Initialize the UART for console I/O. */
    UARTStdioConfig(0, 115200, 16000000);
}
/*-----------------------------------------------------------*/

static void prvSetupHardware( void )
{
    /* Run from the PLL at 80 MHz.  Any updates to the PLL rate here would
     * need to be reflected in FreeRTOSConfig.h by updating the value of
     * configCPU_CLOCK_HZ with the new system clock frequency. */
    MAP_SysCtlClockSet(SYSCTL_SYSDIV_2_5 | SYSCTL_USE_PLL | SYSCTL_OSC_INT |
                       SYSCTL_XTAL_16MHZ);

    /* Configure device pins. */
    PinoutSet(false); //HAbilita varios perifericos, incluindo todos as GPIOS

    /* Configure UART0 to send messages to terminal. */
    prvConfigureUART();

    // Habilitar GPIOS q ligarao dois LEDs
    GPIODirModeSet(GPIO_PORTE_BASE, (GPIO_PIN_4 | GPIO_PIN_5), GPIO_DIR_MODE_OUT);
    // Habilitar GPIOS q serao botos Pull-Ups
    GPIODirModeSet(GPIO_PORTB_BASE, GPIO_PIN_4, GPIO_DIR_MODE_IN);
    GPIODirModeSet(GPIO_PORTA_BASE, GPIO_PIN_5, GPIO_DIR_MODE_IN);
    
    //Configura as 4 portas
    GPIOPadConfigSet(GPIO_PORTE_BASE, (GPIO_PIN_4 | GPIO_PIN_5), \
                     GPIO_STRENGTH_2MA, GPIO_PIN_TYPE_STD_WPU);

    GPIOPadConfigSet(GPIO_PORTB_BASE, GPIO_PIN_4, \
                     GPIO_STRENGTH_2MA, GPIO_PIN_TYPE_STD);
    GPIOPadConfigSet(GPIO_PORTA_BASE, GPIO_PIN_5, \
                     GPIO_STRENGTH_2MA, GPIO_PIN_TYPE_STD);
    
    //Habilita e Configura Interrupção
    GPIOIntEnable(GPIO_PORTB_BASE, GPIO_PIN_4);
    GPIOIntTypeSet(GPIO_PORTB_BASE, GPIO_PIN_4,\
                   GPIO_RISING_EDGE);
    IntEnable(INT_GPIOB);

    GPIOIntEnable(GPIO_PORTA_BASE, GPIO_PIN_5);
    GPIOIntTypeSet(GPIO_PORTA_BASE, GPIO_PIN_5,\
                   GPIO_RISING_EDGE);
    IntEnable(INT_GPIOA);
    
    // GPIOIntClear()
    // GPIOIntDisable()
    // GPIOIntTypeSet() ->RISING_EDGE
}
/*-----------------------------------------------------------*/

void vApplicationMallocFailedHook( void )
{
    /* vApplicationMallocFailedHook() will only be called if
    configUSE_MALLOC_FAILED_HOOK is set to 1 in FreeRTOSConfig.h.  It is a hook
    function that will get called if a call to pvPortMalloc() fails.
    pvPortMalloc() is called internally by the kernel whenever a task, queue,
    timer or semaphore is created.  It is also called by various parts of the
    demo application.  If heap_1.c or heap_2.c are used, then the size of the
    heap available to pvPortMalloc() is defined by configTOTAL_HEAP_SIZE in
    FreeRTOSConfig.h, and the xPortGetFreeHeapSize() API function can be used
    to query the size of free heap space that remains (although it does not
    provide information on how the remaining heap might be fragmented). */
    IntMasterDisable();
    for( ;; );
}
/*-----------------------------------------------------------*/

void vApplicationIdleHook( void )
{
    /* vApplicationIdleHook() will only be called if configUSE_IDLE_HOOK is set
    to 1 in FreeRTOSConfig.h.  It will be called on each iteration of the idle
    task.  It is essential that code added to this hook function never attempts
    to block in any way (for example, call xQueueReceive() with a block time
    specified, or call vTaskDelay()).  If the application makes use of the
    vTaskDelete() API function (as this demo application does) then it is also
    important that vApplicationIdleHook() is permitted to return to its calling
    function, because it is the responsibility of the idle task to clean up
    memory allocated by the kernel to any task that has since been deleted. */
}
/*-----------------------------------------------------------*/

void vApplicationStackOverflowHook( TaskHandle_t pxTask, char *pcTaskName )
{
    ( void ) pcTaskName;
    ( void ) pxTask;

    /* Run time stack overflow checking is performed if
    configCHECK_FOR_STACK_OVERFLOW is defined to 1 or 2.  This hook
    function is called if a stack overflow is detected. */
    IntMasterDisable();
    for( ;; );
}

void TriggerLed1(void *pvParameters){
  while(1){
    if(xSemaphoreTake(ToogleLED1, portMAX_DELAY)){
      UARTprintf("\033[1;35m");
      UARTprintf("\r\n\n[LED1] \n");
      if(Led1Value)
        GPIOPinWrite(GPIO_PORTE_BASE, GPIO_PIN_4, GPIO_PIN_4);
      else
        GPIOPinWrite(GPIO_PORTE_BASE, GPIO_PIN_4, 0);
      Led1Value=!Led1Value;
      UARTprintf("\033[0m");
    }
  }
}
void TriggerLed2(void *pvParameters){
  while(1){
    if(xSemaphoreTake(ToogleLED2, portMAX_DELAY)){
      UARTprintf("\033[1;33m");
      UARTprintf("\r\n\n[LED2] \n");
      if(Led2Value)
        GPIOPinWrite(GPIO_PORTE_BASE, GPIO_PIN_5, GPIO_PIN_5);
      else
        GPIOPinWrite(GPIO_PORTE_BASE, GPIO_PIN_5, 0);
      Led2Value=!Led2Value;
      UARTprintf("\033[0m");
    }
  }
}

//void *malloc( size_t xSize )
//{
//    /* There should not be a heap defined, so trap any attempts to call
//    malloc. */
//    IntMasterDisable();
//    for( ;; );
//}
/*-----------------------------------------------------------*/

