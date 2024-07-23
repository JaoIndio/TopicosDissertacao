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
#include "DRV8825/drv8825.h"

//#include "myLib.h"
//#include "external_devices/AS7341_photo.h"

/**
 * hello.c
 */
/* Set up the hardware ready to run this demo. */
static void prvSetupHardware( void );

/* This function sets up UART0 to be used for a console to display information
 * as the example is running. */
static void prvConfigureUART(void);


int main(void){
  prvSetupHardware();


  NemaConfig();
  NemaInterruptionConfig();
  UARTprintf("Hello World!\n");
  vTaskStartScheduler();
  while(1){ }

	return 0;
}


static void prvConfigureUART(void)
{
    /* Enable GPIO port A which is used for UART0 pins.
     * TODO: change this to whichever GPIO port you are using. */
    MAP_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOA);

    /* Configure the pin muxing for UART0 functions on port A0 and A1.
     * This step is not necessary if your part does not support pin muxing.
     * TODO: change this to select the port/pin you are using. */


    /* Enable UART0 so that we can configure the clock. */
    MAP_SysCtlPeripheralEnable(SYSCTL_PERIPH_UART0);
    MAP_GPIOPinConfigure(GPIO_PA0_U0RX);
    MAP_GPIOPinConfigure(GPIO_PA1_U0TX);
    /* Use the internal 16MHz oscillator as the UART clock source. */

    /* Select the alternate (UART) function for these pins.
     * TODO: change this to select the port/pin you are using. */
    MAP_GPIOPinTypeUART(GPIO_PORTA_BASE, GPIO_PIN_0 | GPIO_PIN_1);

    UARTClockSourceSet(UART0_BASE, UART_CLOCK_PIOSC);



    /* Initialize the UART for console I/O. */
    UARTStdioConfig(0, 115200, 16000000);
}
/*-----------------------------------------------------------*/

static void prvSetupHardware( void )
{
    /* Run from the PLL at 80 MHz.  Any updates to the PLL rate here would
     * need to be reflected in FreeRTOSConfig.h by updating the value of
     * configCPU_CLOCK_HZ with the new system clock frequency. */
    MAP_SysCtlClockSet(SYSCTL_SYSDIV_2_5 | SYSCTL_USE_PLL | SYSCTL_OSC_MAIN |
                       SYSCTL_XTAL_16MHZ);

    /* Configure device pins. */
    PinoutSet(false); //HAbilita varios perifericos, incluindo todos as GPIOS

    /* Configure UART0 to send messages to terminal. */
    prvConfigureUART();
/*
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

*/
    uint32_t var = 0xAAAA;
    //AS7341_init();
    //AS7341_send(&var);
    // GPIOIntClear()
    // GPIOIntDisable()
    // GPIOIntTypeSet() ->RISING_EDGE
}

void vApplicationTickHook( void )
{
    /* This function will be called by each tick interrupt if
        configUSE_TICK_HOOK is set to 1 in FreeRTOSConfig.h.  User code can be
        added here, but the tick hook is called from an interrupt context, so
        code must not attempt to block, and only the interrupt safe FreeRTOS API
        functions can be used (those that end in FromISR()). */

}

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
