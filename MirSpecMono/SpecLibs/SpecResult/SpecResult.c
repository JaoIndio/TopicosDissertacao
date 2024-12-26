#include "SpecResult/SpecResult.h"
#include "FreeRTOS.h"
#include "task.h"
#include "semphr.h"

void UART5_Init(uint32_t baud_rate) {
  // Enable the peripherals for UART5 and GPIOE
  SysCtlPeripheralEnable(SYSCTL_PERIPH_UART5);
  SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOE);

  // Configure GPIO Pins for UART mode (PE4 as RX, PE5 as TX)
  GPIOPinConfigure(GPIO_PE4_U5RX);
  GPIOPinConfigure(GPIO_PE5_U5TX);
  GPIOPinTypeUART(GPIO_PORTE_BASE, GPIO_PIN_4 | GPIO_PIN_5);

  // Configure UART5 with the desired baud rate, 8-N-1 format
  UARTConfigSetExpClk(UART5_BASE, SysCtlClockGet(), baud_rate, \
                      (UART_CONFIG_WLEN_8 | UART_CONFIG_PAR_NONE |\
                       UART_CONFIG_PAR_NONE));
}

// Function to send a single byte over UART5
void UART5_SendByte(uint8_t byte) {
  UARTCharPut(UART5_BASE, byte);  // Waits until there is space in the FIFO and sends byte
}

// Function to send an array of floats with the start, count, and stop bytes
void UART5_SendDataPacket(float *array, uint16_t count){
  UART5_SendByte(START_BYTE);  // Send the Start Byte
  UART5_SendByte(START_BYTE);  // Send the Start Byte
  UART5_SendByte(START_BYTE);  // Send the Start Byte
  UART5_SendByte(START_BYTE);  // Send the Start Byte
  UART5_SendByte(START_BYTE);  // Send the Start Byte

  // Loop through each element in the array
  uint16_t i, j, k;
  uint8_t *floatPtr;
  for(i = 0; i < count; i++){
    // Send the count byte
    UART5_SendByte((uint8_t)((i>>8) & 0xFF)); //High count
    UART5_SendByte((uint8_t)(( i & 0xFF))); //Low count

    // Send the float data (4 bytes each)
    floatPtr = (uint8_t *)&array[i];
    for( j = 0; j < 4; j++){
      UART5_SendByte(floatPtr[j]);
    }

    //vTaskDelay(pdMS_TO_TICKS(1));
  }

  UART5_SendByte(STOP_BYTE);  // Send the Stop Byte
  UART5_SendByte(STOP_BYTE);  // Send the Stop Byte
  UART5_SendByte(STOP_BYTE);  // Send the Stop Byte
  UART5_SendByte(STOP_BYTE);  // Send the Stop Byte
  UART5_SendByte(STOP_BYTE);  // Send the Stop Byte
}
