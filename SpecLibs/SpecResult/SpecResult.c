#include "SpecResult/SpecResult.h"
#include "FreeRTOS.h"
#include "task.h"
#include "semphr.h"

#include "ADC_DMA_BurstMode/ADC_DMA.h"

#define DMA_CHANNEL UDMA_CH7_UART5TX

uint32_t uart_count=0;
void UART5_DbgRst(){
  GPIOPinWrite(UART_MONITOR_BASE, UART_MONITOR_GPIO, 0);
}
void UART5_DbgSet(){
  GPIOPinWrite(UART_MONITOR_BASE, UART_MONITOR_GPIO, UART_MONITOR_GPIO);
}
void UART5_DbgInit(){
  //Performance Pin Configuration
/*******************************************/
  SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOC);
  GPIOUnlockPin(UART_MONITOR_BASE, UART_MONITOR_GPIO);
  GPIOPinTypeGPIOOutput(UART_MONITOR_BASE, UART_MONITOR_GPIO);
  GPIOPadConfigSet(UART_MONITOR_BASE,  UART_MONITOR_GPIO, \
                   GPIO_STRENGTH_2MA, GPIO_PIN_TYPE_STD);
/*******************************************/
}

void UART5IntHandler(void) {
  // Get and clear the interrupt status
  uart_count++;
  uint32_t ui32Status = UARTIntStatus(UART5_BASE, true);
  UARTIntClear(UART5_BASE, ui32Status);
  
  // Check if the interrupt is for DMA TX completion
  //if (ui32Status & UART_INT_DMATX){
    UART5_DbgRst();
    uint32_t ui32Mode = uDMAChannelModeGet(UDMA_CH14_ADC0_0);
/*
    if(ui32Mode == UDMA_MODE_STOP){
      uDMAChannelTransferSet(UDMA_CH14_ADC0_0 | UDMA_PRI_SELECT, \
                           UDMA_MODE_BASIC,\
                           (void *)(ADC0_BASE + ADC_O_SSFIFO0), adcBuffer, ADC_BUFFER_SIZE);
      errorStatus = uDMAErrorStatusGet();
      uDMAChannelEnable(UDMA_CH14_ADC0_0);
*/
      //ADCTriggerDbgRst();
    // DMA transfer complete
    // Perform post-transfer actions, such as preparing the next buffer
  //}
  taskYIELD();
}

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
  UARTFIFOEnable(UART5_BASE);

  // Step 4: Enable UART DMA features  UDMA_CH7_UART5TX
  SysCtlPeripheralEnable(SYSCTL_PERIPH_UDMA);
  UARTDMAEnable(UART5_BASE, UART_DMA_TX);

  uDMAEnable();
  uDMAControlBaseSet(pui8ControlTableUART);
  uDMAChannelAssign(DMA_CHANNEL);
  uDMAChannelAttributeDisable(DMA_CHANNEL, 
                              UDMA_ATTR_ALTSELECT | UDMA_ATTR_HIGH_PRIORITY|
                              UDMA_ATTR_REQMASK);
  uDMAChannelControlSet(DMA_CHANNEL | UDMA_PRI_SELECT, \
                        UDMA_SIZE_8 | UDMA_SRC_INC_8 | UDMA_DST_INC_NONE | UDMA_ARB_2);

/*
  uDMAChannelTransferSet(UDMA_CH7_UART5TX | UDMA_PRI_SELECT, \
                           UDMA_MODE_BASIC,\
                           (void *)(ADC0_BASE + ADC_O_SSFIFO3), \
                           (void *)(UART5_BASE + UART_O_DR),\
                           1);
    uDMAChannelEnable(UDMA_CH7_UART5TX);
*/

  uDMAChannelAttributeEnable(DMA_CHANNEL, UDMA_ATTR_HIGH_PRIORITY);
  uDMAChannelAttributeEnable(DMA_CHANNEL, UDMA_ATTR_USEBURST);

  //UART DMA Interruption Enable
  // Enable UART DMA TX interrupt UART_INT_DMATX
  UARTIntEnable(UART5_BASE, UART_INT_DMATX|UART_INT_TX);
  // Enable NVIC interrupt for UART0
  IntEnable(INT_UART5);

  // Set the UART interrupt priority (optional)
  IntPrioritySet(INT_UART5, 0x05);
  IntRegister(INT_UART5, UART5IntHandler);
  
  UART5_DbgInit();
} 

// Function to send a single byte over UART5
void UART5_SendByte(uint8_t byte) {
  //UARTCharPut(UART5_BASE, byte);  // Waits until there is space in the FIFO and sends byte
  UARTCharPutNonBlocking(UART5_BASE, byte);  // Envia o byte sem executar espera ocupada
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
      //UARTprintf("\t\t\t\tfloatPtr  %d\n", floatPtr[j]);
    }

    //vTaskDelay(pdMS_TO_TICKS(1));
  }

  UART5_SendByte(STOP_BYTE);  // Send the Stop Byte
  UART5_SendByte(STOP_BYTE);  // Send the Stop Byte
  UART5_SendByte(STOP_BYTE);  // Send the Stop Byte
  UART5_SendByte(STOP_BYTE);  // Send the Stop Byte
  UART5_SendByte(STOP_BYTE);  // Send the Stop Byte
}
