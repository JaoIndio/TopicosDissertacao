
#include "AS7341_photo.h"

void AS7341_init(void){
  //Inicializa I2C master
  //I2CMasterInitExpClk();
}
void AS7341_send(uint32_t *data){
/*
  -Data is transferred by first setting the slave address using I2CMasterSlaveAddrSet(). That function
    is also used to define whether the transfer is a send (a write to the slave from the master) or a
    receive (a read from the slave by the master).
  
  - Then, if connected to an I2C bus that has multiple masters, the Tiva I2C master must first call 
    I2CMasterBusBusy() before attempting to initiate the desired transaction. 

  - After determining that the bus is not busy, if trying to send data, the user must
    call the I2CMasterDataPut() function

  - The transaction can then be initiated on the bus by calling
    the I2CMasterControl() function with any of the following commands:
      * I2C_MASTER_CMD_SINGLE_SEND
      * I2C_MASTER_CMD_SINGLE_RECEIVE
      * I2C_MASTER_CMD_BURST_SEND_START
      * I2C_MASTER_CMD_BURST_RECEIVE_START
  
  - The remainder of the transaction can then be driven using either a polling or interrupt-driven method.
  
  - For the single send and receive cases, the polling method involves looping on the return from
    I2CMasterBusy(). Once that function indicates that the I2C master is no longer busy, the bus transaction 
    has been completed and can be checked for errors using I2CMasterErr(). If there are no
    errors, then the data has been sent or is ready to be read using I2CMasterDataGet(). For the burst
    send and receive cases, the polling method also involves calling the I2CMasterControl() function for
    each byte transmitted or received
    
    + For the interrupt-driven transaction, the user must register an interrupt handler for the I2C devices and 
      enable the I2C master interrupt; the interrupt occurs when the master is no longer busy.
     
      
      
*/
}
void AS7341_read(uint32_t *data){
  
}
