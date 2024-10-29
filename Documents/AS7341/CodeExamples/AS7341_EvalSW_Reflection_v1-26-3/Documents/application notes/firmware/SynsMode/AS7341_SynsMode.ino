
  #include <Wire.h>

  // I2C device address - 0x39
  #define _i2cAddr (0x39)

        void setup() 
        {
  
        // Initiate the Wire library and join the I2C bus as a master or slave
        Wire.begin();

        // communication with the host computer serial monitor
        Serial.begin(9600);
        
        }

          void loop() 
          {
          // put your main code here, to run repeatedly:
          synsMode();

          //Sets the Atime for integration time from 0 to 255 in register (0x81), integration time = (ATIME + 1) * (ASTEP + 1) * 2.78µS
          setATIME(byte (0x64));

      
          // Sets the Astep for integration time from 0 to 65535 in register (0xCA[7:0]) and (0xCB[15:8]), integration time = (ATIME + 1) * (ASTEP + 1) * 2.78µS
          setASTEP(byte (0xE7), byte (0x03));

          // Sets the Spectral Gain in CFG1 Register (0xAA) in [4:0] bit
          setGAIN(byte (0x09));
          }


            void synsMode()
            {
            bool isEnabled = true;
            bool isDataReady = false;
            
            // Setting the PON bit in Enable register 0x80     
            PON();

                
            // Disable SP_EN bit in Enable register 0x80
            SpEn(false);
          
            // Write SMUX configuration from RAM to set SMUX chain registers (Write 0x10 to CFG6)
            SmuxConfigRAM();

            
            // Write new configuration to all the 20 registers for reading channels from F5-F8, Clear and NIR         
            F5F8_Clear_NIR();
            
            
            // Start SMUX command: Enable the SMUXEN bit (bit 4) in register ENABLE
            SMUXEN();


            // Checking on the enabled SMUXEN bit whether back to zero- Poll the SMUXEN bit -> if it is 0 SMUX command is started           
            while(isEnabled)
            
            {
              isEnabled = getSmuxEnabled();
            }

            //Enabling the gpio_in_en (Bit 2) and gpio_out(Bit 1) in GPIO register 0xBE
            GPIO_MODE();

            //reg_bank bit(4) is set to '1' for setting the 0x00-0x7f regiater to reg_bank register (0xA9)
            RegBankConfig();


            // CONFIG (0x70) is used to set the INT_MODE (Bit 1:0) to SYNS Mode by  writing 0x01
            INT_MODE(0x01); 

            
            // writing back the Reg_bank priorty to RAM bank select to access the RAM registers
            writeRegister(byte (0xA9), byte (0x00));


            // Enabling the SP_EN bit in Enable register 0x80
            SpEn(true);

         
            // Reading and Polling the the AVALID bit in Status 2 Register 0xA3           
            while(!(isDataReady))
             {
            
              isDataReady = getIsDataReady();
             
             }

            // Steps defined to printout 6 channels F5,F6,F7,F8,NIR,Clear                      
            Serial.print("ADC0/F5-");
            Serial.println(readTwoRegister1(0x95));
            Serial.print("ADC1/F6-");
            Serial.println(readTwoRegister1(0x97));
            Serial.print("ADC2/F7-");
            Serial.println(readTwoRegister1(0x99));
            Serial.print("ADC3/F8-");
            Serial.println(readTwoRegister1(0x9B));
            Serial.print("ADC4/Clear-");
            Serial.println(readTwoRegister1(0x9D));
            Serial.print("ADC5/NIR-");
            Serial.println(readTwoRegister1(0x9F));
            Serial.println("");
                    
        }  
     
     
     
     /*----- Register configuration  -----*/

       // <summary>  
       // Setting the PON (Power on) bit on the chip (bit0 at register ENABLE 0x80)
       // Attention: This function clears only the PON bit in ENABLE register and keeps the other bits
       // <summary>
        
        void PON()
          {
   
            byte regVal = readRegister(byte(0x80));
            byte temp = regVal;
            regVal = regVal & 0xFE;
            regVal = regVal | 0x01;
            writeRegister(byte (0x80), byte (regVal));
    
          }


       // <summary>
       // Setting the SP_EN (spectral measurement enabled) bit on the chip (bit 1 in register ENABLE)
       // <summary>
       // <param name="isEnable">Enabling (true) or disabling (false) the SP_EN bit</param>
          
         void SpEn(bool isEnable)
          {
          
             byte regVal = readRegister(byte(0x80));
             byte temp = regVal;
             regVal = regVal & 0xFD;
             
             if(isEnable == true)
               {
                regVal= regVal | 0x02;
               }
             else 
               {
                regVal = temp & 0xFD;
               }
            
             writeRegister(byte (0x80), byte (regVal));
             
         }

        // <summary>  
       // Write SMUX configration from RAM to set SMUX chain in CFG6 register 0xAF
       // <summary>
          
          void SmuxConfigRAM()
          {
       
            
            writeRegister(byte (0xAF), byte (0x10));
          
          }


        // <summary>
        // Starting the SMUX command via enabling the SMUXEN bit (bit 4) in register ENABLE 0x80
        // The SMUXEN bit gets cleared automatically as soon as SMUX operation is finished
        // <summary>
        
         void SMUXEN()
        {

          byte regVal = readRegister(byte(0x80));
          byte temp = regVal;
          regVal = regVal & 0xEF;
          regVal = regVal | 0x10;
          writeRegister(byte (0x80), byte (regVal));
                           
        }


        // <summary>
        // Reading and Polling the the SMUX Enable bit in Enable Register 0x80
        // The SMUXEN bit gets cleared automatically as soon as SMUX operation is finished
        // <summary>
   
         bool getSmuxEnabled()
          {
          
            bool isEnabled = false;
            byte regVal = readRegister(byte(0x80));
      
            if( (regVal & 0x10) == 0x10)
              {
                return isEnabled = true;      
              }
                
            else
              {
                return isEnabled = false;
              }
              
           }


          // <summary>
          // Reading and Polling the the AVALID bit in Status 2 Register 0xA3,if the spectral measurement is ready or busy.
          // True indicates that a cycle is completed since the last readout of the Raw Data register
          // <summary>
      
         bool getIsDataReady()
            {
              bool isDataReady = false;
              byte regVal = readRegister(byte(0xA3));
    
              if( (regVal & 0x40) == 0x40){
    
              return isDataReady = true;
              }
              
              else
              {
              return isDataReady = false;
              }
            }

            

         // <summary>
         //Enabling the gpio_in_en (Bit 2) and gpio_out(Bit 1) in GPIO register 0xBE
         // <summary>
         void GPIO_MODE()
        {

          byte regVal = readRegister(byte(0xBE));
          byte temp = regVal;
          regVal = regVal & 0x0B;
          regVal = regVal | 0x06;
          writeRegister(byte (0xBE), byte (regVal));
                           
        }

         // <summary>
         //CONFIG (0x70) is used to set the INT_MODE (Bit 1:0) SYNSis set by writing 0x01 and SYND by  writing 0x03
         // <summary>
         void INT_MODE(byte mode)
         {

          byte regVal = readRegister(byte(0x70));
          byte temp = regVal;
          regVal = regVal & 0xFC;
          regVal = regVal | mode;
          writeRegister(byte (0x70), byte (regVal));
                           
        }

        // <summary>
        // select which Register bank to address in registers 0x00-0x7f has priority over ram_bank 
        //reg_bank bit(4) is set to '1' for setting the 0x00-0x7f regiater to reg_bank register. by default it is RAM register 
        // <summary>
        void RegBankConfig()
        {

          byte regVal = readRegister(byte(0xA9));
          byte temp = regVal;
          regVal = regVal & 0xEF;
          regVal = regVal | 0x10;
          writeRegister(byte (0xA9), byte (regVal));
                           
        }


        
/*----- Set integration time = (ATIME + 1) * (ASTEP + 1) * 2.78µS -----*/


        //<summary>
        // Sets the ATIME for integration time from 0 to 255, integration time = (ATIME + 1) * (ASTEP + 1) * 2.78µS
        //<summary>
        // param name = "value"> integer value from 0 to 255 written to ATIME register 0x81     
        void setATIME(byte value)
          {
          
          writeRegister(byte (0x81), value);
         
          }

          
        //<summary>
        // Sets the ASTEP for integration time from 0 to 65535, integration time = (ATIME + 1) * (ASTEP + 1) * 2.78µS
        //<summary>
        // param name = "value1,"> Defines the lower byte[7:0] of the base step time written to ASTEP register 0xCA
        // param name = "value2,"> Defines the higher byte[15:8] of the base step time written to ASTEP register 0xCB
        void setASTEP(byte value1, byte value2)
          {
          
            // astep[7:0]
            writeRegister(byte (0xCA), value1);
            
            // astep[15:8]
            writeRegister(byte (0xCB), value2);
          
          }


        //<summary>
        // Sets the Spectral Gain in CFG1 Register (0xAA) in [4:0] bit
        //<summary>
        // param name = "value"> integer value from 0 to 10 written to AGAIN register 0xAA
        void setGAIN(byte value)
          {
            writeRegister(byte (0xAA), value);
          }
        
          
/*----- SMUX Configuration for F1,F2,F3,F4,CLEAR,NIR -----*/

        //<summary>
        // Mapping the individual Photo diodes to dedicated ADCs using SMUX Configuration for F1-F4,Clear,NIR
        //<summary> 
        
        void F1F4_Clear_NIR()
        {
          //SMUX Config for F1,F2,F3,F4,NIR,Clear
          writeRegister(byte (0x00), byte (0x30)); // F3 left set to ADC2
          writeRegister(byte (0x01), byte (0x01)); // F1 left set to ADC0
          writeRegister(byte (0x02), byte (0x00)); // Reserved or disabled
          writeRegister(byte (0x03), byte (0x00)); // F8 left disabled
          writeRegister(byte (0x04), byte (0x00)); // F6 left disabled
          writeRegister(byte (0x05), byte (0x42)); // F4 left connected to ADC3/f2 left connected to ADC1
          writeRegister(byte (0x06), byte (0x00)); // F5 left disbled
          writeRegister(byte (0x07), byte (0x00)); // F7 left disbled
          writeRegister(byte (0x08), byte (0x50)); // CLEAR connected to ADC4
          writeRegister(byte (0x09), byte (0x00)); // F5 right disabled
          writeRegister(byte (0x0A), byte (0x00)); // F7 right disabled
          writeRegister(byte (0x0B), byte (0x00)); // Reserved or disabled
          writeRegister(byte (0x0C), byte (0x20)); // F2 right connected to ADC1
          writeRegister(byte (0x0D), byte (0x04)); // F4 right connected to ADC3
          writeRegister(byte (0x0E), byte (0x00)); // F6/F7 right disabled
          writeRegister(byte (0x0F), byte (0x30)); // F3 right connected to AD2
          writeRegister(byte (0x10), byte (0x01)); // F1 right connected to AD0
          writeRegister(byte (0x11), byte (0x50)); // CLEAR right connected to AD4
          writeRegister(byte (0x12), byte (0x00)); // Reserved or disabled
          writeRegister(byte (0x13), byte (0x06)); // NIR connected to ADC5
        }


/*----- SMUX Configuration for F5,F6,F7,F8,CLEAR,NIR -----*/

        //<summary>
        // Mapping the individual Photo diodes to dedicated ADCs using SMUX Configuration for F5-F8,Clear,NIR
        //<summary> 
        
        void F5F8_Clear_NIR()
        {
          //SMUX Config for F5,F6,F7,F8,NIR,Clear
          writeRegister(byte (0x00), byte (0x00)); // F3 left disable
          writeRegister(byte (0x01), byte (0x00)); // F1 left disable
          writeRegister(byte (0x02), byte (0x00)); // reserved/disable
          writeRegister(byte (0x03), byte (0x40)); // F8 left connected to ADC3
          writeRegister(byte (0x04), byte (0x02)); // F6 left connected to ADC1
          writeRegister(byte (0x05), byte (0x00)); // F4/ F2 disabled
          writeRegister(byte (0x06), byte (0x10)); // F5 left connected to ADC0
          writeRegister(byte (0x07), byte (0x03)); // F7 left connected to ADC2
          writeRegister(byte (0x08), byte (0x50)); // CLEAR Connected to ADC4
          writeRegister(byte (0x09), byte (0x10)); // F5 right connected to ADC0
          writeRegister(byte (0x0A), byte (0x03)); // F7 right connected to ADC2
          writeRegister(byte (0x0B), byte (0x00)); // Reserved or disabled
          writeRegister(byte (0x0C), byte (0x00)); // F2 right disabled
          writeRegister(byte (0x0D), byte (0x00)); // F4 right disabled
          writeRegister(byte (0x0E), byte (0x24)); // F7 connected to ADC2/ F6 connected to ADC1
          writeRegister(byte (0x0F), byte (0x00)); // F3 right disabled
          writeRegister(byte (0x10), byte (0x00)); // F1 right disabled
          writeRegister(byte (0x11), byte (0x50)); // CLEAR right connected to AD4
          writeRegister(byte (0x12), byte (0x00)); // Reserved or disabled
          writeRegister(byte (0x13), byte (0x06)); // NIR connected to ADC5
        }
























          
          /* ----- Read/Write to i2c register ----- */

      // <summary>  
      //Read a single i2c register
      // <summary>
      // param name = "addr">Register address of the the register to be read
      // param name = "_i2cAddr">Device address 0x39
      
      byte readRegister(byte addr)
       {
            Wire.beginTransmission(_i2cAddr);
            Wire.write(addr);
            Wire.endTransmission();
              
            Wire.requestFrom(_i2cAddr, 1);
            
            if (Wire.available()) 
                {
                  //Serial.println(Wire.read());
                  return (Wire.read());
                }
       
            else 
               {
                 Serial.println("I2C Error");
                 return (0xFF); //Error
               }
       }

 void readRegisterPrint(byte addr)
       {
            Wire.beginTransmission(_i2cAddr);
            Wire.write(addr);
            Wire.endTransmission();
              
            Wire.requestFrom(_i2cAddr, 1);
            
            if (Wire.available()) 
                {
                  Serial.println(Wire.read());
                  //return (Wire.read());
                }
       
            else 
               {
                 Serial.println("I2C Error");
                 //return (0xFF); //Error
               }
       }
       // <summary>  
      // Read two consecutive i2c registers
      // <summary>
      // param name = "addr">First register address of two consecutive registers to be read
      // param name = "_i2cAddr">Device address 0x39
         
      uint16_t readTwoRegister1(byte addr)
          {
            uint8_t readingL; uint16_t readingH; uint16_t reading = 0; 
            Wire.beginTransmission(_i2cAddr);
            Wire.write(addr);
            Wire.endTransmission();
        
            Wire.requestFrom(_i2cAddr, 2);
          
            if (2<=Wire.available()) 
              {
              readingL = Wire.read();
              readingH = Wire.read();
              readingH = readingH << 8;
              reading = (readingH | readingL);
              return(reading);
              }
            else 
              {
              Serial.println("I2C Error");
              return (0xFFFF); //Error
              }
          }

        
      // <summary>  
      // Write a value to a single i2c register
      // <summary>
      // param name = "addr">Register address of the the register to the value to be written
      // param name = "val">The value written to the Register
      // param name = "_i2cAddr">Device address 0x39
    
      void writeRegister(byte addr, byte val)
       {
          Wire.beginTransmission(_i2cAddr);
          Wire.write(addr);
          Wire.write(val);
          Wire.endTransmission();
        }
        
   
