  /*  This is an example code written for the AS7241 XWing Spectral Sensor I2C interface with Arduino µC for Flicker Detection.
   *   Script to detect 1.0kHz and 1.2kHz flicker - sample frequency fs = 5538Hz
   *  
   *  Written by Sijo John @ ams AG, Application Support in November, 2018
   *  Development environment specifics: Arduino IDE 1.8.5
   *  This program is distributed in the hope that it will be useful,
      but WITHOUT ANY WARRANTY; without even the implied warranty of
      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
   */
  
  
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

             FlickerDetection();
      
            // reading the flicker status in FD_STATUS register 0xDB
            // data=0x2c=0b00101100  FD_STATUS(fd_measurement_valid=1 fd_120Hz_flicker_valid=1 fd_100Hz_flicker_valid=1)
            // data=0x2d=0b00101101  FD_STATUS(fd_measurement_valid=1 fd_1200Hz_flicker_valid=1 fd_1000Hz_flicker_valid=1 fd_1000Hz_flicker)
            // data=0x2e=0b00101110  FD_STATUS(fd_measurement_valid=1 fd_1200Hz_flicker_valid=1 fd_1000Hz_flicker_valid=1 fd_1200Hz_flicker)
            
             int flicker_value = readRegister(byte(0xDB));
             Serial.print("Flicker Status-");
             Serial.println(flicker_value); 
             
             if(flicker_value == 44)
              {
                Serial.println("Unknown frequency");
              }
             else if(flicker_value == 45)
              {
                Serial.println("1000 Hz detected");
              }
             else if(flicker_value == 46)
              {
                Serial.println("1200 Hz detected");
              }
             else 
              {
                Serial.println("Error in reading");
              }
      
          }
      // <summary>  
      // To detect the target frequency of 1kHz and further 1.2kHz Flickering
      // <summary>  
      void FlickerDetection()
        {
  
          //RAM_BANK 0 select which RAM bank to access in register addresses 0x00-0x7f
          writeRegister(byte (0xA9), byte(0x00));

          //The coefficient calculated are stored into the RAM bank 0 and RAM bank 1, they are used instead of 100Hz and 120Hz coefficients which are the default flicker detection coefficients 
          // write new coefficients to detect the 1000Hz and 1200Hz - part 1
          writeRegister(byte (0x04), byte(0x9E));
          writeRegister(byte (0x05), byte(0x36));
          writeRegister(byte (0x0E), byte(0x2E));
          writeRegister(byte (0x0F), byte(0x1B));
          writeRegister(byte (0x18), byte(0x7D));
          writeRegister(byte (0x19), byte(0x36));
          writeRegister(byte (0x22), byte(0x09));
          writeRegister(byte (0x23), byte(0x1B));
          writeRegister(byte (0x2C), byte(0x5B));
          writeRegister(byte (0x2D), byte(0x36));
          writeRegister(byte (0x36), byte(0xE5));
          writeRegister(byte (0x37), byte(0x1A));
          writeRegister(byte (0x40), byte(0x3A));
          writeRegister(byte (0x41), byte(0x36));
          writeRegister(byte (0x4A), byte(0xC1));
          writeRegister(byte (0x4B), byte(0x1A));
          writeRegister(byte (0x54), byte(0x18));
          writeRegister(byte (0x55), byte(0x36));
          writeRegister(byte (0x5E), byte(0x9C));  
          writeRegister(byte (0x5F), byte(0x1A));
          writeRegister(byte (0x68), byte(0xF6));
          writeRegister(byte (0x69), byte(0x35));
          writeRegister(byte (0x72), byte(0x78));
          writeRegister(byte (0x73), byte(0x1A));
          writeRegister(byte (0x7C), byte(0x4D));
          writeRegister(byte (0x7D), byte(0x35));

          //RAM_BANK 1 select which RAM bank to access in register addresses 0x00-0x7f
          writeRegister(byte (0xA9), byte(0x01));

          // write new coefficients to detect the 1000Hz and 1200Hz - part 1
          writeRegister(byte (0x06), byte(0x54));
          writeRegister(byte (0x07), byte(0x1A));
          writeRegister(byte (0x10), byte(0xB3));
          writeRegister(byte (0x11), byte(0x35));
          writeRegister(byte (0x1A), byte(0x2F));
          writeRegister(byte (0x1B), byte(0x1A));


          writeRegister(byte (0xA9), byte(0x01));

          //select RAM coefficients for flicker detection by setting fd_disable_constant_init to „1“ (FD_CFG0 register)
          //in FD_CFG0 register - 0xd7  fd_disable_constant_init=1 fd_samples=4
          writeRegister(byte (0xD7), byte(0x60));
          //readRegisterPrint(byte(0xD7));

          //in FD_CFG1 register - 0xd8 fd_time(7:0) = 0x40
          writeRegister(byte (0xD8), byte(0x40));
          //readRegisterPrint(byte(0xD8));


          // in FD_CFG2 register - 0xd9  fd_dcr_filter_size=1 fd_nr_data_sets(2:0)=5
          writeRegister(byte (0xD9), byte(0x25));
          //readRegisterPrint(byte(0xD9));

          // in FD_CFG3 register - 0xda fd_gain=9
          writeRegister(byte (0xDA), byte(0x48));
          //readRegisterPrint(byte(0xDA));

          // in CFG9 register - 0xb2 sien_fd=1 
          writeRegister(byte (0xB2), byte(0x40));
          //readRegisterPrint(byte(0xB2));


          // in ENABLE - 0x80  fden=1 and pon=1 are enabled 
          writeRegister(byte (0x80), byte(0x41));
          //readRegisterPrint(byte(0x80));

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
        
