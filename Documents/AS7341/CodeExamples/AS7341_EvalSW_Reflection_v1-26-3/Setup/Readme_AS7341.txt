*******************************************************************************

AS7341 EvalSw Software

*******************************************************************************

Thank you for choosing AS7341 Evaluation Board. 

For details in using the Eval Board please refer to the User’s Guide included in the documents. 

If any technical difficulties are encountered, use the Tech Support page at www.ams.com/Support to submit a technical support request anytime or call ams sales FAE. 

You may also use www.ams.com to find worldwide local representatives in your area. 

Use the AS7341 setup to install all necessary software and driver on your Windows 10 based PC. 

Make sure, the following tools are installed on your PC:
1. Microsoft .NET Framework 4.5.2 (x86 and x64) (see directory DotNetFX452)
2. Visual C++ "14" Runtime Libraries (x86) (see directory vcredist_x86)
3. FTDI USB-MPSSE Cable Driver (see directory AS726x FTDI USB-MPSSE Cable Driver)

Please see in our manual for the details how the sensor board must be connected. All technical documents are in the directory ‘Documents’.




Please, note the following tips for the current release:

De-install former existing AS7341 Eval Kit software from your PC before installation and using the new setup. 

Make sure to use the new GUI and the standard initialization files only with a AS7341 Eval Kit without add-one IR-blocking.

Install the GUI at first before you copy (and replace) specific initialization files in the working directory.

The standard working directory for the GUI is C:\Users\xxxx\AppData\Roaming\ams-OSRAM\AS7341 EvalSw.

The GUI works with the Standard EVKs in combination with following initialization files:

1. init_file.txt - necessary for all GUI functions.
   It includes all-important values to control functions and limitations, e.g. sensor signal correction, light detection, reflective mode.
   We recommend making no changings in this file.

2. CM_L1_v1_0_0xxx.csv - includes the correction matrix for light detection.
   The matrix is a general solution and can be optimized by a matrix based on a device-to-device calibration.
   Note, a changing of the matrix can affect the sensor accuracy or following CIE1931 functions. The dimensions of the matrix determine
   direct following dimensions of alternative matrices and mask files.

3. CM_R_v1_0_0xxxxx.csv - includes the correction matrix for reflected mode as typical example. Several versions are included to show alternative 
   calibration methods. Note, calibration file must be adapted to the specific EVK to get acuracy. For more details, see (2.)

4. Mask_L_v1_0_0.csv - is a start mask file for spectral comparison. 
   It can be changed or created by the GUI. Do not use an older Mask file in combination with this new GUI. We recommend deleting older files.
   Use only the file that was installed by the GUI and/or make a new Mask file in combination with the actual correction matrix.

5. Mask_R_v1_0_0.csv - is a start mask file for reflected mode. For more details, see (4.)

After installation, connect the hardware via USB to PC. Place the sensor to a light source with a known spectrum.
Select TINT = 182ms and a values for gain to get highest signals for all channels before saturation.
Select an Application Specific Function like Light detection or Flicker to test the new functions for GUI.



For more details, please see the manuals or contact us.


*******************************************************************************

Version History

*******************************************************************************

Version     1.26.3
Date        14.06.2022

- Save offset / white balance values renamed to set
- Context menu entry write ini files moved down
- Warning added on writing ini files if offset / white balance values are set

*******************************************************************************

Version     1.26.2
Date        03.06.2022

- Wavelength values removed from spectrum plot x axis

*******************************************************************************

Version     1.26.1
Date        28.03.2022

- Script editor disable added
- Bug fix in script processing LED handling
- Bug fix max TInt changed to 46602

*******************************************************************************

Version     1.26.0
Date        03.03.2022

- Signed release version

*******************************************************************************

Version     1.25.4
Date        24.02.2022

- Temperature compensation file added to initialization file
- Temperature compensation file added to EEPROM
- Bug fix in measurement / dialog synchronization
- Bug fix in context menu handling
- Bug fix in EEPROM reinitialization
- Bug fix in FTDI register communication

*******************************************************************************

Version     1.25.3
Date        15.02.2022

- Bug fix in continuous measurement / script processing temperature measurement
- Bug fix in script processing command readTime

*******************************************************************************

Version     1.25.2
Date        24.01.2022

- Max loop count in script processing changed to 100000

*******************************************************************************

Version     1.25.1
Date        19.01.2022

- Loop count added to script processing

*******************************************************************************

Version     1.25.0
Date        08.12.2021

- Signed release version

*******************************************************************************

Version     1.24.3
Date        03.12.2021

- About dialog changed
- PAR and PPF added to logging

*******************************************************************************

Version     1.24.2
Date        29.11.2021

- PAR and PPF calculation added

*******************************************************************************

Version     1.24.0
Date        11.11.2021

- Signed release version
- Reconnect with last sensor after EEPROM change added

*******************************************************************************

Version     1.23.0
Date        29.10.2021

- Signed release version
- Flicker removed from reflection app
- Bug fix in gain optimization
- Bug fix in thread management
- Bug fix in backside compensation
- Bug fix in script measurement

*******************************************************************************

Version     1.22.0
Date        09.09.2021

- Signed release version
- Name changed from AS7341 Demo to AS7341 EvalSw
- Context menu for Ctrl commands added
- Hot key Ctrl+P for EEPROM reinitialization added

*******************************************************************************

Version     1.21.5
Date        19.08.2021

- Manufacturer name including folder names changed from ams AG to ams-OSRAM
- Bug fix in EEPROM data

*******************************************************************************

Version     1.21.4
Date        23.07.2021

- Global channel order added
- Calibration matrix correction values only used if set
- Local correction added to EEPROM
- Commands maxAutoGain and maxAutoTint added to script processing
- Saturation logging changed

*******************************************************************************

Version     1.21.3
Date        15.07.2021

- Bug fix in offset measurement
- Auto gain calculation changed
- Spectrum dialog changed

*******************************************************************************

Version     1.21.1
Date        23.06.2021

- Spectral library version 1.2.1 used
- CCT value added to mask file
- Command writeLine added to script processing
- Command setComment added to script processing
- Commands beginLoop and endLoop added to script processing
- Command backSideComp added to script processing
- Bug fix in local correction

*******************************************************************************

Version     1.21.0
Date        21.05.2021

- Signed release version

*******************************************************************************

Version     1.20.12
Date        20.05.2021

- Bug fix in saturation handling

*******************************************************************************

Version     1.20.11
Date        19.05.2021

- Bug fix in spectrum and line graph dialog handling
- Bug fix in saturation handling

*******************************************************************************

Version     1.20.10
Date        19.05.2021

- Bug fix in EEPROM data reading

*******************************************************************************

Version     1.20.9
Date        18.05.2021

- Spectrum normalization changed
- Bug fix in correction gain handling
- Bug fix in spectrum dialog update
- MaxAutoGain handling changed

*******************************************************************************

Version     1.20.8
Date        06.05.2021

- Integration time optimization changed

*******************************************************************************

Version     1.20.7
Date        06.05.2021

- Configuration of using low raw values for optimization added

*******************************************************************************

Version     1.20.6
Date        05.05.2021

- Bug fix in logging saturation handling
- Tracer command autoTint added

*******************************************************************************

Version     1.20.5
Date        04.05.2021

- Gain correction factor matrix EEPROM support added

*******************************************************************************

Version     1.20.4
Date        29.04.2021

- Gain correction factor matrix added

*******************************************************************************

Version     1.20.3
Date        22.04.2021

- Display of estimated measurement time added
- Configuration of empty EEPROM dialog added
- Measurement thread added
- Tracer thread added

*******************************************************************************

Version     1.20.2
Date        13.04.2021

- Macros added to mask files
- Trace added to spectrum graphic
- Bug fix in EEPROM size detection

*******************************************************************************

Version     1.20.1
Date        11.03.2021

- EEPROM size detection added
- Bug fix in mask file EEPROM handling

*******************************************************************************

Version     1.19.10
Date        19.02.2021

- Signed release version

*******************************************************************************

Version     1.19.6
Date        17.02.2021

- VIS and NIR measurement configuration added

*******************************************************************************

Version     1.19.5
Date        16.02.2021

- Restart after EEPROM initialization and clear
- Reconnect after EEPROM disable
- Configuration of VIS / NIR in correction factor and offset added

*******************************************************************************

Version     1.19.4
Date        05.02.2021

- Configuration of VIS / NIR in reflection calibration matrix added

*******************************************************************************

Version     1.19.3
Date        03.02.2021

- Configuration of VIS / NIR in calibration matrix added

*******************************************************************************

Version     1.19.2
Date        29.01.2021

- Bug fix in set gain function
- Bug fix in register map

*******************************************************************************

Version     1.19.1
Date        27.01.2021

- Bug fix in set gain function
- Bug fix in continuous measurement
- Bug fix in spectrum graph x axis scaling

*******************************************************************************

Version     1.19.0
Date        21.01.2021

- AS7352 support added

*******************************************************************************

Version     1.18.601
Date        11.12.2020

- Aqua sensor support added

*******************************************************************************

Version     1.18.506
Date        05.11.2020

- Show selected mask in color space
- Don't reset zoom after measurement

*******************************************************************************

Version     1.18.505
Date        29.10.2020

- Bug fix in Luv calculation
- Bug fix in mask spectrum display
- Bug fix in spectrum min normalization
- Bug fix in mask compare
- Write register function removed

*******************************************************************************

Version     1.18.503
Date        29.09.2020

- Bug fix in EEPROM calibration matrix handling

*******************************************************************************

Version     1.18.502
Date        25.09.2020

- Local correction for XYZ calibration matrix added
- Bug fix in spectrum output shift
- Correction gain values changed

*******************************************************************************

Version     1.18.501
Date        18.09.2020

- Only positive spectrum output
- Bug fix in min StDev(StDev) calculation
- Bug fix in light detection color point mask compare
- Bug fix in tracer gain command
- Unique chip id format changed
- Wavelength values changed

*******************************************************************************

Version     1.18.500
Date        26.08.2020

- Signed release version
- AS7351 support added
- Bug fix in overwriting existing mask

*******************************************************************************

Version     1.18.406
Date        12.08.2020

- Bug fix in Luv calculation

*******************************************************************************

Version     1.18.405
Date        30.07.2020

- Bug fix in mask compare standard deviation calculation

*******************************************************************************

Version     1.18.402
Date        27.07.2020

- AS7342 support added
- Mask compare based on standard deviation ** 2 added
- Bug fix in EEPROM data version management

*******************************************************************************

Version     1.18.401
Date        22.06.2020

- Mask compare based on standard deviation
- Min normalization added
- Plot spectrum of selected mask added

*******************************************************************************

Version     1.18.400
Date        14.05.2020

- Signed release version
- Tracer command to open measurement dialog added
- Hot key Ctrl+C for abort script added
- Display of color coordinate error added
- EEPROM version check on saving initialization data
- Bug fix in offset measurement
- Bug fix in mask management

*******************************************************************************

Version     1.18.21
Date        06.05.2020

- Tracer updated
- Mask CCT calculation added
- Bug fix in zoom for continuous measurement
- Bug fix in register bank management
- Bug fix in NIR correction for zero values
- Hot key Ctrl+B for saving white balance values added
- Installer for ALS and reflection kits

*******************************************************************************

Version     1.18.20
Date        21.04.2020

- Calibration matrix wavelength step width added to XYZ calculation
- Bug fix in reflection white balance management
- Show reflection mask compare even if spectrum is displayed

*******************************************************************************

Version     1.18.19
Date        17.04.2020

- Bug fix in offset measurement and saving management
- Hot key Ctrl+V to remove version information from main window title bar added
- Show spectrum for spectral reflection calibration matrix
- Measured and mask RGB values added to debug log csv file

*******************************************************************************

Version     1.18.200
Date        17.04.2020

- Signed release version

*******************************************************************************

Version     1.18.100
Date        16.04.2020

- Signed release version

*******************************************************************************

Version     1.18.18
Date        08.04.2020

- Norm light source added to XYZ calculation for reflection

*******************************************************************************

Version     1.18.17
Date        08.04.2020

- Scroll bars added to main and spectrum dialog
- Hot key Ctrl+O for measurement of offset values added
- Hot key Ctrl+I for write init data from memory to EEPROM or init file added
- Yuv changed to Luv
- Tool tip added to blue light ratio
- Bug fix in handling of all masks deleted
- Retry added to save mask file to EEPROM
- Calculation of required EEPROM size added
- Bug fix in Luv calculation
- Order of corrected values calculation changed
- XYZ to RGB conversion matrix added to reflection calibration matrix
- Spectral calibration matrix for reflection support added

*******************************************************************************

Version     1.18.16
Date        01.04.2020

- Max AGAIN and TInt added to init file and calibration matrix file
- Temperature compensation added
- Blue light ratio added to light detection
- Zoom added to graphics
- Deletion of multiple masks added
- Trace added to color space diagram
- EEPROM errors added to EEPROM log
- Display of measured and detected color added to reflection dialog
- Backside compensation added

*******************************************************************************

Version     1.18.15
Date        16.03.2020

- AS7350 support added

*******************************************************************************

Version     1.18.14
Date        11.03.2020

- Mask detection for light detection color space based on delta xy / delta uv / delta Yuv
- Mask detection for reflection based on delta xy / delta ab / delta Lab

*******************************************************************************

Version     1.18.13
Date        05.03.2020

- Mask detection for reflection based on delta Yxy / delta xy
- Limit_Delta_ab changed to Limit_Delta_xy in ini file
- Automatic TInt changed to get max TInt

*******************************************************************************

Version     1.18.12
Date        02.03.2020

- Display of Delta uv, Delta Y(lx), Delta CCT for light detection added
- Hot key Ctrl+F for EEPROM clear added
- Bug fix in extension LED setting
- Logging for mask compare and save extended
- Hot key Ctrl+L for save log files added

*******************************************************************************

Version     1.18.11
Date        20.12.2019

- Bug fix in basic value calculation
- Debug log file for mask read / write added
- Bug fix in GUI layout

*******************************************************************************

Version     1.18.10
Date        12.12.2019

- Handling of extension LEDs changed (at least one LED is on)

*******************************************************************************

Version     1.18.9
Date        11.11.2019

- Light dialog color point values changed from xy to Lu'v'
- Bug fix in handling of existing masks
- Display of flicker saturation error changed
- Bug fix in handling of TInt optimization if reflection is opened

*******************************************************************************

Version     1.18.8
Date        30.10.2019

- Signed release version
- Initialization files installed into user directory
- Version history added to Readme.txt
- Select new mask file with Ctrl+R
- Bug fix in calibration matrix wait time handling

*******************************************************************************

Version     1.18.7
Date        29.10.2019

- CCT calculation with Robertson algorithm
- Calibration matrix / mask file are loaded from ini file if entries are present
- Select new calibration matrix with Ctrl+M
- Optimization and Spectrum Display are switched off in Expert Mode
- New button Delete Mask

*******************************************************************************

Version     1.18.6
Date        21.10.2019

- Mask detection mode min delta ab / min delta L changed
- New temperature display for single temperature sensor

*******************************************************************************

Version     1.18.5
Date        18.10.2019

- Mask detection mode min delta ab / min delta E replaced by min delta ab / min delta L
- Tool tips on graphics changed
- Button Update Mask extended with question dialogs

*******************************************************************************

Version     1.18.4
Date        17.10.2019

- Bug fix in add mask function
- Bug fix in logging wait time

*******************************************************************************

Version     1.18.3
Date        16.10.2019

- Bug fix in eeprom writing in release mode
- Button replace mask added

*******************************************************************************

Version     1.18.2
Date        15.10.2019

- Chip wait time replaced by sw wait time
- Unique chip ID display controlled by config file

*******************************************************************************

Version     1.18.1
Date        08.10.2019

- Unique chip ID and chip time stamp added

*******************************************************************************

Version     1.18.0
Date        17.09.2019

- Release version

*******************************************************************************

Version     1.17.35
Date        10.09.2019

- Configuratio file added

*******************************************************************************

Version     1.17.34
Date        06.09.2019

- Wait time displayed as difference to the integration time
- Valid flags added to logging
- Point marker size in color space graphics increased
- Bug fix in flicker auto gain
- Bug fix in calling light detection / reflection dialog with activated continous measurement

*******************************************************************************

Version     1.17.33
Date        28.08.2019

- Wait time added to calibration matrix
- Error display for wait time changed
- Mask detection mode min delta ab / min delta E added
- Logging file for mask detection mode min delta ab / min delta E is possible

*******************************************************************************

Version     1.17.32
Date        26.08.2019

- Easy / expert mode selection in base window
- Status values, wait time, led current in logging changed
- Reading of float values with , produces a warning

*******************************************************************************
