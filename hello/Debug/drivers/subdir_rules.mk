################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Each subdirectory must supply rules for building sources it contributes
drivers/rtos_hw_drivers.obj: /home/jao/curso/ufsm/Mestrado/Dissertacao/Pototip_DeVereda/ProjetoPrototipo/Tiva/TM4C/examples/boards/ek-tm4c123gxl/drivers/rtos_hw_drivers.c $(GEN_OPTS) | $(GEN_FILES) $(GEN_MISC_FILES)
	@echo 'Building file: "$<"'
	@echo 'Invoking: ARM Compiler'
	"" -mv7M4 --code_state=16 --float_support=FPv4SPD16 -me -Ooff --include_path="/include" --include_path="/home/jao/workspace_v10/hello/hellow_dir" --include_path="/home/jao/Documentos/tiva_test/hello" --include_path="/home/jao/curso/ufsm/Mestrado/Dissertacao/Pototip_DeVereda/ProjetoPrototipo/Tiva/TM4C/examples/boards/ek-tm4c123gxl" --include_path="/home/jao/curso/ufsm/Mestrado/Dissertacao/Pototip_DeVereda/ProjetoPrototipo/Tiva/TM4C" --include_path="/home/jao/curso/ufsm/Mestrado/Dissertacao/Pototip_DeVereda/ProjetoPrototipo/Tiva/TM4C/third_party" --include_path="/home/jao/curso/ufsm/Mestrado/Dissertacao/Pototip_DeVereda/ProjetoPrototipo/Tiva/TM4C/third_party/FreeRTOS/Source/include" --include_path="/home/jao/curso/ufsm/Mestrado/Dissertacao/Pototip_DeVereda/ProjetoPrototipo/Tiva/TM4C/third_party/FreeRTOS" --include_path="/home/jao/curso/ufsm/Mestrado/Dissertacao/Pototip_DeVereda/ProjetoPrototipo/Tiva/TM4C/third_party/FreeRTOS/Source/portable/CCS/ARM_CM4F" --define=ccs="ccs" --define=TARGET_IS_TM4C123_RB1 --define=PART_TM4C123GH6PM -g --gcc --diag_warning=225 --diag_wrap=off --display_error_number --gen_func_subsections=on --abi=eabi --ual --preproc_with_compile --preproc_dependency="drivers/$(basename $(<F)).d_raw" --obj_directory="drivers" $(GEN_OPTS__FLAG) "$(shell echo $<)"
	@echo 'Finished building: "$<"'
	@echo ' '


