################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Each subdirectory must supply rules for building sources it contributes
third_party/FreeRTOS/Source/portable/CCS/ARM_CM4F/%.obj: ../third_party/FreeRTOS/Source/portable/CCS/ARM_CM4F/%.c $(GEN_OPTS) | $(GEN_FILES) $(GEN_MISC_FILES)
	@echo 'Building file: "$<"'
	@echo 'Invoking: ARM Compiler'
	"/home/jao/ti/ccs1010/ccs/tools/compiler/ti-cgt-arm_20.2.1.LTS/bin/armcl" -mv7M4 --code_state=16 --float_support=FPv4SPD16 -me --include_path="/home/jao/workspace_v10/Nema17" --include_path="/home/jao/workspace_v10/Nema17/third_party/FreeRTOS/Source/include" --include_path="/home/jao/workspace_v10/Nema17/third_party/FreeRTOS/Source/portable/CCS/ARM_CM4F" --include_path="/home/jao/workspace_v10/Nema17/third_party/FreeRTOS/Source/portable/MemMang" --include_path="/home/jao/curso/ufsm/Mestrado/Dissertacao/Pototip_DeVereda/ProjetoPrototipo/Tiva/EK-TM4C" --include_path="/home/jao/ti/ccs1010/ccs/tools/compiler/ti-cgt-arm_20.2.1.LTS/include" --define=ccs="ccs" --define=TARGET_IS_TM4C123_RB1 --define=PART_TM4C123GH6PM -g --gcc --diag_warning=225 --diag_wrap=off --display_error_number --abi=eabi --preproc_with_compile --preproc_dependency="third_party/FreeRTOS/Source/portable/CCS/ARM_CM4F/$(basename $(<F)).d_raw" --obj_directory="third_party/FreeRTOS/Source/portable/CCS/ARM_CM4F" $(GEN_OPTS__FLAG) "$(shell echo $<)"
	@echo 'Finished building: "$<"'
	@echo ' '

third_party/FreeRTOS/Source/portable/CCS/ARM_CM4F/%.obj: ../third_party/FreeRTOS/Source/portable/CCS/ARM_CM4F/%.asm $(GEN_OPTS) | $(GEN_FILES) $(GEN_MISC_FILES)
	@echo 'Building file: "$<"'
	@echo 'Invoking: ARM Compiler'
	"/home/jao/ti/ccs1010/ccs/tools/compiler/ti-cgt-arm_20.2.1.LTS/bin/armcl" -mv7M4 --code_state=16 --float_support=FPv4SPD16 -me --include_path="/home/jao/workspace_v10/Nema17" --include_path="/home/jao/workspace_v10/Nema17/third_party/FreeRTOS/Source/include" --include_path="/home/jao/workspace_v10/Nema17/third_party/FreeRTOS/Source/portable/CCS/ARM_CM4F" --include_path="/home/jao/workspace_v10/Nema17/third_party/FreeRTOS/Source/portable/MemMang" --include_path="/home/jao/curso/ufsm/Mestrado/Dissertacao/Pototip_DeVereda/ProjetoPrototipo/Tiva/EK-TM4C" --include_path="/home/jao/ti/ccs1010/ccs/tools/compiler/ti-cgt-arm_20.2.1.LTS/include" --define=ccs="ccs" --define=TARGET_IS_TM4C123_RB1 --define=PART_TM4C123GH6PM -g --gcc --diag_warning=225 --diag_wrap=off --display_error_number --abi=eabi --preproc_with_compile --preproc_dependency="third_party/FreeRTOS/Source/portable/CCS/ARM_CM4F/$(basename $(<F)).d_raw" --obj_directory="third_party/FreeRTOS/Source/portable/CCS/ARM_CM4F" $(GEN_OPTS__FLAG) "$(shell echo $<)"
	@echo 'Finished building: "$<"'
	@echo ' '


