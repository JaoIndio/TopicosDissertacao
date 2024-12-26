################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Each subdirectory must supply rules for building sources it contributes
SpecLibs/DRV8825/%.obj: ../SpecLibs/DRV8825/%.c $(GEN_OPTS) | $(GEN_FILES) $(GEN_MISC_FILES)
	@echo 'Building file: "$<"'
	@echo 'Invoking: ARM Compiler'
	"/home/jao/ti/ccs1271/ccs/tools/compiler/ti-cgt-arm_20.2.1.LTS/bin/armcl" -mv7M4 --code_state=16 --float_support=FPv4SPD16 -me -O2 --fp_mode=strict --include_path="/home/jao/Documentos/tiva_test/Nema17" --include_path="/home/jao/Documentos/tiva_test/arm_dsp/CMSIS_4/CMSIS/Include" --include_path="/home/jao/Documentos/tiva_test/Nema17/SpecLibs" --include_path="/home/jao/Documentos/tiva_test/Nema17/FreeRTOS/Source/portable/MemMang" --include_path="/home/jao/Documentos/tiva_test/Nema17/FreeRTOS/Source/include" --include_path="/home/jao/Documentos/tiva_test/Nema17/FreeRTOS/Source" --include_path="/home/jao/Documentos/tiva_test/Nema17/FreeRTOS/Source/portable/CCS/ARM_CM4F" --include_path="/home/jao/curso/ufsm/Mestrado/Dissertacao/Pototip_DeVereda/ProjetoPrototipo/Tiva/EK-TM4C" --include_path="/home/jao/ti/ccs1271/ccs/tools/compiler/ti-cgt-arm_20.2.1.LTS/include" --define=ccs="ccs" --define=__FPU_PRESENT=1 --define=__GNUC_=1 --define=TARGET_IS_TM4C123_RB1 --define=PART_TM4C123GH6PM --define=ARM_MATH_CM4 -g --gcc --diag_warning=225 --diag_wrap=off --display_error_number --gen_func_subsections=on --abi=eabi --preproc_with_compile --preproc_dependency="SpecLibs/DRV8825/$(basename $(<F)).d_raw" --obj_directory="SpecLibs/DRV8825" $(GEN_OPTS__FLAG) "$(shell echo $<)"
	@echo 'Finished building: "$<"'
	@echo ' '


