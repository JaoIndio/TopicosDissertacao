################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Each subdirectory must supply rules for building sources it contributes
DistanceFunctions/%.obj: ../DistanceFunctions/%.c $(GEN_OPTS) | $(GEN_FILES) $(GEN_MISC_FILES)
	@echo 'Building file: "$<"'
	@echo 'Invoking: ARM Compiler'
	"/home/jao/ti/ccs1271/ccs/tools/compiler/ti-cgt-arm_20.2.1.LTS/bin/armcl" -mv7M4 --code_state=32 --float_support=FPv4SPD16 -me -O2 --fp_mode=strict --include_path="/home/jao/Documentos/tiva_test/Nema17" --include_path="/home/jao/Documentos/tiva_test/arm_dsp/CMSIS_6-6.1.0/CMSIS/Core/Include" --include_path="/home/jao/Documentos/tiva_test/arm_dsp/CMSIS-DSP/Include" --include_path="/home/jao/curso/ufsm/Mestrado/Dissertacao/Pototip_DeVereda/ProjetoPrototipo/Tiva/SpecLibs" --include_path="/home/jao/Documentos/tiva_test/Nema17/FreeRTOS/Source/portable/MemMang" --include_path="/home/jao/Documentos/tiva_test/Nema17/FreeRTOS/Source/include" --include_path="/home/jao/Documentos/tiva_test/Nema17/FreeRTOS/Source" --include_path="/home/jao/Documentos/tiva_test/Nema17/FreeRTOS/Source/portable/CCS/ARM_CM4F" --include_path="/home/jao/curso/ufsm/Mestrado/Dissertacao/Pototip_DeVereda/ProjetoPrototipo/Tiva/EK-TM4C" --include_path="/home/jao/ti/ccs1271/ccs/tools/compiler/ti-cgt-arm_20.2.1.LTS/include" --define=ccs="ccs" --define=_FPU_PRESENT=1 --define=_GNUC_=1 --define=TARGET_IS_TM4C123_RB1 --define=PART_TM4C123GH6PM --define=ARM_MATH_CM4 -g --gcc --diag_warning=225 --diag_wrap=off --display_error_number --gen_func_subsections=on --abi=eabi --preproc_with_compile --preproc_dependency="DistanceFunctions/$(basename $(<F)).d_raw" --obj_directory="DistanceFunctions" $(GEN_OPTS__FLAG) "$(shell echo $<)"
	@echo 'Finished building: "$<"'
	@echo ' '


