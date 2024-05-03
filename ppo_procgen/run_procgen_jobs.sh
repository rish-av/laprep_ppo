PYTHON_SCRIPTS=("train_lap_reg.py" "train_feature_fusion.py")
ENV_NAMES=("climber" "starpilot" "coinrun")
SEEDS=(42 64 81 89 57)

for SCRIPT in "${PYTHON_SCRIPTS[@]}"; do
    for ENV_NAME in "${ENV_NAMES[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            # Create a temporary sbatch file
            TEMP_SBATCH="temp_${SCRIPT}_${ENV_NAME}_${SEED}.sh"
            cp procgen_job_template.sh "${TEMP_SBATCH}"

            # Replace placeholders in the SBATCH script
            sed -i "s|\"\${SCRIPT}\"|${SCRIPT}|g" "${TEMP_SBATCH}"
            sed -i "s|\"\${ENV_NAME}\"|${ENV_NAME}|g" "${TEMP_SBATCH}"
            sed -i "s|\"\${SEED}\"|${SEED}|g" "${TEMP_SBATCH}"

            # Submit the job
            sbatch "${TEMP_SBATCH}"

            # Optionally, remove the temp file after submission
            rm "${TEMP_SBATCH}"
        done
    done
done

echo "All jobs have been submitted."
